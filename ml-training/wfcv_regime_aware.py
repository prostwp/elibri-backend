"""wfcv_regime_aware.py — regime-aware walk-forward CV for AutoGluon predictors.

Loop:
    1. Build full labelled feature frame (with optional macro).
    2. Classify each row's regime (bull / chop / bear / warm) using the parent
       regime classifier driven by 1d BTC slope, projected onto the training
       TF via merge_asof.
    3. Pick 5 test windows (one per regime where possible — see
       regime_classifier.pick_five_test_windows). For each window:
         a. Train AutoGluon on EVERYTHING strictly before the window's start.
            (Walk-forward, not full-data — leakage hygiene.)
         b. Evaluate on the window: per-regime metrics, HC precision per
            direction, simple Sharpe-after-fees on signal-thresholded trades.
    4. Report mean ± std across windows + per-regime breakdown.

Output: a single JSON at logs/wfcv_regime_{symbol}_{interval}.json with full
provenance — windows used, metrics per fold, AutoGluon leaderboard per fold.

This is the validator the WFCV-collapse incident from Patch 4 Stage 1 (avg
Sharpe -1.07 across 5 expanding folds, 100% short bias) was missing.

Usage on vast.ai:
    python wfcv_regime_aware.py --symbol BTCUSDT --interval 4h --with-macro \
        --presets medium_quality --time-limit 600

Time-limit applies PER FOLD; 5 folds × 600s = 50 min budget plus AutoGluon
overhead (~10 min). Total ≈ 1 hr per (symbol, interval).
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from feature_engine import (
    FEATURE_NAMES, FEATURE_NAMES_WITH_MACRO,
    build_features, attach_macro_features,
    make_target_triple_class,
    TRIPLE_CLASS_HOLD, TRIPLE_CLASS_LONG, TRIPLE_CLASS_SHORT, TRIPLE_CLASS_UNLABELED,
    _atr,
)
from data_fetcher import fetch_or_cache
from regime_classifier import (
    classify_bars, bucket_windows, pick_five_test_windows, RegimeWindow,
    DEFAULT_BULL_THRESHOLD, DEFAULT_BEAR_THRESHOLD,
)
from train_autogluon import (
    TF_CONFIG, TARGET_COL, build_dataset, train_predictor, evaluate_predictor,
    _validate_symbol, atomic_write_json,
)


ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
WFCV_DIR = MODELS_DIR / "wfcv"
WFCV_DIR.mkdir(exist_ok=True)


# Bars/year per TF for Sharpe annualization. CRYPTO trades 24/7/365 — using
# the equity-market 252 days for 1d (Round-5 analyst find) under-annualizes
# 1d Sharpe by sqrt(252/365) ≈ 0.83 vs other TFs that already use 365.
BARS_PER_YEAR = {
    "5m": 365 * 24 * 12, "15m": 365 * 24 * 4, "1h": 365 * 24,
    "4h": 365 * 6, "1d": 365,
}

# Below this many non-zero positions per fold, Sharpe is dominated by single
# trades (e.g. one big winner in 1k bars produces Sharpe ~600 — meaningless).
# Folds with fewer trades return NaN; aggregator uses np.nanmean.
MIN_TRADES_FOR_SHARPE = 30


@dataclass
class FoldResult:
    fold: int
    window_name: str
    regime: str
    n_train: int
    n_test: int
    test_start: str
    test_end: str
    accuracy: float
    f1_macro: float
    log_loss: float
    sharpe: float            # net of fees + slippage at HC>=0.55 thresholds
    n_long_signals: int
    prec_long_55: float      # precision at threshold 0.55
    n_short_signals: int
    prec_short_55: float
    direction_balance: dict  # {"long_share": x, "short_share": y, "hold_share": z} of signals
    leaderboard_top5: list   # [{model, score_val, score_test}, ...]


# ─── Regime tagging on training-TF bars ──────────────────────────────────────

def attach_regime_to_frame(feat: pd.DataFrame, regime_source_path: Path) -> pd.DataFrame:
    """Reads BTC 1d, classifies, and merge_asof's the regime label onto feat.

    LEAKAGE FIX (2026-04-27): the regime score for `2024-01-15 00:00 UTC`
    1d bar is computed using EMA-200 of closes that include the day's own
    close at `2024-01-15 23:59:59`. When merged onto a 4h bar at e.g.
    `2024-01-15 04:00 UTC`, that 4h bar inherits a regime that "knew" about
    20 hours of future intra-day price action.

    We shift the bar_open_time +1 calendar day before merging, so the 4h
    bar at `2024-01-15 04:00` gets the regime computed from data up through
    `2024-01-14` close — strictly past info.
    """
    btc_df = pd.read_parquet(regime_source_path)
    # Round-7 architecture fix: regime parquet bypassed _validate_ohlcv
    # (which only ran for fetch_or_cache returns). Apply same validation here.
    from data_fetcher import _validate_ohlcv
    _validate_ohlcv(btc_df, f"regime source {regime_source_path.name}")
    # Dedup defensively — duplicate dates (e.g. from incremental refetch
    # overlap) cause non-deterministic regime labels via merge_asof.
    # Round-3 security review.
    if btc_df["open_time"].duplicated().any():
        n_dup = int(btc_df["open_time"].duplicated().sum())
        print(f"  WARN: {regime_source_path.name} had {n_dup} duplicate dates; deduping (keep last)")
        btc_df = (btc_df.sort_values("open_time")
                        .drop_duplicates(subset="open_time", keep="last")
                        .reset_index(drop=True))
    if not btc_df["open_time"].is_monotonic_increasing:
        raise ValueError(
            f"{regime_source_path.name}: open_time not monotonically increasing "
            f"after dedup. Data feed corrupted — investigate before training."
        )
    high = btc_df["high"].to_numpy()
    low = btc_df["low"].to_numpy()
    close = btc_df["close"].to_numpy()
    regimes = classify_bars(high, low, close)

    btc_ts = pd.DatetimeIndex(btc_df["open_time"])
    if btc_ts.tz is None:
        btc_ts = btc_ts.tz_localize("UTC")
    else:
        btc_ts = btc_ts.tz_convert("UTC")
    # Shift +1d so the regime computed from bar `D` (which uses close-of-D)
    # is only AVAILABLE to bars at time `D+1` onward — strict no-look-ahead.
    btc_ts_shifted = btc_ts + pd.Timedelta(days=1)
    rdf = pd.DataFrame({
        "bar_open_time": btc_ts_shifted,
        "regime": regimes,
    })
    feat_ts = pd.DatetimeIndex(feat["open_time"])
    if feat_ts.tz is None:
        feat_ts = feat_ts.tz_localize("UTC")
    # tolerance=2d: if 1d data has a multi-day gap (Binance maintenance,
    # downloader skip), don't keep stale regime forever — fall back to "warm".
    # Round-5 byzantine review.
    merged = pd.merge_asof(
        pd.DataFrame({"bar_open_time": feat_ts}),
        rdf.sort_values("bar_open_time"),
        on="bar_open_time", direction="backward",
        tolerance=pd.Timedelta(days=2),
    )
    feat = feat.copy()
    feat["regime"] = merged["regime"].fillna("warm").to_numpy()
    return feat


# ─── Sharpe with fees (used per fold) ────────────────────────────────────────

def _positions_from_proba(
    proba_3class: np.ndarray,
    hc_long: float = 0.55, hc_short: float = 0.55,
) -> np.ndarray:
    """Single source of truth: 3-class proba → positions ∈ {-1, 0, +1}.

    Both `sharpe_after_fees` and `compute_direction_share` MUST share the same
    rule so reported direction shares match the actual trades. Earlier these
    drifted (compute_direction_share double-counted bars where both p_long and
    p_short cleared their thresholds).

    Long preferred over short when both fire — same as np.where chain.
    """
    p_long = proba_3class[:, 1]
    p_short = proba_3class[:, 2]
    return np.where(p_long > hc_long, 1.0,
                    np.where(p_short > hc_short, -1.0, 0.0))


def sharpe_after_fees(
    proba_3class: np.ndarray, returns: np.ndarray, interval: str,
    fee: float = 0.001, slippage: float = 0.0003,
    hc_long: float = 0.55, hc_short: float = 0.55,
) -> float:
    """3-class Sharpe with realistic execution costs.

    FIXES (2026-04-27):
    - Position-return alignment: `returns[i]` is the PnL realized on bar i+1
      (i.e. close[i+1] - close[i] / close[i]); the position decided at bar i
      should multiply that NEXT-bar return, not the past-bar return. Earlier
      `pos[i] * returns[i]` was a sign-reversed look-ahead — the strategy
      "earned" returns that already happened before the prediction.
    - Fees only on entry/exit (transitions), not on every hold bar. Old
      logic charged 0.26% on every bar where pos != 0 — for a strategy that
      held short for 50 bars in a row, fees ate 13% per trade instead of 0.26%.
    - Sharpe denominator: use std of GROSS returns (pos * ret_next), not
      net (gross - cost). Cost is a step function on entry/exit bars and
      its variance shouldn't depress the volatility estimate.

    `returns` MUST be one-bar pct returns (close[t+1]-close[t])/close[t] passed
    in already shifted (so returns[i] is the bar-i+1 outcome). Caller must
    align — this fn does NOT shift further.
    """
    pos = _positions_from_proba(proba_3class, hc_long, hc_short)

    # Single-trade explosion guard (Round-5 byzantine review).
    # Sharpe = mean/std with √(bars/year) annualization. With 1 non-zero bar
    # in a 1000-bar fold, std is tiny → Sharpe explodes to ~600. Treat as NaN.
    n_trades = int((pos != 0).sum())
    if n_trades < MIN_TRADES_FOR_SHARPE:
        return float("nan")

    # Transitional fees: cost is proportional to ABS change in position.
    #   0 → 1   = |1−0| = 1 fee  (open long)
    #   1 → 1   = |1−1| = 0 fees (hold)
    #   1 → 0   = |0−1| = 1 fee  (close long)
    #   1 → -1  = |−1−1| = 2 fees (close long AND open short — flip)
    #  -1 → 1   = |1−(−1)| = 2 fees (close short AND open long — flip)
    # The earlier `(pos != pos_prev).astype(float)` charged 1 fee on flips,
    # 50% under-charge on direction reversals. Flagged in re-review (2026-04-27).
    pos_prev = np.concatenate([[0.0], pos[:-1]])
    pos_change = np.abs(pos - pos_prev)
    cost = pos_change * (fee + slippage)

    gross = pos * returns
    net = gross - cost
    if gross.std() < 1e-9:
        return 0.0
    bpy = BARS_PER_YEAR.get(interval, 252)
    # Numerator uses NET (so fees subtract honestly); denominator uses GROSS
    # (so the entry-bar cost spike doesn't fake-volatility the strategy).
    return float(np.sqrt(bpy) * net.mean() / gross.std())


def compute_direction_share(proba_3class: np.ndarray,
                             hc_long: float = 0.55,
                             hc_short: float = 0.55) -> dict:
    """How balanced are signals? 100%-short bias is the failure mode we hunt.

    Mirrors `sharpe_after_fees` position logic via _positions_from_proba so
    `long_share + short_share + hold_share == 1.0` exactly, instead of
    counting long-and-short-both-fire bars in both buckets.
    """
    n = len(proba_3class)
    if n == 0:
        return {"long_share": 0.0, "short_share": 0.0, "hold_share": 1.0}
    pos = _positions_from_proba(proba_3class, hc_long, hc_short)
    return {
        "long_share": float((pos == 1.0).mean()),
        "short_share": float((pos == -1.0).mean()),
        "hold_share": float((pos == 0.0).mean()),
    }


# ─── Single-fold worker ──────────────────────────────────────────────────────

def run_fold(
    fold_idx: int,
    window: RegimeWindow,
    feat: pd.DataFrame,
    feature_cols: List[str],
    interval: str,
    presets: str,
    time_limit_per_fold: int,
    eval_metric: str,
    output_dir: Path,
    num_gpus: int = 1,
    require_gpu: bool = True,
    seed: int = 42,
) -> Optional[FoldResult]:
    """Train AG on bars before window.start; evaluate on bars in [window.start, window.end)."""

    # window.start / window.end are ISO strings from RegimeWindow.dataclass which
    # bucket_windows produces as tz-aware UTC timestamps. feat['open_time'] is
    # likewise UTC after build_features. Force both sides through pd.Timestamp
    # with explicit tz='UTC' so a tz-naive feed (e.g. older parquet) doesn't
    # crash the comparison with "Invalid comparison between dtype=datetime64[ns]
    # and Timestamp".
    feat_ts = pd.DatetimeIndex(feat["open_time"])
    if feat_ts.tz is None:
        feat_ts = feat_ts.tz_localize("UTC")
    w_start = pd.Timestamp(window.start)
    w_end = pd.Timestamp(window.end)
    if w_start.tz is None:
        w_start = w_start.tz_localize("UTC")
    if w_end.tz is None:
        w_end = w_end.tz_localize("UTC")
    # PURGE GAP between train and test: target = close[t+horizon], so the last
    # `horizon` train rows have labels computed from close prices that sit
    # INSIDE the test window. Without purging, the model trains on labels
    # that informed by bars in test → calibration corrupted. Standard López
    # de Prado fix: drop the last `horizon` train rows (purge).
    horizon = TF_CONFIG[interval]["horizon"]
    _bar_dt_map = {"5m": "5min", "15m": "15min", "1h": "1h",
                    "4h": "4h", "1d": "1d"}
    if interval not in _bar_dt_map:
        raise ValueError(
            f"Unsupported interval '{interval}' for purge gap calc. "
            f"Add to bar_dt map (~{interval}) and TF_CONFIG."
        )
    bar_dt = pd.Timedelta(_bar_dt_map[interval])
    purge_cutoff = w_start - horizon * bar_dt
    train_mask = feat_ts < purge_cutoff
    test_mask = (feat_ts >= w_start) & (feat_ts < w_end)
    train_df = feat[train_mask]
    test_df = feat[test_mask]
    if len(train_df) < 500 or len(test_df) < 30:
        print(f"  fold {fold_idx}: skip — train={len(train_df)} test={len(test_df)}")
        return None
    if train_df[TARGET_COL].nunique() < 3:
        print(f"  fold {fold_idx}: skip — train missing a class")
        return None

    print(f"  fold {fold_idx} ({window.name}, {window.regime}): "
          f"train={len(train_df):,} ({train_df['open_time'].min().date()} → "
          f"{train_df['open_time'].max().date()}), "
          f"test={len(test_df):,} ({test_df['open_time'].min().date()} → "
          f"{test_df['open_time'].max().date()})")

    fold_out = output_dir / f"fold_{fold_idx:02d}_{window.name}"
    pred = train_predictor(
        train_df, feature_cols, fold_out,
        presets=presets, time_limit=time_limit_per_fold,
        eval_metric=eval_metric,
        num_gpus=num_gpus, require_gpu=require_gpu,
        horizon=horizon, seed=seed,
    )
    eval_res = evaluate_predictor(pred, test_df, feature_cols)

    # Canonical class-label lookup via predictor.class_labels — robust across
    # AutoGluon versions where columns may be int/np.int/string.
    # Coerce to plain Python int (Round-5 byzantine fix): AG 1.6+ may return
    # np.int64; mixed types break `.index()` lookup silently.
    proba_df = pred.predict_proba(test_df[feature_cols])
    class_labels = [int(x) for x in pred.class_labels]
    n = len(test_df)
    y_test_arr = test_df[TARGET_COL].to_numpy()

    def _col_for(class_id: int) -> np.ndarray:
        if class_id in class_labels:
            return proba_df.iloc[:, class_labels.index(class_id)].to_numpy()
        # Class missing from predictor's training labels.
        # SAFE PATH: model genuinely never saw this class → zero is honest
        # (HC thresholds never trigger).
        # CORRUPTION PATH: y_test contains this class but predictor lost it →
        # log_loss/accuracy will silently lie. Raise so we notice.
        n_in_test = int((y_test_arr == class_id).sum())
        if n_in_test > 0:
            raise RuntimeError(
                f"Class {class_id} present in y_test ({n_in_test} bars) but "
                f"missing from predictor.class_labels={class_labels}. Silent "
                f"zero would corrupt log_loss/accuracy. This signals AG "
                f"dropped a class during training — investigate."
            )
        return np.zeros(n)

    proba_arr = np.stack([
        _col_for(TRIPLE_CLASS_HOLD),
        _col_for(TRIPLE_CLASS_LONG),
        _col_for(TRIPLE_CLASS_SHORT),
    ], axis=1)

    # ret_next[i] = pct return REALIZED on bar i+1, i.e. (close[i+1]-close[i])/close[i].
    # The position pos[i] decided on bar i is held into bar i+1 and earns ret_next[i].
    # Last bar has no future return → 0 (we don't trade the last bar anyway).
    close_te = test_df["close"].to_numpy()
    ret_next = np.zeros_like(close_te)
    if len(close_te) > 1:
        ret_next[:-1] = (close_te[1:] - close_te[:-1]) / close_te[:-1]

    sh = sharpe_after_fees(proba_arr, ret_next, interval=interval,
                           hc_long=0.55, hc_short=0.55)
    dir_balance = compute_direction_share(proba_arr, 0.55, 0.55)

    hc_55 = eval_res["hc"]["thr_0.55"]

    return FoldResult(
        fold=fold_idx,
        window_name=window.name,
        regime=window.regime,
        n_train=int(len(train_df)),
        n_test=int(len(test_df)),
        test_start=window.start,
        test_end=window.end,
        accuracy=eval_res["accuracy"],
        f1_macro=eval_res["f1_macro"],
        log_loss=eval_res["log_loss"],
        sharpe=sh,
        n_long_signals=hc_55["n_long"],
        prec_long_55=hc_55["prec_long"],
        n_short_signals=hc_55["n_short"],
        prec_short_55=hc_55["prec_short"],
        direction_balance=dir_balance,
        leaderboard_top5=eval_res["leaderboard"][:5],
    )


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT", type=_validate_symbol)
    ap.add_argument("--interval", default="4h", choices=list(TF_CONFIG))
    ap.add_argument("--years", type=float, default=8.0)
    ap.add_argument("--with-macro", action="store_true")
    ap.add_argument("--presets", default="medium_quality")
    ap.add_argument("--time-limit", type=int, default=1200,
                    help="AutoGluon training budget per fold (seconds). "
                         "600s is too tight for medium_quality on full train sets — "
                         "AG silently skips models that don't fit in the budget.")
    ap.add_argument("--eval-metric", default="log_loss")
    ap.add_argument("--regime-source", default="data/BTCUSDT_1d.parquet",
                    help="OHLCV parquet used to classify regimes (1d strongly recommended)")
    ap.add_argument("--bull-thr", type=float, default=DEFAULT_BULL_THRESHOLD)
    ap.add_argument("--bear-thr", type=float, default=DEFAULT_BEAR_THRESHOLD)
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--no-require-gpu", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    print(f"━━━ WFCV regime-aware: {args.symbol} {args.interval} "
          f"(macro={args.with_macro}) ━━━")

    btc_close = None
    if args.symbol != "BTCUSDT":
        btc_df = fetch_or_cache("BTCUSDT", args.interval, years=args.years)
        btc_close = btc_df["close"].to_numpy()

    feat, _, meta = build_dataset(args.symbol, args.interval, args.years,
                                   args.with_macro, btc_close=btc_close)
    feature_cols = meta["feature_cols"]

    regime_path = Path(args.regime_source)
    if not regime_path.is_absolute():
        regime_path = ROOT / regime_path
    feat = attach_regime_to_frame(feat, regime_path)

    # Build windows from regime tags on this TF's frame.
    regimes_arr = feat["regime"].to_numpy()
    ts = pd.DatetimeIndex(feat["open_time"])
    # Bucket per TF directly (so chop_2024_h2 has bars at this TF's resolution).
    min_window = max(50, int(7 * BARS_PER_YEAR[args.interval] / 365))  # ~1 week
    windows = bucket_windows(ts, regimes_arr, min_window_bars=min_window)
    print(f"  built {len(windows)} regime windows on {args.interval} grid "
          f"(min {min_window} bars)")
    test_windows = pick_five_test_windows(windows)
    print(f"  selected {len(test_windows)} test windows:")
    for w in test_windows:
        print(f"    {w.name:25s} {w.regime:5s} "
              f"{w.start[:10]} → {w.end[:10]}  ({w.n_bars} bars)")

    macro_tag = "_macro" if args.with_macro else ""
    output_root = WFCV_DIR / f"{args.symbol}_{args.interval}{macro_tag}"
    output_root.mkdir(parents=True, exist_ok=True)

    results: List[FoldResult] = []
    for i, w in enumerate(test_windows):
        fr = run_fold(
            i, w, feat, feature_cols, args.interval,
            presets=args.presets, time_limit_per_fold=args.time_limit,
            eval_metric=args.eval_metric, output_dir=output_root,
            num_gpus=args.num_gpus, require_gpu=not args.no_require_gpu,
            seed=args.seed,
        )
        if fr is not None:
            results.append(fr)

    if not results:
        print("\nNo folds completed — aborting.")
        return

    # Aggregate.
    # Use nan-aware aggregations: folds with <MIN_TRADES_FOR_SHARPE return NaN
    # to suppress single-trade explosion artefacts (Round-5 fix).
    sharpes = np.array([r.sharpe for r in results])
    accs = np.array([r.accuracy for r in results])
    f1s = np.array([r.f1_macro for r in results])
    long_shares = np.array([r.direction_balance["long_share"] for r in results])
    short_shares = np.array([r.direction_balance["short_share"] for r in results])

    n_valid_sharpe = int(np.isfinite(sharpes).sum())
    n_nan_sharpe = len(sharpes) - n_valid_sharpe
    print(f"\n=== WFCV summary ({len(results)} folds, {n_valid_sharpe} sharpe-valid, "
          f"{n_nan_sharpe} too few trades) ===")
    if n_valid_sharpe > 0:
        print(f"  Sharpe:    mean={np.nanmean(sharpes):+.2f} +/- {np.nanstd(sharpes):.2f}  "
              f"(min {np.nanmin(sharpes):+.2f}, max {np.nanmax(sharpes):+.2f})")
    else:
        print(f"  Sharpe:    ALL FOLDS UNDERTRADED — verdict cannot be determined")
    print(f"  Accuracy:  mean={accs.mean():.3f} +/- {accs.std():.3f}")
    print(f"  F1 macro:  mean={f1s.mean():.3f} +/- {f1s.std():.3f}")
    print(f"  Long share avg:  {long_shares.mean():.1%} (target: ~33%)")
    print(f"  Short share avg: {short_shares.mean():.1%} (target: ~33%)")

    # Per-regime breakdown.
    print("  Per regime:")
    for regime in ("bull", "chop", "bear"):
        sub = [r for r in results if r.regime == regime]
        if not sub:
            continue
        s = np.array([r.sharpe for r in sub])
        l = np.array([r.direction_balance["long_share"] for r in sub])
        sh = np.array([r.direction_balance["short_share"] for r in sub])
        print(f"    {regime:5s}: n={len(sub)}  sharpe={s.mean():+.2f}  "
              f"long={l.mean():.0%}  short={sh.mean():.0%}")

    # Verdict logic — uses nan-aware aggregation. If too many folds are NaN,
    # we can't decide.
    if n_valid_sharpe < 3:
        verdict = "INCONCLUSIVE"
    else:
        sh_mean = float(np.nanmean(sharpes))
        sh_std = float(np.nanstd(sharpes))
        sh_min = float(np.nanmin(sharpes))
        if sh_mean >= 1.0 and sh_min >= -0.5:
            verdict = "GREEN"
        elif sh_mean >= 0.0 and sh_std < 3.0:
            verdict = "CONDITIONAL"
        elif sh_mean <= -1.0:
            verdict = "DROP"
        else:
            verdict = "KNOWN_LIMITATION"
    print(f"\n  VERDICT: {verdict}")

    out_path = LOGS_DIR / f"wfcv_regime_{args.symbol}_{args.interval}{macro_tag}.json"
    atomic_write_json(out_path, {
        "version": "wfcv_regime_v1",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": time.time() - t0,
        "symbol": args.symbol,
        "interval": args.interval,
        "with_macro": args.with_macro,
        "presets": args.presets,
        "time_limit_per_fold": args.time_limit,
        "eval_metric": args.eval_metric,
        "n_features": meta["n_features"],
        "windows_picked": [asdict(w) for w in test_windows],
        "folds": [asdict(r) for r in results],
        "summary": {
            "n_folds": len(results),
            "n_sharpe_valid": n_valid_sharpe,
            "n_sharpe_nan_undertraded": n_nan_sharpe,
            "sharpe_mean": float(np.nanmean(sharpes)) if n_valid_sharpe > 0 else None,
            "sharpe_std": float(np.nanstd(sharpes)) if n_valid_sharpe > 0 else None,
            "sharpe_min": float(np.nanmin(sharpes)) if n_valid_sharpe > 0 else None,
            "sharpe_max": float(np.nanmax(sharpes)) if n_valid_sharpe > 0 else None,
            "accuracy_mean": float(accs.mean()),
            "f1_macro_mean": float(f1s.mean()),
            "long_share_mean": float(long_shares.mean()),
            "short_share_mean": float(short_shares.mean()),
            "verdict": verdict,
        },
    })
    print(f"\n  Report -> {out_path.relative_to(ROOT)}")
    print(f"  ⏱  {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
