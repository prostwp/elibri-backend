"""
backtest_v2.py — Patch 2C risk-tier OOS backtest.

Runs the trained ensemble through the test window (last 30%) with:
  - Real transaction costs (fee per side + slippage on entry).
  - 1d-filtered MTF direction alignment.
  - Label classification (trend_aligned / mean_reversion / random).
  - Four-layer gate pipeline: HC -> vol -> label -> rate. (Patch 2E dropped
    the redundant tier.min_confidence gate; HC threshold is the single
    confidence filter.)
  - Three tiers (conservative/balanced/aggressive) in parallel with the
    SAME model predictions — so we're comparing risk rules, not randomness.

Outputs per tier/TF:
  logs/backtest_v2/{sym}_{tier}_{tf}.json   — trades + metrics + reject counts
  logs/backtest_v2/{sym}_{tier}_{tf}.png    — equity curve
  logs/backtest_v2/{sym}_{tier}_summary.json
  logs/backtest_v2/comparison.{json,png}    — 3 tiers side-by-side

Sanity guard:
  Before the main loop we print proba_min/max on the holdout per TF. If
  proba_max < the configured HC threshold (i.e. the model NEVER fires a
  long signal at that threshold — known BTC 5m issue where proba_max=0.71
  but best_thresholds says 0.775), we auto-lower the HC threshold to the
  95th percentile of |proba-0.5|+0.5 so the backtest doesn't silently
  return "0 trades". The adjustment is logged.

Intentionally NOT run locally (per
  memory/feedback_ml_training_server_only.md) — validate AST then scp.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engine import FEATURE_NAMES, build_features, make_target, _atr
from data_fetcher import fetch_or_cache
from train import HORIZON_MAP, train_ensemble, ensemble_predict
from risk_tiers_config import TIERS, tier_names


ROOT = Path(__file__).parent
LOGS_DIR = ROOT / "logs"
DEFAULT_OUT = LOGS_DIR / "backtest_v2"


# Fallback HC thresholds if best_thresholds.json is missing or lacks an entry.
FALLBACK_HC = {
    "5m":  0.75,
    "15m": 0.73,
    "1h":  0.65,
    "4h":  0.60,
    "1d":  0.55,
}

# Max bars to hold before forced close. Matches horizon-ish bucketing so a
# 5m trade doesn't sit open for a week hoping for TP.
HORIZON_EXIT = {
    "5m":  48,    # ~4h
    "15m": 64,    # ~16h
    "1h":  48,    # ~2d
    "4h":  36,    # ~6d
    "1d":  20,    # ~20d
}


# ------------------------------------------------------------ best thresholds --

def load_best_thresholds(path: Path | None = None) -> dict[str, float]:
    """Return {"{SYM}_{TF}": threshold_float}. Missing file => empty dict."""
    p = path or (LOGS_DIR / "best_thresholds.json")
    if not p.exists():
        print(f"[warn] {p} missing, falling back to FALLBACK_HC")
        return {}
    try:
        raw = json.loads(p.read_text())
    except Exception as e:
        print(f"[warn] failed to parse {p}: {e}; falling back")
        return {}
    out: dict[str, float] = {}
    for row in raw.get("results", []):
        sym = row.get("symbol")
        iv = row.get("interval")
        best = row.get("best") or {}
        thr = best.get("threshold")
        if sym and iv and thr is not None:
            out[f"{sym}_{iv}"] = float(thr)
    return out


# ------------------------------------------------------------ train + predict --

def train_and_predict(symbol: str, interval: str, years: float,
                      btc_close: np.ndarray | None = None):
    """Train on first 70%, predict on remaining 30%.

    Returns a dict with:
      proba    : np.ndarray[float]  (test set)
      dates    : np.ndarray[datetime64]
      closes   : np.ndarray[float]
      highs    : np.ndarray[float]
      lows     : np.ndarray[float]
      atr      : np.ndarray[float]  (absolute ATR14 aligned to test bars)
      features : pd.DataFrame       (rsi_14, bb_position, adx_14, atr_norm_14)
    """
    df = fetch_or_cache(symbol, interval, years=years)
    if len(df) < 500:
        raise RuntimeError(f"{symbol} {interval}: only {len(df)} candles, need >= 500")

    horizon = HORIZON_MAP[interval]
    feat = build_features(df, btc_close=btc_close)
    target = make_target(df["close"].to_numpy(), horizon=horizon)
    mask = target >= 0
    feat = feat.loc[mask].reset_index(drop=True)
    target = target[mask]
    df_m = df[mask].reset_index(drop=True)

    split = int(len(feat) * 0.7)
    X_train = feat[FEATURE_NAMES].iloc[:split].to_numpy()
    y_train = target[:split]
    X_test = feat[FEATURE_NAMES].iloc[split:].to_numpy()

    xgb, lgbm, rf, meta = train_ensemble(X_train, y_train, quick=False)
    proba, _ = ensemble_predict(xgb, lgbm, rf, meta, X_test)

    # ATR absolute on full masked frame, then slice the test region.
    atr_all = _atr(
        df_m["high"].to_numpy(),
        df_m["low"].to_numpy(),
        df_m["close"].to_numpy(),
        14,
    )

    slc = slice(split, len(feat))
    # Feature subset we need downstream for label classification + vol gate.
    feats_test = feat[["rsi_14", "bb_position", "adx_14", "atr_norm_14"]].iloc[slc].reset_index(drop=True)

    return {
        "proba": np.asarray(proba, dtype=float),
        "dates": pd.to_datetime(feat["open_time"].iloc[slc].values),
        "closes": feat["close"].iloc[slc].to_numpy().astype(float),
        "highs": df_m["high"].iloc[slc].to_numpy().astype(float),
        "lows": df_m["low"].iloc[slc].to_numpy().astype(float),
        "atr": atr_all[slc].astype(float),
        "features": feats_test,
    }


# ---------------------------------------------------------------- MTF filter --

def align_1d_direction(dates_tf: pd.DatetimeIndex,
                       dates_1d: pd.DatetimeIndex,
                       proba_1d: np.ndarray,
                       threshold: float = 0.5,
                       neutral_band: float = 0.05) -> np.ndarray:
    """Map daily-TF prob back onto entry-TF timeline via np.interp.

    Returns int array with values {+1, -1, 0}:
      +1 if interp(proba_1d) >= threshold + neutral_band
      -1 if interp(proba_1d) <= threshold - neutral_band
       0 otherwise (flat / uncertain)
    """
    # DatetimeIndex -> ns ints. np.asarray on a DatetimeIndex can return an
    # object dtype array, so go through .values to force datetime64[ns] first.
    x_tf = pd.DatetimeIndex(dates_tf).values.astype("datetime64[ns]").astype(np.int64)
    x_1d = pd.DatetimeIndex(dates_1d).values.astype("datetime64[ns]").astype(np.int64)
    # np.interp requires the xp array to be monotonic increasing.
    if len(x_1d) > 1 and not np.all(np.diff(x_1d) >= 0):
        order = np.argsort(x_1d)
        x_1d = x_1d[order]
        proba_1d = np.asarray(proba_1d)[order]
    interp_proba = np.interp(x_tf, x_1d, proba_1d, left=0.5, right=0.5)
    out = np.zeros(len(x_tf), dtype=np.int8)
    out[interp_proba >= threshold + neutral_band] = 1
    out[interp_proba <= threshold - neutral_band] = -1
    return out


# ------------------------------------------------------------ label classifier --

def label_signal(rsi: float, bb_pos: float, adx_scaled: float,
                 signal_dir: int, dir_1d: int, interval: str = "") -> str:
    """Return 'trend_aligned', 'mean_reversion', or 'random'.

    adx_14 in the feature frame is already scaled /100.0, so ADX>20 means
    adx_scaled > 0.20. Pass the scaled value in unchanged.

    Patch 2F: `interval` added so 5m signals can NEVER be labelled
    'mean_reversion' — disabled by Patch 2F: 5m mean-rev is pure noise
    (backtest-confirmed: Aggressive -69% return / 70% DD).
    """
    if signal_dir == dir_1d and dir_1d != 0 and adx_scaled > 0.20:
        return "trend_aligned"
    if dir_1d == 0 and ((rsi < 30.0 or rsi > 70.0) or (bb_pos < 0.10 or bb_pos > 0.90)):
        # disabled by Patch 2F: 5m mean-rev is pure noise (backtest-confirmed)
        if interval == "5m":
            return "random"
        return "mean_reversion"
    return "random"


# ---------------------------------------------------------------- gate logic --

def apply_gates(proba: float, features_row: pd.Series, tier_cfg: dict,
                hc_threshold: float, interval: str, label: str,
                trades_today: int) -> tuple[bool, str]:
    """Four-layer gate pipeline. Returns (passed, reject_reason).

    Patch 2E: removed the tier.min_confidence gate — HC threshold (per-TF
    from best_thresholds.json) is now the single confidence filter. The
    tier-level confidence floor was double-gating the same signal and
    causing 0 trades on Conservative/Balanced.
    """
    # 1. HC threshold — must be outside [1-hc, hc].
    if (proba <= hc_threshold) and (proba >= 1.0 - hc_threshold):
        return False, "hc_threshold"

    # 2. Volatility floor.
    min_vol = tier_cfg["min_vol_pct"].get(interval, 0.0)
    atr_norm = float(features_row["atr_norm_14"])
    if atr_norm < min_vol:
        return False, "vol_floor"

    # 3. Label allow-list.
    if label not in tier_cfg["allowed_labels"]:
        return False, "label_not_allowed"

    # 4. Daily rate limit.
    if trades_today >= tier_cfg["max_trades_per_day"]:
        return False, "rate_limit"

    return True, ""


# ---------------------------------------------------------------- trade sim --

def sim_trade(i: int, direction: int, entry_ref: float, atr: float,
              tier_cfg: dict, equity: float, fee: float, slippage: float,
              horizon_bars: int, highs: np.ndarray, lows: np.ndarray,
              closes: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    """Simulate one trade with real costs. Returns dict with all trade fields
    plus ``equity_after``, ready to wrap into TradeRec.
    """
    sl_mult = tier_cfg["sl_atr_mult"]
    tp_mult = tier_cfg["tp_atr_mult"]
    risk_pct = tier_cfg["risk_per_trade_pct"]

    # Slippage pushes the entry AGAINST us. ATR fallback guards against zero.
    if atr <= 0 or not np.isfinite(atr):
        atr = entry_ref * 0.005
    entry_price = entry_ref * (1.0 + slippage) if direction > 0 else entry_ref * (1.0 - slippage)

    if direction > 0:
        sl = entry_price - sl_mult * atr
        tp = entry_price + tp_mult * atr
    else:
        sl = entry_price + sl_mult * atr
        tp = entry_price - tp_mult * atr

    # Turtle sizing: (equity * risk_pct) / stop_distance. volume_usd is
    # notional exposure; fees are on each side of the notional.
    stop_dist = abs(entry_price - sl)
    if stop_dist <= 0 or not np.isfinite(stop_dist):
        stop_dist = entry_price * 0.005
    volume_usd = (equity * risk_pct) / stop_dist * entry_price

    # Walk forward for exit.
    n = len(closes)
    end = min(n, i + 1 + horizon_bars)
    # Default exit (timeout) = last bar we have permission to use. Guard
    # against the edge where i is at the very end of the series.
    if end > i + 1:
        exit_idx = end - 1
    elif i + 1 < n:
        exit_idx = i + 1
    else:
        exit_idx = i
    exit_price = closes[exit_idx]
    exit_reason = "timeout"

    for j in range(i + 1, end):
        if direction > 0:
            if lows[j] <= sl:
                exit_idx, exit_price, exit_reason = j, sl, "sl"
                break
            if highs[j] >= tp:
                exit_idx, exit_price, exit_reason = j, tp, "tp"
                break
        else:
            if highs[j] >= sl:
                exit_idx, exit_price, exit_reason = j, sl, "sl"
                break
            if lows[j] <= tp:
                exit_idx, exit_price, exit_reason = j, tp, "tp"
                break

    # PnL as fraction of entry.
    if direction > 0:
        pnl_pct = (exit_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - exit_price) / entry_price
    gross_dollars = volume_usd * pnl_pct
    fees_paid = volume_usd * fee * 2.0  # entry + exit
    pnl_dollars = gross_dollars - fees_paid
    equity_after = equity + pnl_dollars

    return {
        "entry_idx": i,
        "exit_idx": exit_idx,
        "entry_date": str(dates[i]),
        "exit_date": str(dates[exit_idx]),
        "direction": direction,
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "sl": float(sl),
        "tp": float(tp),
        "bars_held": exit_idx - i,
        "pnl_pct": float(pnl_dollars / max(equity, 1e-9)),  # realised return on equity
        "raw_pnl_pct": float(pnl_pct),
        "pnl_dollars": float(pnl_dollars),
        "fees_paid": float(fees_paid),
        "equity_after": float(equity_after),
        "exit_reason": exit_reason,
    }


# ---------------------------------------------------------------- aggregation --

def _sharpe(returns: np.ndarray, trades_per_year: float) -> float:
    if returns.size < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(trades_per_year))


def _annual_factor(interval: str, avg_bars: float) -> float:
    if avg_bars <= 0:
        avg_bars = 1.0
    return {
        "5m":  12 * 24 * 365 / avg_bars,
        "15m":  4 * 24 * 365 / avg_bars,
        "1h":       24 * 365 / avg_bars,
        "4h":        6 * 365 / avg_bars,
        "1d":            252,
    }.get(interval, 252)


def aggregate(trades: list[dict], rejected: dict[str, int],
              initial_equity: float, interval: str) -> dict:
    n = len(trades)
    if n == 0:
        return {
            "n_trades": 0,
            "n_rejected_by_reason": dict(rejected),
            "total_rejected": int(sum(rejected.values())),
            "initial_equity": initial_equity,
            "final_equity": initial_equity,
            "total_return_pct": 0.0,
            "win_rate": 0.0,
            "avg_trade_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_after_fees": 0.0,
            "fees_paid_total": 0.0,
            "per_label": {},
            "per_exit_reason": {},
        }

    wins = [t for t in trades if t["pnl_dollars"] > 0]
    losses = [t for t in trades if t["pnl_dollars"] <= 0]
    total_wins = sum(t["pnl_dollars"] for t in wins)
    total_losses = abs(sum(t["pnl_dollars"] for t in losses))
    pf = (total_wins / total_losses) if total_losses > 0 else 0.0

    eq_vals = [initial_equity] + [t["equity_after"] for t in trades]
    peak = eq_vals[0]
    max_dd = 0.0
    for v in eq_vals:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    returns = np.array([t["pnl_pct"] for t in trades], dtype=float)
    avg_bars = float(np.mean([t["bars_held"] for t in trades]))
    sharpe = _sharpe(returns, _annual_factor(interval, avg_bars))

    # Breakdown by label.
    per_label: dict[str, dict] = {}
    for lbl in ("trend_aligned", "mean_reversion", "random"):
        sub = [t for t in trades if t["label"] == lbl]
        if not sub:
            per_label[lbl] = {"n": 0, "win_rate": 0.0, "total_pnl_dollars": 0.0}
            continue
        sub_w = [t for t in sub if t["pnl_dollars"] > 0]
        per_label[lbl] = {
            "n": len(sub),
            "win_rate": len(sub_w) / len(sub),
            "total_pnl_dollars": float(sum(t["pnl_dollars"] for t in sub)),
            "avg_pnl_pct": float(np.mean([t["pnl_pct"] for t in sub])),
        }

    per_exit: dict[str, int] = defaultdict(int)
    for t in trades:
        per_exit[t["exit_reason"]] += 1

    final_equity = float(eq_vals[-1])
    return {
        "n_trades": n,
        "n_rejected_by_reason": dict(rejected),
        "total_rejected": int(sum(rejected.values())),
        "initial_equity": float(initial_equity),
        "final_equity": final_equity,
        "total_return_pct": (final_equity - initial_equity) / initial_equity * 100.0,
        "win_rate": len(wins) / n,
        "avg_trade_pct": float(np.mean([t["pnl_pct"] for t in trades])) * 100.0,
        "profit_factor": pf,
        "max_drawdown_pct": max_dd * 100.0,
        "sharpe_after_fees": sharpe,
        "avg_bars_held": avg_bars,
        "fees_paid_total": float(sum(t["fees_paid"] for t in trades)),
        "per_label": per_label,
        "per_exit_reason": dict(per_exit),
    }


# ---------------------------------------------------------------- plotting --

def plot_equity_curve(trades: list[dict], initial_equity: float,
                      save_path: Path, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not trades:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "0 trades (all rejected)", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(save_path, dpi=100)
        plt.close(fig)
        return

    dates = [pd.Timestamp(trades[0]["entry_date"])] + [pd.Timestamp(t["exit_date"]) for t in trades]
    eq = [initial_equity] + [t["equity_after"] for t in trades]

    peak = eq[0]
    dd = []
    for v in eq:
        peak = max(peak, v)
        dd.append((v - peak) / peak * 100.0 if peak > 0 else 0.0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), gridspec_kw={"height_ratios": [3, 1]},
                                   sharex=True)
    ax1.plot(dates, eq, lw=1.2, color="#6366f1", label="Equity")
    ax1.axhline(initial_equity, ls=":", color="#666", alpha=0.7)
    # Trade markers — green long win / red short win / gray losses.
    for t in trades:
        exit_ts = pd.Timestamp(t["exit_date"])
        if t["pnl_dollars"] > 0:
            color = "#16a34a" if t["direction"] > 0 else "#059669"
        else:
            color = "#94a3b8"
        ax1.scatter([exit_ts], [t["equity_after"]], s=10, c=color, alpha=0.6, zorder=3)

    ax1.set_ylabel("Equity $")
    ax1.set_title(title)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left")

    ax2.fill_between(dates, dd, 0, color="#dc2626", alpha=0.35)
    ax2.set_ylabel("Drawdown %")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def plot_comparison(tier_curves: dict[str, list[tuple]], initial_equity: float,
                    save_path: Path, title: str) -> None:
    """tier_curves: {tier_name: [(timestamp, equity), ...]}."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    colors = {"conservative": "#2563eb", "balanced": "#16a34a", "aggressive": "#dc2626"}
    fig, ax = plt.subplots(figsize=(11, 5))
    for tier, curve in tier_curves.items():
        if not curve:
            continue
        xs, ys = zip(*curve)
        ax.plot(xs, ys, lw=1.4, label=tier, color=colors.get(tier, "#666"))
    ax.axhline(initial_equity, ls=":", color="#666", alpha=0.7)
    ax.set_ylabel("Equity $")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------- main loop --

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _sanity_adjust_hc(interval: str, proba: np.ndarray, desired_hc: float) -> tuple[float, str | None]:
    """If proba_max < desired_hc, lower threshold to the 95th percentile of
    the long-side distribution. Returns (effective_hc, note_or_None).
    """
    if len(proba) == 0:
        return desired_hc, None
    pmax = float(np.max(proba))
    pmin = float(np.min(proba))
    # Longs and shorts both need coverage. If the model never exceeds
    # desired_hc on either side, pick the 95th-pctile of the side that
    # has more mass.
    long_reachable = pmax >= desired_hc
    short_reachable = pmin <= (1.0 - desired_hc)
    if long_reachable or short_reachable:
        return desired_hc, None

    # Use 95th percentile of |proba-0.5|+0.5 — i.e. the 95% confidence
    # level actually achievable. Clamp to >= 0.52 so we still have a
    # meaningful band.
    conf = np.abs(proba - 0.5) + 0.5
    new_hc = float(np.clip(np.quantile(conf, 0.95), 0.52, desired_hc))
    note = (f"HC threshold adjusted from {desired_hc:.3f} to {new_hc:.3f} "
            f"due to holdout proba cap (min={pmin:.3f}, max={pmax:.3f})")
    return new_hc, note


def run_tier(tier: str, symbol: str, intervals: list[str], years: float,
             fee: float, slippage: float, initial_equity: float,
             thresholds: dict[str, float], out_dir: Path,
             predictions_cache: dict,
             daily: dict) -> dict:
    """Run a single tier across all entry-TFs. Shares predictions_cache with
    other tiers so we only train once per (symbol, tf)."""
    cfg = TIERS[tier]
    print(f"\n═══ TIER: {tier.upper()} ═══")
    per_tf_metrics: dict[str, dict] = {}
    equity_curves_for_comparison: list[tuple] = []

    for tf in intervals:
        pred = predictions_cache[tf]
        proba = pred["proba"]
        dates = pred["dates"]
        closes = pred["closes"]
        highs = pred["highs"]
        lows = pred["lows"]
        atr = pred["atr"]
        feats = pred["features"]

        # Higher-TF direction (daily).
        dir_1d = align_1d_direction(
            dates,
            daily["dates"],
            daily["proba"],
        )

        # HC threshold — best_thresholds, else fallback, then sanity-adjust.
        desired_hc = thresholds.get(f"{symbol}_{tf}", FALLBACK_HC[tf])
        effective_hc, note = _sanity_adjust_hc(tf, proba, desired_hc)
        if note:
            print(f"  [sanity] {tf}: {note}")

        trades: list[dict] = []
        rejected: dict[str, int] = defaultdict(int)
        trades_today: dict = defaultdict(int)
        # Equity starts fresh per TF so per-TF JSONs are directly comparable.
        # Cross-TF compounding would couple the ordering of intervals.
        equity = initial_equity

        horizon_bars = HORIZON_EXIT.get(tf, 48)
        n = len(proba)
        i = 0
        while i < n:
            p = float(proba[i])
            if p > effective_hc:
                direction = 1
            elif p < (1.0 - effective_hc):
                direction = -1
            else:
                rejected["hc_threshold"] += 1
                i += 1
                continue

            row = feats.iloc[i]
            label = label_signal(
                rsi=float(row["rsi_14"]),
                bb_pos=float(row["bb_position"]),
                adx_scaled=float(row["adx_14"]),
                signal_dir=direction,
                dir_1d=int(dir_1d[i]),
                interval=tf,
            )
            day_key = pd.Timestamp(dates[i]).date()
            passed, reason = apply_gates(
                proba=p,
                features_row=row,
                tier_cfg=cfg,
                hc_threshold=effective_hc,
                interval=tf,
                label=label,
                trades_today=trades_today[day_key],
            )
            if not passed:
                rejected[reason] += 1
                i += 1
                continue

            trade = sim_trade(
                i=i,
                direction=direction,
                entry_ref=float(closes[i]),
                atr=float(atr[i]),
                tier_cfg=cfg,
                equity=equity,
                fee=fee,
                slippage=slippage,
                horizon_bars=horizon_bars,
                highs=highs,
                lows=lows,
                closes=closes,
                dates=dates,
            )
            trade["label"] = label
            trade["probability"] = p
            trade["atr_norm"] = float(row["atr_norm_14"])
            trades.append(trade)
            equity = trade["equity_after"]
            trades_today[day_key] += 1
            i = trade["exit_idx"] + 1

        metrics = aggregate(trades, rejected, initial_equity, tf)
        metrics["hc_threshold_effective"] = effective_hc
        metrics["hc_threshold_desired"] = desired_hc
        metrics["fee"] = fee
        metrics["slippage"] = slippage
        per_tf_metrics[tf] = metrics

        out_file = out_dir / f"{symbol}_{tier}_{tf}.json"
        with open(out_file, "w") as f:
            json.dump({
                "symbol": symbol, "tier": tier, "interval": tf,
                "trades": trades,
                "metrics": metrics,
                "rejected": dict(rejected),
            }, f, indent=2, default=str)

        plot_equity_curve(
            trades, initial_equity, out_dir / f"{symbol}_{tier}_{tf}.png",
            f"{symbol} {tf} / {tier}  trades={len(trades)} "
            f"ret={metrics['total_return_pct']:+.1f}% "
            f"WR={metrics['win_rate']*100:.0f}% "
            f"DD={metrics['max_drawdown_pct']:.1f}%",
        )

        # Build an aggregated curve (time-stamped) for comparison plot.
        curve = [(pd.Timestamp(dates[0]), initial_equity)]
        for t in trades:
            curve.append((pd.Timestamp(t["exit_date"]), t["equity_after"]))
        equity_curves_for_comparison.extend(curve)

        print(f"  {tf:>3}  trades={len(trades):>4}  rej={metrics['total_rejected']:>5}  "
              f"WR={metrics['win_rate']*100:>4.0f}%  "
              f"ret={metrics['total_return_pct']:+7.1f}%  "
              f"Sharpe={metrics['sharpe_after_fees']:+.2f}  "
              f"DD={metrics['max_drawdown_pct']:>4.1f}%  "
              f"fees=${metrics['fees_paid_total']:.0f}")

    # Tier summary across all TFs (simple additive).
    tier_sum = {
        "symbol": symbol,
        "tier": tier,
        "per_tf": per_tf_metrics,
        "aggregated": {
            "n_trades_total": int(sum(m["n_trades"] for m in per_tf_metrics.values())),
            "n_rejected_total": int(sum(m["total_rejected"] for m in per_tf_metrics.values())),
            "avg_return_pct": float(np.mean([m["total_return_pct"] for m in per_tf_metrics.values()])),
            "avg_win_rate": float(np.mean([m["win_rate"] for m in per_tf_metrics.values()])),
            "avg_sharpe": float(np.mean([m["sharpe_after_fees"] for m in per_tf_metrics.values()])),
            "max_drawdown_pct": float(max(m["max_drawdown_pct"] for m in per_tf_metrics.values())),
            "fees_paid_total": float(sum(m["fees_paid_total"] for m in per_tf_metrics.values())),
        },
    }
    with open(out_dir / f"{symbol}_{tier}_summary.json", "w") as f:
        json.dump(tier_sum, f, indent=2, default=str)

    # Sort curve by time for comparison plotting.
    equity_curves_for_comparison.sort(key=lambda x: x[0])
    return {"metrics": tier_sum, "curve": equity_curves_for_comparison}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="all",
                    help="Which tier(s) to run. One of: all | conservative | balanced | aggressive")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--intervals", nargs="+", default=["5m", "15m", "1h", "4h"])
    ap.add_argument("--fee", type=float, default=0.001,
                    help="Fee per side (taker). Binance default=0.001 i.e. 10bps.")
    ap.add_argument("--slippage", type=float, default=0.0003,
                    help="Entry slippage as fraction of price, e.g. 3bps.")
    ap.add_argument("--years", type=float, default=8.0)
    ap.add_argument("--initial-equity", type=float, default=10000.0)
    ap.add_argument("--use-best-thresholds", action="store_true",
                    help="Read logs/best_thresholds.json (else falls back to FALLBACK_HC).")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    # Tier selection.
    if args.tier == "all":
        tiers_to_run = tier_names()
    elif args.tier in TIERS:
        tiers_to_run = [args.tier]
    else:
        print(f"[err] unknown tier: {args.tier}", file=sys.stderr)
        sys.exit(2)

    # HC thresholds.
    thresholds = load_best_thresholds() if args.use_best_thresholds else {}
    if args.use_best_thresholds:
        print(f"[info] loaded {len(thresholds)} thresholds from best_thresholds.json")

    # ---- Train + predict ONCE per TF (shared across tiers) ------------------
    print(f"\n── Training {args.symbol} models ({len(args.intervals)} entry TFs + 1d filter) ──")
    predictions: dict[str, dict] = {}
    t_train = time.time()
    for tf in args.intervals:
        print(f"  training {args.symbol} {tf} …")
        predictions[tf] = train_and_predict(args.symbol, tf, args.years)
    # 1d always trained for MTF filter.
    print(f"  training {args.symbol} 1d (MTF filter) …")
    daily_pred = train_and_predict(args.symbol, "1d", args.years)
    daily = {"proba": daily_pred["proba"], "dates": daily_pred["dates"]}
    print(f"  trained in {time.time() - t_train:.1f}s")

    # ---- Sanity check — log proba caps per TF -------------------------------
    print("\n── Proba sanity check (holdout) ──")
    print(f"  {'TF':<5} {'min':>7} {'max':>7} {'desired_hc':>12} {'reachable?':>11}")
    for tf in args.intervals:
        proba = predictions[tf]["proba"]
        pmin, pmax = float(proba.min()), float(proba.max())
        desired = thresholds.get(f"{args.symbol}_{tf}", FALLBACK_HC[tf])
        reachable = (pmax >= desired) or (pmin <= 1.0 - desired)
        mark = "ok" if reachable else "CAPPED"
        print(f"  {tf:<5} {pmin:>7.3f} {pmax:>7.3f} {desired:>12.3f} {mark:>11}")

    # ---- Run each tier ------------------------------------------------------
    tier_results: dict = {}
    for tier in tiers_to_run:
        tr = run_tier(
            tier=tier,
            symbol=args.symbol,
            intervals=args.intervals,
            years=args.years,
            fee=args.fee,
            slippage=args.slippage,
            initial_equity=args.initial_equity,
            thresholds=thresholds,
            out_dir=out_dir,
            predictions_cache=predictions,
            daily=daily,
        )
        tier_results[tier] = tr

    # ---- Comparison output --------------------------------------------------
    comparison_metrics = {
        tier: tr["metrics"]["aggregated"] for tier, tr in tier_results.items()
    }
    comparison_metrics["_meta"] = {
        "symbol": args.symbol,
        "intervals": args.intervals,
        "fee": args.fee,
        "slippage": args.slippage,
        "years": args.years,
        "initial_equity": args.initial_equity,
        "use_best_thresholds": args.use_best_thresholds,
        "generated_at": str(pd.Timestamp.now()),
    }
    with open(out_dir / "comparison.json", "w") as f:
        json.dump(comparison_metrics, f, indent=2, default=str)

    plot_comparison(
        {tier: tr["curve"] for tier, tr in tier_results.items()},
        args.initial_equity,
        out_dir / "comparison.png",
        f"{args.symbol} — risk tier comparison (fee={args.fee:.4f}, slip={args.slippage:.4f})",
    )

    # ---- Console summary ----------------------------------------------------
    print("\n═══ Tier comparison ═══")
    for tier, tr in tier_results.items():
        a = tr["metrics"]["aggregated"]
        print(f"  {tier:<13}  trades={a['n_trades_total']:>4}  "
              f"avg_ret={a['avg_return_pct']:+7.1f}%  "
              f"avg_WR={a['avg_win_rate']*100:>4.0f}%  "
              f"avg_Sharpe={a['avg_sharpe']:+.2f}  "
              f"maxDD={a['max_drawdown_pct']:>4.1f}%")

    print(f"\n[done] artifacts → {out_dir}")


if __name__ == "__main__":
    main()
