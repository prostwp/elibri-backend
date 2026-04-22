"""
iter1b_backtest_mtf.py — Patch 4 Iter 1b MTF-aware execution backtest.

Same 3-class model as iter_backtest_3class.py, but applies two simple rule-based
filters per PATCH_4_GLOBAL_RETRAIN_PLAN.md:

  1. Global trend filter (1d):
       - EMA-200 slope over 20-bar window:
           slope = (ema200[-1] - ema200[-20]) / ema200[-20]
       - close vs ema200[-1]
       - bull:  slope > 0.02  AND  close > ema200
       - bear:  slope < -0.02 AND  close < ema200
       - chop:  otherwise
     Application:
       bull  → allow longs, block shorts (or down-weight)
       bear  → allow shorts, block longs
       chop  → allow both

  2. Local momentum filter (5m):
       - ema20 vs ema50
       - long_ok:  ema20 > ema50  AND  close > ema20
       - short_ok: ema20 < ema50  AND  close < ema20
     Application: require matching momentum at entry bar.

The MTF mode is "gated, pullback, skip":
  aligned  = 1d filter direction matches trade direction → pass
  pullback = 1d trend + 4h (not used yet) disagrees → allow only mean-reversion
  skip     = 1d or 5m blocks → drop signal

For Iter 1b we keep it SIMPLE: 1d filter (bull/bear/chop) + 5m momentum filter.
4h gate deferred to Iter 2 when we have a retrained 4h 3-class model.

Usage:
    python3 iter1b_backtest_mtf.py \
        --model models/BTCUSDT_15m_v<ts>_3class.json \
        --symbol BTCUSDT --interval 15m \
        --long-thr 0.45 --short-thr 0.45 \
        --years 8 \
        --out-dir logs/iter1b_backtest
"""
import argparse
import json
import math
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_engine import FEATURE_NAMES, build_features, _atr
from data_fetcher import fetch_or_cache

# Single-tier Aggressive config.
TIER = {
    "risk_per_trade_pct": 0.01,
    "sl_atr_mult": 1.2,
    "tp_atr_mult": 2.0,
    "max_trades_per_day": 20,
    "min_vol_pct_by_tf": {"5m": 0.0025, "15m": 0.004, "1h": 0.006, "4h": 0.010, "1d": 0.015},
}
FEE_PER_SIDE = 0.00075
SLIPPAGE = 0.0003
HORIZON_EXIT = {"5m": 48, "15m": 64, "1h": 48, "4h": 36, "1d": 20}


# ──────────────── Filters (rule-based, no ML) ────────────────

def compute_global_trend(daily_df: pd.DataFrame) -> pd.Series:
    """Return Series aligned to daily_df.index with values in {-1, 0, 1}.

    -1 = bear, 0 = chop, 1 = bull. Based on EMA-200 slope + price vs ema200.
    """
    close = daily_df["close"].to_numpy()
    ema200 = pd.Series(close).ewm(span=200, adjust=False).mean().to_numpy()
    trend = np.zeros(len(close), dtype=np.int8)
    for i in range(20, len(close)):
        slope = (ema200[i] - ema200[i - 20]) / (ema200[i - 20] + 1e-12)
        if slope > 0.02 and close[i] > ema200[i]:
            trend[i] = 1
        elif slope < -0.02 and close[i] < ema200[i]:
            trend[i] = -1
        else:
            trend[i] = 0
    # First 20 bars → 0 (unknown, treat as chop → allow both directions).
    return pd.Series(trend, index=pd.to_datetime(daily_df["open_time"]))


def compute_local_momentum(five_min_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return (long_ok, short_ok) Series aligned to five_min_df index.

    long_ok  = ema20 > ema50 AND close > ema20
    short_ok = ema20 < ema50 AND close < ema20
    First 50 bars → both False (warmup).
    """
    close = five_min_df["close"].to_numpy()
    ema20 = pd.Series(close).ewm(span=20, adjust=False).mean().to_numpy()
    ema50 = pd.Series(close).ewm(span=50, adjust=False).mean().to_numpy()
    long_ok = (ema20 > ema50) & (close > ema20)
    short_ok = (ema20 < ema50) & (close < ema20)
    long_ok[:50] = False
    short_ok[:50] = False
    idx = pd.to_datetime(five_min_df["open_time"])
    return pd.Series(long_ok, index=idx), pd.Series(short_ok, index=idx)


def align_filter_to_tf(
    filter_series: pd.Series,
    tf_times: pd.DatetimeIndex,
) -> np.ndarray:
    """For each tf_time, find the most recent filter value at time <= tf_time.

    Prevents look-ahead: 15m bar at 12:00 uses 1d bar from yesterday (or today's
    00:00 open if exists) — never 'tomorrow'. Forward-fill.
    """
    # Reindex + ffill.
    aligned = filter_series.reindex(
        filter_series.index.union(tf_times)
    ).sort_index().ffill()
    return aligned.reindex(tf_times).fillna(0).to_numpy()


# ──────────────── Model loader (reused from iter_backtest_3class) ────────────────

def load_3class_model(path):
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    import lightgbm as lgb

    with open(path) as f:
        payload = json.load(f)
    assert payload.get("version") == "ensemble_v2_3class"

    xgb = XGBClassifier()
    xgb_bytes = bytes.fromhex(payload["xgb_model"])
    xgb.load_model(bytearray(xgb_bytes))

    booster = lgb.Booster(model_str=payload["lgbm_model"])
    lgbm = LGBMClassifier()
    lgbm._Booster = booster
    lgbm._n_classes = 3
    lgbm._classes = np.array([0, 1, 2])
    lgbm._n_features = len(payload["feature_cols"])
    lgbm.fitted_ = True
    lgbm._le = None

    meta = payload.get("meta", {"class_weights": [[1/3]*3]*3})
    return xgb, lgbm, meta, payload


def predict_3class(xgb, lgbm, meta, X):
    p_xgb = xgb.predict_proba(X)
    p_lgbm_raw = lgbm._Booster.predict(X)
    p_lgbm = p_lgbm_raw if p_lgbm_raw.ndim == 2 else np.vstack([1 - p_lgbm_raw, np.zeros(len(X)), np.zeros(len(X))]).T
    base = np.stack([p_xgb, p_lgbm], axis=1)
    w = np.asarray(meta["class_weights"])[:2, :]
    w = w / (w.sum(axis=0, keepdims=True) + 1e-9)
    weighted = base * w[np.newaxis, :, :]
    proba = weighted.sum(axis=1)
    row_sums = proba.sum(axis=1, keepdims=True) + 1e-12
    return proba / row_sums


# ──────────────── MTF simulate ────────────────

def simulate_mtf(feat, df, proba, atr, trend_per_bar, long_mom_per_bar, short_mom_per_bar,
                  long_thr, short_thr, interval, initial_eq=10000.0):
    """Same as iter_backtest_3class.simulate but with 1d and 5m gates."""
    closes = feat["close"].to_numpy().astype(float)
    highs = df["high"].to_numpy().astype(float)
    lows = df["low"].to_numpy().astype(float)
    dates = pd.to_datetime(feat["open_time"].values)
    atr_norm = feat["atr_norm_14"].to_numpy().astype(float)

    min_vol = TIER["min_vol_pct_by_tf"].get(interval, 0.0)
    horizon_bars = HORIZON_EXIT[interval]

    trades = []
    reject = defaultdict(int)
    equity = initial_eq
    trades_today = defaultdict(int)
    n = len(proba)

    i = 0
    while i < n:
        p_long = proba[i, 1]
        p_short = proba[i, 2]
        # Raw signal direction from proba.
        if p_long > long_thr and p_long > p_short:
            direction = 1
        elif p_short > short_thr and p_short > p_long:
            direction = -1
        else:
            reject["hc"] += 1
            i += 1
            continue

        # Vol floor.
        if atr_norm[i] < min_vol:
            reject["vol_floor"] += 1
            i += 1
            continue

        # Global trend gate (1d).
        trend = int(trend_per_bar[i])
        if trend == 1 and direction == -1:
            reject["mtf_1d_bull_blocks_short"] += 1
            i += 1
            continue
        if trend == -1 and direction == 1:
            reject["mtf_1d_bear_blocks_long"] += 1
            i += 1
            continue
        # chop (trend == 0) — allow both.

        # Local momentum gate (5m).
        if direction == 1 and not long_mom_per_bar[i]:
            reject["mtf_5m_momentum_long"] += 1
            i += 1
            continue
        if direction == -1 and not short_mom_per_bar[i]:
            reject["mtf_5m_momentum_short"] += 1
            i += 1
            continue

        # Rate limit.
        day_key = pd.Timestamp(dates[i]).date()
        if trades_today[day_key] >= TIER["max_trades_per_day"]:
            reject["rate_limit"] += 1
            i += 1
            continue

        entry = closes[i]
        atr_i = atr[i] if atr[i] > 0 else entry * 0.005
        slip = entry * (1 + SLIPPAGE) if direction > 0 else entry * (1 - SLIPPAGE)
        if direction > 0:
            sl = slip - TIER["sl_atr_mult"] * atr_i
            tp = slip + TIER["tp_atr_mult"] * atr_i
        else:
            sl = slip + TIER["sl_atr_mult"] * atr_i
            tp = slip - TIER["tp_atr_mult"] * atr_i

        stop_dist = abs(slip - sl) or entry * 0.005
        volume_usd = (equity * TIER["risk_per_trade_pct"]) / stop_dist * slip

        end = min(n, i + 1 + horizon_bars)
        exit_idx = end - 1 if end > i + 1 else i + 1
        exit_price = closes[min(exit_idx, n - 1)]
        exit_reason = "timeout"
        for j in range(i + 1, end):
            if direction > 0:
                if lows[j] <= sl:
                    exit_idx, exit_price, exit_reason = j, sl, "sl"; break
                if highs[j] >= tp:
                    exit_idx, exit_price, exit_reason = j, tp, "tp"; break
            else:
                if highs[j] >= sl:
                    exit_idx, exit_price, exit_reason = j, sl, "sl"; break
                if lows[j] <= tp:
                    exit_idx, exit_price, exit_reason = j, tp, "tp"; break

        if direction > 0:
            raw_pct = (exit_price - slip) / slip
        else:
            raw_pct = (slip - exit_price) / slip
        gross = volume_usd * raw_pct
        fees = volume_usd * FEE_PER_SIDE * 2.0
        pnl = gross - fees
        equity += pnl
        trades.append({
            "entry_idx": i, "exit_idx": exit_idx,
            "entry_date": str(dates[i]), "exit_date": str(dates[min(exit_idx, n - 1)]),
            "direction": direction, "trend_at_entry": trend,
            "p_long": float(p_long), "p_short": float(p_short), "p_hold": float(proba[i, 0]),
            "entry_price": float(slip), "exit_price": float(exit_price),
            "sl": float(sl), "tp": float(tp),
            "raw_pnl_pct": float(raw_pct), "pnl_dollars": float(pnl),
            "fees_paid": float(fees), "equity_after": float(equity),
            "exit_reason": exit_reason, "bars_held": exit_idx - i,
        })
        trades_today[day_key] += 1
        i = exit_idx + 1

    return trades, dict(reject)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--years", type=float, default=8.0)
    ap.add_argument("--long-thr", type=float, default=0.45)
    ap.add_argument("--short-thr", type=float, default=0.45)
    ap.add_argument("--out-dir", default="logs/iter1b_backtest")
    args = ap.parse_args()

    t0 = time.time()
    print(f"=== iter1b_backtest_mtf: {args.symbol} {args.interval} ===")
    print(f"  model: {args.model}")
    print(f"  thresholds: long={args.long_thr}, short={args.short_thr}")
    print(f"  MTF gates: 1d trend + 5m momentum")

    xgb, lgbm, meta, payload = load_3class_model(args.model)
    print(f"  loaded {payload['version']} trained {payload.get('trained_at', '?')}")

    df = fetch_or_cache(args.symbol, args.interval, years=args.years)
    df_1d = fetch_or_cache(args.symbol, "1d", years=args.years)
    df_5m = fetch_or_cache(args.symbol, "5m", years=args.years)

    feat = build_features(df)
    atr_arr = _atr(df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(), period=14)

    # 70/30 split.
    n_all = len(feat)
    split = int(n_all * 0.7)
    feat_oos = feat.iloc[split:].reset_index(drop=True)
    df_oos = df.iloc[split:].reset_index(drop=True)
    atr_oos = atr_arr[split:]

    # Compute filters on full history, then align to OOS.
    trend_series = compute_global_trend(df_1d)
    long_mom, short_mom = compute_local_momentum(df_5m)

    tf_times = pd.to_datetime(feat_oos["open_time"])
    trend_per_bar = align_filter_to_tf(trend_series, tf_times)
    long_mom_per_bar = align_filter_to_tf(long_mom.astype(int), tf_times).astype(bool)
    short_mom_per_bar = align_filter_to_tf(short_mom.astype(int), tf_times).astype(bool)

    X_oos = feat_oos[FEATURE_NAMES].to_numpy()
    proba = predict_3class(xgb, lgbm, meta, X_oos)
    print(f"  proba[long] range: [{proba[:,1].min():.3f}, {proba[:,1].max():.3f}] mean={proba[:,1].mean():.3f}")
    print(f"  proba[short] range: [{proba[:,2].min():.3f}, {proba[:,2].max():.3f}] mean={proba[:,2].mean():.3f}")

    # Filter diagnostics.
    n_bull = int((trend_per_bar == 1).sum())
    n_bear = int((trend_per_bar == -1).sum())
    n_chop = int((trend_per_bar == 0).sum())
    print(f"  1d trend distribution on OOS: bull={n_bull} ({n_bull/len(trend_per_bar):.0%}) "
          f"bear={n_bear} ({n_bear/len(trend_per_bar):.0%}) "
          f"chop={n_chop} ({n_chop/len(trend_per_bar):.0%})")
    print(f"  5m long_ok fraction: {long_mom_per_bar.mean():.0%}  "
          f"short_ok fraction: {short_mom_per_bar.mean():.0%}")

    trades, reject = simulate_mtf(
        feat_oos, df_oos, proba, atr_oos,
        trend_per_bar, long_mom_per_bar, short_mom_per_bar,
        args.long_thr, args.short_thr, args.interval,
    )

    n = len(trades)
    longs = [t for t in trades if t["direction"] == 1]
    shorts = [t for t in trades if t["direction"] == -1]
    lw = sum(1 for t in longs if t["pnl_dollars"] > 0)
    sw = sum(1 for t in shorts if t["pnl_dollars"] > 0)
    total_wins = lw + sw

    from datetime import datetime
    if trades:
        start_d = datetime.fromisoformat(trades[0]["entry_date"][:19])
        end_d = datetime.fromisoformat(trades[-1]["entry_date"][:19])
        days = max(1, (end_d - start_d).days)
    else:
        days = 1

    gross_sum = sum(t["raw_pnl_pct"] for t in trades) * 100
    fees_sum = sum(t["fees_paid"] for t in trades)
    net_dollars = sum(t["pnl_dollars"] for t in trades)
    net_pct = net_dollars / 10000.0 * 100
    fees_pct = fees_sum / 10000.0 * 100

    if trades:
        returns = np.array([t["pnl_dollars"] / 10000.0 for t in trades])
        avg_bars = float(np.mean([t["bars_held"] for t in trades]))
        bars_per_year = {"5m": 365*24*12, "15m": 365*24*4, "1h": 365*24,
                          "4h": 365*6, "1d": 252}.get(args.interval, 252)
        trades_per_year = bars_per_year / avg_bars if avg_bars > 0 else 0
        sharpe = returns.mean() / returns.std() * math.sqrt(trades_per_year) if returns.std() > 1e-9 else 0.0
        eq = np.array([10000.0] + [t["equity_after"] for t in trades])
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / peak
        max_dd = dd.max() * 100
    else:
        sharpe = 0.0
        max_dd = 0.0

    print()
    print("=== Results (with MTF filters) ===")
    print(f"  trades: {n}")
    print(f"  trades/day: {n/days:.2f}")
    print(f"  longs/shorts: {len(longs)}/{len(shorts)} "
          f"(short%={len(shorts)/max(1,n)*100:.0f}%)")
    print(f"  WR long: {lw}/{len(longs)} = {lw/max(1,len(longs)):.1%}")
    print(f"  WR short: {sw}/{len(shorts)} = {sw/max(1,len(shorts)):.1%}")
    print(f"  WR overall: {total_wins}/{n} = {total_wins/max(1,n):.1%}")
    print(f"  gross: {gross_sum:+.2f}% net: {net_pct:+.2f}% fees: ${fees_sum:.0f}")
    ng = net_pct / gross_sum if abs(gross_sum) > 1e-6 else 0.0
    print(f"  net/gross: {ng:.2f}")
    print(f"  Sharpe: {sharpe:+.2f}  max DD: {max_dd:.2f}%")
    print(f"  rejections: {reject}")

    # Trade breakdown per trend regime at entry.
    regime_counts = defaultdict(lambda: [0, 0])
    for t in trades:
        r = t["trend_at_entry"]
        regime_counts[r][0] += 1
        if t["pnl_dollars"] > 0:
            regime_counts[r][1] += 1
    trend_name = {1: "bull", 0: "chop", -1: "bear"}
    print(f"  per-regime at entry:")
    for r in sorted(regime_counts.keys()):
        total, wins = regime_counts[r]
        print(f"    {trend_name[r]:<6}: {total} trades, WR {wins/max(1,total):.1%}")

    checks = {
        "trades/day >= 1.0":  n / days >= 1.0,
        "short% in [30,70]":  30 <= len(shorts) / max(1, n) * 100 <= 70 if n > 0 else False,
        "WR >= 55%":           total_wins / max(1, n) >= 0.55 if n > 0 else False,
        "net/gross >= 0.40":  ng >= 0.40,
        "Sharpe >= 1.0":       sharpe >= 1.0,
    }
    n_pass = sum(1 for v in checks.values() if v)
    print()
    print(f"  Verdict: {n_pass}/{len(checks)} criteria green")
    for k, v in checks.items():
        print(f"    {'PASS' if v else 'FAIL':<5} {k}")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    thr_tag = f"{int(args.long_thr*100):03d}"
    out_file = out_dir / f"{args.symbol}_{args.interval}_iter1b_thr{thr_tag}.json"
    out_file.write_text(json.dumps({
        "symbol": args.symbol, "interval": args.interval, "model": args.model,
        "long_thr": args.long_thr, "short_thr": args.short_thr,
        "n_trades": n, "trades_per_day": n / days,
        "longs": len(longs), "shorts": len(shorts),
        "wr": total_wins / max(1, n),
        "net_pct": net_pct, "gross_pct": gross_sum, "fees_pct": fees_pct,
        "net_gross": ng, "sharpe": sharpe, "max_dd_pct": max_dd,
        "rejections": reject,
        "trend_distribution_oos": {"bull": n_bull, "bear": n_bear, "chop": n_chop},
        "trades": trades,
    }, indent=2, default=str))
    print(f"  saved: {out_file}")
    print(f"  took {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
