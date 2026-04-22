"""
iter_backtest_3class.py — Patch 4 honest backtest for 3-class models.

Simpler than backtest_v2.py: single tier (Aggressive-like), 3-class proba
inference, dual threshold (long_thr, short_thr) independently.

Usage:
    python3 iter_backtest_3class.py \
      --model models/BTCUSDT_15m_v<ts>_3class.json \
      --symbol BTCUSDT --interval 15m \
      --long-thr 0.50 --short-thr 0.50 \
      --years 8
"""
import argparse
import json
import math
import pathlib
import sys
import tempfile
import time
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_engine import FEATURE_NAMES, build_features, _atr
from data_fetcher import fetch_or_cache


# Single-tier "Aggressive" config (matches risk_tiers_config Aggressive).
TIER = {
    "risk_per_trade_pct": 0.01,
    "sl_atr_mult": 1.2,
    "tp_atr_mult": 2.0,
    "max_trades_per_day": 20,
    "min_vol_pct_by_tf": {"5m": 0.0025, "15m": 0.004, "1h": 0.006, "4h": 0.010, "1d": 0.015},
}

FEE_PER_SIDE = 0.00075   # Binance taker 0.075%
SLIPPAGE = 0.0003         # 0.03% at entry

HORIZON_EXIT = {"5m": 48, "15m": 64, "1h": 48, "4h": 36, "1d": 20}


def load_3class_model(path):
    """Load saved 3-class model weights. Uses pickle-free XGBoost/LGBM loaders."""
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier

    with open(path) as f:
        payload = json.load(f)

    assert payload.get("version") == "ensemble_v2_3class", f"not a 3class model: {payload.get('version')}"

    # XGB: save_raw(ubj).hex()  →  load_model from buffer.
    xgb = XGBClassifier()
    xgb_bytes = bytes.fromhex(payload["xgb_model"])
    xgb.load_model(bytearray(xgb_bytes))

    # LGBM: model_to_string() → LGBM Booster(model_str=...)
    import lightgbm as lgb
    booster = lgb.Booster(model_str=payload["lgbm_model"])
    # Wrap as sklearn for predict_proba API.
    lgbm = LGBMClassifier()
    lgbm._Booster = booster
    lgbm._n_classes = 3
    lgbm._classes = np.array([0, 1, 2])
    lgbm._n_features = len(payload["feature_cols"])
    lgbm.fitted_ = True
    lgbm._le = None

    # RF: reconstruct from serialized trees. Too complex to fully rebuild —
    # load existing trees via sklearn's _tree API is fragile. Instead, just
    # refit RF on the fly (slow but reliable). User can skip by passing --skip-rf.
    rf = None

    meta = payload.get("meta", {"kind": "soft_voting_3class", "class_weights": [[1/3]*3]*3})
    return xgb, lgbm, rf, meta, payload


def predict_3class(xgb, lgbm, rf, meta, X, skip_rf=True):
    """Soft-voting ensemble. If rf is None, evenly split weights between xgb and lgbm."""
    p_xgb = xgb.predict_proba(X)
    p_lgbm_raw = lgbm._Booster.predict(X)
    # LGBM booster.predict returns (n, 3) for multiclass.
    p_lgbm = p_lgbm_raw if p_lgbm_raw.ndim == 2 else np.vstack([1-p_lgbm_raw, np.zeros(len(X)), np.zeros(len(X))]).T
    if rf is not None and not skip_rf:
        p_rf = rf.predict_proba(X)
        base = np.stack([p_xgb, p_lgbm, p_rf], axis=1)
        w = np.asarray(meta["class_weights"])  # (3, 3)
    else:
        base = np.stack([p_xgb, p_lgbm], axis=1)
        # Use only first 2 rows of weights, renormalize.
        w = np.asarray(meta["class_weights"])[:2, :]
        w = w / (w.sum(axis=0, keepdims=True) + 1e-9)
    weighted = base * w[np.newaxis, :, :]
    proba = weighted.sum(axis=1)
    row_sums = proba.sum(axis=1, keepdims=True) + 1e-12
    return proba / row_sums


def simulate(feat, df, proba, atr, long_thr, short_thr, interval, initial_eq=10000.0):
    closes = feat["close"].to_numpy().astype(float)
    highs = df["high"].to_numpy().astype(float)
    lows = df["low"].to_numpy().astype(float)
    dates = pd.to_datetime(feat["open_time"].values)
    atr_norm = feat["atr_norm_14"].to_numpy().astype(float)

    min_vol = TIER["min_vol_pct_by_tf"].get(interval, 0.0)
    horizon_bars = HORIZON_EXIT[interval]

    trades = []
    reject_vol = 0
    reject_rate = 0
    reject_hc = 0
    equity = initial_eq
    trades_today = defaultdict(int)
    n = len(proba)

    i = 0
    while i < n:
        p_hold, p_long, p_short = proba[i, 0], proba[i, 1], proba[i, 2]
        if p_long > long_thr and p_long > p_short:
            direction = 1
        elif p_short > short_thr and p_short > p_long:
            direction = -1
        else:
            reject_hc += 1
            i += 1
            continue

        if atr_norm[i] < min_vol:
            reject_vol += 1
            i += 1
            continue

        day_key = pd.Timestamp(dates[i]).date()
        if trades_today[day_key] >= TIER["max_trades_per_day"]:
            reject_rate += 1
            i += 1
            continue

        entry = closes[i]
        atr_i = atr[i] if atr[i] > 0 else entry * 0.005
        slip_entry = entry * (1 + SLIPPAGE) if direction > 0 else entry * (1 - SLIPPAGE)
        if direction > 0:
            sl = slip_entry - TIER["sl_atr_mult"] * atr_i
            tp = slip_entry + TIER["tp_atr_mult"] * atr_i
        else:
            sl = slip_entry + TIER["sl_atr_mult"] * atr_i
            tp = slip_entry - TIER["tp_atr_mult"] * atr_i

        stop_dist = abs(slip_entry - sl) or entry * 0.005
        volume_usd = (equity * TIER["risk_per_trade_pct"]) / stop_dist * slip_entry

        end = min(n, i + 1 + horizon_bars)
        exit_idx = end - 1 if end > i + 1 else i + 1
        exit_price = closes[min(exit_idx, n - 1)]
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

        if direction > 0:
            raw_pct = (exit_price - slip_entry) / slip_entry
        else:
            raw_pct = (slip_entry - exit_price) / slip_entry
        gross_dollars = volume_usd * raw_pct
        fees = volume_usd * FEE_PER_SIDE * 2.0
        pnl = gross_dollars - fees
        equity += pnl
        trades.append({
            "entry_idx": i,
            "exit_idx": exit_idx,
            "entry_date": str(dates[i]),
            "exit_date": str(dates[min(exit_idx, n - 1)]),
            "direction": direction,
            "p_long": float(p_long),
            "p_short": float(p_short),
            "p_hold": float(p_hold),
            "entry_price": float(slip_entry),
            "exit_price": float(exit_price),
            "sl": float(sl),
            "tp": float(tp),
            "raw_pnl_pct": float(raw_pct),
            "pnl_dollars": float(pnl),
            "fees_paid": float(fees),
            "equity_after": float(equity),
            "exit_reason": exit_reason,
            "bars_held": exit_idx - i,
        })
        trades_today[day_key] += 1
        i = exit_idx + 1

    return trades, {
        "reject_hc": reject_hc,
        "reject_vol": reject_vol,
        "reject_rate": reject_rate,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--years", type=float, default=8.0)
    ap.add_argument("--long-thr", type=float, default=0.50)
    ap.add_argument("--short-thr", type=float, default=0.50)
    ap.add_argument("--skip-rf", action="store_true", default=True)
    ap.add_argument("--out-dir", default="logs/iter_backtest")
    args = ap.parse_args()

    t0 = time.time()
    print(f"=== iter_backtest_3class: {args.symbol} {args.interval} ===")
    print(f"  model: {args.model}")
    print(f"  thresholds: long={args.long_thr}, short={args.short_thr}")
    print(f"  fees: {FEE_PER_SIDE*200:.2f}% round trip, slippage {SLIPPAGE*100:.2f}% at entry")

    xgb, lgbm, rf, meta, payload = load_3class_model(args.model)
    print(f"  loaded model v={payload['version']} trained {payload.get('trained_at', '?')}")

    df = fetch_or_cache(args.symbol, args.interval, years=args.years)
    feat = build_features(df)
    atr_arr = _atr(df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(), period=14)

    # 70/30 split — last 30% = OOS.
    n_all = len(feat)
    split = int(n_all * 0.7)
    feat_oos = feat.iloc[split:].reset_index(drop=True)
    df_oos = df.iloc[split:].reset_index(drop=True)
    atr_oos = atr_arr[split:]

    # Use the feature_cols stored in the model payload (not live FEATURE_NAMES
    # which may have grown — e.g. Patch 4 Night added 13 volume features,
    # but models trained before that import only the original 32).
    model_feature_cols = payload.get("feature_cols", FEATURE_NAMES)
    X_oos = feat_oos[model_feature_cols].to_numpy()
    print(f"  OOS rows: {len(X_oos)} ({feat_oos['open_time'].iloc[0]} .. {feat_oos['open_time'].iloc[-1]})")
    print(f"  model expects {len(model_feature_cols)} features, feature_engine provides {len(FEATURE_NAMES)}")

    proba = predict_3class(xgb, lgbm, rf, meta, X_oos, skip_rf=args.skip_rf)
    print(f"  proba[long] range: [{proba[:,1].min():.3f}, {proba[:,1].max():.3f}] mean={proba[:,1].mean():.3f}")
    print(f"  proba[short] range: [{proba[:,2].min():.3f}, {proba[:,2].max():.3f}] mean={proba[:,2].mean():.3f}")
    print(f"  proba[hold] range: [{proba[:,0].min():.3f}, {proba[:,0].max():.3f}] mean={proba[:,0].mean():.3f}")

    trades, rejections = simulate(feat_oos, df_oos, proba, atr_oos,
                                   args.long_thr, args.short_thr,
                                   args.interval)

    n = len(trades)
    longs = [t for t in trades if t["direction"] == 1]
    shorts = [t for t in trades if t["direction"] == -1]
    lw = sum(1 for t in longs if t["pnl_dollars"] > 0)
    sw = sum(1 for t in shorts if t["pnl_dollars"] > 0)
    total_wins = lw + sw

    from datetime import datetime
    if trades:
        start = datetime.fromisoformat(trades[0]["entry_date"][:19])
        end = datetime.fromisoformat(trades[-1]["entry_date"][:19])
        days = max(1, (end - start).days)
    else:
        days = 1

    gross_sum = sum(t["raw_pnl_pct"] for t in trades) * 100
    fees_sum = sum(t["fees_paid"] for t in trades)
    net_dollars = sum(t["pnl_dollars"] for t in trades)
    net_pct = net_dollars / 10000.0 * 100
    initial_eq = 10000.0
    fees_pct = fees_sum / initial_eq * 100

    if trades:
        returns = np.array([t["pnl_dollars"] / initial_eq for t in trades])
        avg_bars = float(np.mean([t["bars_held"] for t in trades]))
        bars_per_year = {"5m": 365*24*12, "15m": 365*24*4, "1h": 365*24,
                          "4h": 365*6, "1d": 252}.get(args.interval, 252)
        trades_per_year = bars_per_year / avg_bars if avg_bars > 0 else 0
        if returns.std() > 1e-9:
            sharpe = returns.mean() / returns.std() * math.sqrt(trades_per_year)
        else:
            sharpe = 0.0

        equity_curve = np.array([initial_eq] + [t["equity_after"] for t in trades])
        peak = np.maximum.accumulate(equity_curve)
        dd = (peak - equity_curve) / peak
        max_dd = dd.max() * 100
    else:
        sharpe = 0.0
        max_dd = 0.0

    print()
    print("=== Results ===")
    print(f"  trades: {n}")
    print(f"  trades/day: {n/days:.2f}  (target >= 1.0)")
    print(f"  longs/shorts: {len(longs)}/{len(shorts)} "
          f"(short% = {len(shorts)/max(1,n)*100:.0f}%; target 30-70%)")
    print(f"  WR long: {lw}/{len(longs)} = {lw/max(1,len(longs)):.1%}")
    print(f"  WR short: {sw}/{len(shorts)} = {sw/max(1,len(shorts)):.1%}")
    print(f"  WR overall: {total_wins}/{n} = {total_wins/max(1,n):.1%}  (target >= 55%)")
    print(f"  gross return: {gross_sum:+.2f}%")
    print(f"  net return: {net_pct:+.2f}%")
    print(f"  fees: ${fees_sum:.0f} ({fees_pct:.2f}% initial)")
    if abs(gross_sum) > 1e-6:
        ng = net_pct / gross_sum
    else:
        ng = 0.0
    print(f"  net/gross: {ng:.2f}  (target >= 0.40)")
    print(f"  Sharpe (after fees): {sharpe:+.2f}  (target >= 1.0)")
    print(f"  max DD: {max_dd:.2f}%")
    print(f"  rejections: hc={rejections['reject_hc']} vol={rejections['reject_vol']} rate={rejections['reject_rate']}")

    # Pass/fail.
    checks = {
        "trades/day >= 1.0":  n / days >= 1.0,
        "short% in [30,70]":  30 <= len(shorts) / max(1, n) * 100 <= 70,
        "WR >= 55%":           total_wins / max(1, n) >= 0.55,
        "net/gross >= 0.40":  ng >= 0.40,
        "Sharpe >= 1.0":       sharpe >= 1.0,
    }
    n_pass = sum(1 for v in checks.values() if v)
    print()
    print(f"  Verdict: {n_pass}/{len(checks)} criteria green")
    for k, v in checks.items():
        print(f"    {'PASS' if v else 'FAIL':<5} {k}")

    # Save full trade dump.
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.symbol}_{args.interval}_iter_3class.json"
    out_file.write_text(json.dumps({
        "symbol": args.symbol,
        "interval": args.interval,
        "model": args.model,
        "long_thr": args.long_thr,
        "short_thr": args.short_thr,
        "n_trades": n,
        "trades_per_day": n / days,
        "longs": len(longs), "shorts": len(shorts),
        "wr": total_wins / max(1, n),
        "net_pct": net_pct,
        "gross_pct": gross_sum,
        "fees_pct": fees_pct,
        "net_gross": ng,
        "sharpe": sharpe,
        "max_dd_pct": max_dd,
        "rejections": rejections,
        "trades": trades,
    }, indent=2, default=str))
    print(f"  saved: {out_file}")
    print(f"  took {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
