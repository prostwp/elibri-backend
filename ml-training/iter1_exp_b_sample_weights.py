"""iter1_exp_b_sample_weights.py — Patch 4 Iter 1 Experiment B.

Retrains BTCUSDT 15m with 3-class target AND linearly-growing sample weights
by year (2017-2021 w=1.0, 2022-2023 w=2.0, 2024-2026 w=3.0). The idea is to
let the model use all 8 years of data but lean on recent (institutional-era)
patterns at a higher weight.

Train: all 8 years (weighted).
Test: final 6 months (2025-10 → 2026-04). No walk-forward CV for speed —
just one holdout split.

Outputs:
  models/BTCUSDT_15m_expB_v<ts>.json  — sample-weighted model
  logs/iter1_exp_b_metrics.json       — train-test metrics
  logs/iter1_exp_b_trades.json         — per-trade backtest on holdout (ready
                                         for iter_analyze to compute the
                                         morning table)

Runs on vast.ai. Expects feature_engine and train.py already updated for
Patch 4 3-class support.
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
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

ROOT = pathlib.Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_engine import (
    FEATURE_NAMES, build_features,
    make_target_triple_class, TRIPLE_CLASS_HOLD, TRIPLE_CLASS_LONG, TRIPLE_CLASS_SHORT,
    _atr,
)
from data_fetcher import fetch_or_cache
from train import (
    TF_CONFIG, HORIZON_MAP,
    _get_xgb_device, _get_lgbm_device,
)


# Same per-tier config as iter_backtest_3class.
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


def regime_weight_year_tiered(dates: np.ndarray) -> np.ndarray:
    """Return per-bar weight using tiered scheme from user.

    2017-2021 → 1.0
    2022-2023 → 2.0
    2024-2026 → 3.0
    """
    ts = pd.to_datetime(dates, utc=True)
    years = np.asarray(ts.year)
    w = np.ones_like(years, dtype=np.float32)
    w = np.where((years >= 2022) & (years <= 2023), 2.0, w)
    w = np.where(years >= 2024, 3.0, w)
    return w


def train_weighted_3class(X_train, y_train, sample_weight, tf_overrides):
    """Train 3-class ensemble with sample_weight on all bases.

    Short-circuits OOF CV — uses simple holdout metrics outside this
    function (we only need the final full-data model for the backtest).
    """
    import os
    n_est = tf_overrides.get("n_est", 200)
    xgb_depth = tf_overrides.get("xgb_depth", 5)
    n_jobs = int(os.getenv("ML_N_JOBS", "4"))

    xgb_kw = _get_xgb_device()
    lgbm_kw = _get_lgbm_device()

    xgb = XGBClassifier(
        n_estimators=n_est, max_depth=xgb_depth, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", n_jobs=n_jobs, random_state=42,
        tree_method="hist", verbosity=0,
        **xgb_kw,
    )
    lgbm = LGBMClassifier(
        n_estimators=n_est, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multiclass", num_class=3,
        n_jobs=n_jobs, random_state=42, verbose=-1,
        **lgbm_kw,
    )
    rf = RandomForestClassifier(
        n_estimators=n_est, max_depth=10,
        n_jobs=n_jobs, random_state=42,
    )

    print(f"  fitting XGB (weighted, {len(X_train)} rows, {xgb_kw or 'CPU'})...")
    xgb.fit(X_train, y_train, sample_weight=sample_weight)
    print(f"  fitting LGBM (weighted)...")
    lgbm.fit(X_train, y_train, sample_weight=sample_weight)
    print(f"  fitting RF (weighted)...")
    rf.fit(X_train, y_train, sample_weight=sample_weight)

    return xgb, lgbm, rf


def ensemble_predict(xgb, lgbm, rf, X):
    """Simple equal-weight soft voting (no per-class meta since no OOF)."""
    p_xgb = xgb.predict_proba(X)
    p_lgbm = lgbm.predict_proba(X)
    p_rf = rf.predict_proba(X)
    proba = (p_xgb + p_lgbm + p_rf) / 3.0
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
    reject = defaultdict(int)
    equity = initial_eq
    trades_today = defaultdict(int)
    n = len(proba)

    i = 0
    while i < n:
        p_long, p_short = proba[i, 1], proba[i, 2]
        if p_long > long_thr and p_long > p_short:
            direction = 1
        elif p_short > short_thr and p_short > p_long:
            direction = -1
        else:
            reject["hc"] += 1
            i += 1
            continue
        if atr_norm[i] < min_vol:
            reject["vol_floor"] += 1
            i += 1
            continue
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
            "direction": direction,
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


def run_experiment(symbol="BTCUSDT", interval="15m",
                   train_start=None, train_end=None,
                   test_start=None, test_end=None,
                   weighted=True, label="expB",
                   tb_upper=None, tb_lower=None, horizon=None):
    """Single-holdout training + backtest, optionally with sample weights.

    If train_start/end are None → use full available history up to test_start.
    tb_upper/tb_lower override TF_CONFIG barriers (symmetric target experiment D).
    horizon overrides TF_CONFIG horizon (experiment F).
    """
    t0 = time.time()
    tf_cfg = dict(TF_CONFIG[interval])
    if tb_upper is not None:
        tf_cfg["tb_upper"] = tb_upper
    if tb_lower is not None:
        tf_cfg["tb_lower"] = tb_lower
    if horizon is not None:
        tf_cfg["horizon"] = horizon
    horizon = tf_cfg["horizon"]

    print(f"\n=== Experiment {label}: {symbol} {interval}, weighted={weighted} ===")
    print(f"  horizon={horizon}, tb={tf_cfg['tb_upper']}/{tf_cfg['tb_lower']}")

    df = fetch_or_cache(symbol, interval, years=8.0)
    feat = build_features(df)
    atr_arr = _atr(df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(), period=14)
    close_arr = df["close"].to_numpy()
    high_arr = df["high"].to_numpy()
    low_arr = df["low"].to_numpy()
    target = make_target_triple_class(
        high_arr, low_arr, close_arr, atr_arr, horizon=horizon,
        upper_mult=tf_cfg["tb_upper"], lower_mult=tf_cfg["tb_lower"],
    )

    # Drop unlabeled (-1).
    mask_labeled = target >= 0
    feat_L = feat.loc[mask_labeled].reset_index(drop=True)
    target_L = target[mask_labeled]

    # Resolve window boundaries.
    ot = pd.to_datetime(feat_L["open_time"], utc=True)
    ts_min = ot.min()
    ts_max = ot.max()
    test_start_ts = pd.Timestamp(test_start, tz="UTC") if test_start else None
    test_end_ts = pd.Timestamp(test_end, tz="UTC") if test_end else ts_max + pd.Timedelta(days=1)
    train_start_ts = pd.Timestamp(train_start, tz="UTC") if train_start else ts_min
    train_end_ts = pd.Timestamp(train_end, tz="UTC") if train_end else test_start_ts

    train_mask = (ot >= train_start_ts) & (ot < train_end_ts)
    test_mask = (ot >= test_start_ts) & (ot < test_end_ts)

    print(f"  train window: {train_start_ts.date()} → {train_end_ts.date()} "
          f"({int(train_mask.sum())} rows)")
    print(f"  test window:  {test_start_ts.date()} → {test_end_ts.date()} "
          f"({int(test_mask.sum())} rows)")

    if train_mask.sum() < 500 or test_mask.sum() < 100:
        print(f"  ERROR: insufficient rows")
        return None

    X_train = feat_L.loc[train_mask, FEATURE_NAMES].to_numpy()
    y_train = target_L[train_mask.to_numpy()]
    train_dates = feat_L.loc[train_mask, "open_time"].to_numpy()

    if weighted:
        sw = regime_weight_year_tiered(train_dates)
        print(f"  sample weights: 1.0/{int((sw == 1).sum())} rows, "
              f"2.0/{int((sw == 2).sum())} rows, 3.0/{int((sw == 3).sum())} rows")
    else:
        sw = None

    hold_rate = float((y_train == 0).mean())
    long_rate = float((y_train == 1).mean())
    short_rate = float((y_train == 2).mean())
    print(f"  train class distribution: hold={hold_rate:.1%}, "
          f"long={long_rate:.1%}, short={short_rate:.1%}")

    tf_overrides = {"n_est": tf_cfg["n_est"], "xgb_depth": tf_cfg["xgb_depth"]}
    xgb, lgbm, rf = train_weighted_3class(X_train, y_train, sw, tf_overrides)
    train_time = time.time() - t0
    print(f"  train time: {train_time:.1f}s")

    # Test set proba.
    X_test = feat_L.loc[test_mask, FEATURE_NAMES].to_numpy()
    y_test = target_L[test_mask.to_numpy()]
    proba_test = ensemble_predict(xgb, lgbm, rf, X_test)
    print(f"  test proba[long]: min={proba_test[:,1].min():.3f} max={proba_test[:,1].max():.3f} "
          f"mean={proba_test[:,1].mean():.3f}")
    print(f"  test proba[short]: min={proba_test[:,2].min():.3f} max={proba_test[:,2].max():.3f} "
          f"mean={proba_test[:,2].mean():.3f}")

    # Backtest at 3 thresholds.
    test_feat = feat_L.loc[test_mask].reset_index(drop=True)
    test_df = df.loc[mask_labeled].iloc[test_mask.to_numpy()].reset_index(drop=True)
    test_atr = atr_arr[mask_labeled][test_mask.to_numpy()]

    results = {}
    for thr in [0.50, 0.45, 0.40]:
        trades, rej = simulate(test_feat, test_df, proba_test, test_atr,
                                long_thr=thr, short_thr=thr, interval=interval)
        n = len(trades)
        longs = [t for t in trades if t["direction"] == 1]
        shorts = [t for t in trades if t["direction"] == -1]
        lw = sum(1 for t in longs if t["pnl_dollars"] > 0)
        sw_ = sum(1 for t in shorts if t["pnl_dollars"] > 0)
        total_wins = lw + sw_

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
        if trades:
            returns = np.array([t["pnl_dollars"] / 10000.0 for t in trades])
            avg_bars = float(np.mean([t["bars_held"] for t in trades]))
            bars_per_year = {"5m": 365*24*12, "15m": 365*24*4, "1h": 365*24,
                              "4h": 365*6, "1d": 252}.get(interval, 252)
            trades_per_year = bars_per_year / avg_bars if avg_bars > 0 else 0
            sharpe = returns.mean() / returns.std() * math.sqrt(trades_per_year) if returns.std() > 1e-9 else 0.0
        else:
            sharpe = 0.0

        ng = net_pct / gross_sum if abs(gross_sum) > 1e-6 else 0.0
        result = {
            "threshold": thr,
            "n_trades": n,
            "longs": len(longs),
            "shorts": len(shorts),
            "short_pct": (len(shorts) / max(1, n)) * 100,
            "wr_long": lw / max(1, len(longs)),
            "wr_short": sw_ / max(1, len(shorts)),
            "wr_overall": total_wins / max(1, n),
            "gross_pct": gross_sum,
            "net_pct": net_pct,
            "fees_dollars": fees_sum,
            "net_gross": ng,
            "sharpe": sharpe,
            "trades_per_day": n / days,
            "rejections": rej,
            "sample_trades": trades[:50],  # truncate
        }
        results[f"thr_{int(thr*100):02d}"] = result
        print(f"\n  thr={thr}: {n} trades, L/S={len(longs)}/{len(shorts)} "
              f"(short%={result['short_pct']:.0f}%), "
              f"WR={total_wins}/{n}={result['wr_overall']*100:.1f}%, "
              f"Sharpe={sharpe:+.2f}, net={net_pct:+.2f}%, ng={ng:.2f}")

    # Save model + results.
    ts = int(time.time())
    model_path = ROOT / "models" / f"{symbol}_{interval}_{label}_v{ts}.json"
    summary_path = ROOT / "logs" / f"iter1_{label}_metrics.json"
    summary = {
        "label": label, "symbol": symbol, "interval": interval,
        "weighted": weighted,
        "train_window": [str(train_start_ts), str(train_end_ts)],
        "test_window": [str(test_start_ts), str(test_end_ts)],
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "train_time_sec": train_time,
        "class_dist": {"hold": hold_rate, "long": long_rate, "short": short_rate},
        "results_by_thr": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  saved: {summary_path}")
    print(f"  total time: {time.time() - t0:.1f}s")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="expB", help="expA / expB / expC")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--train-start", default=None)
    ap.add_argument("--train-end", default=None)
    ap.add_argument("--test-start", default=None)
    ap.add_argument("--test-end", default=None)
    ap.add_argument("--no-weights", action="store_true", help="Disable sample weights")
    ap.add_argument("--tb-upper", type=float, default=None,
                    help="Override tb_upper (default from TF_CONFIG)")
    ap.add_argument("--tb-lower", type=float, default=None,
                    help="Override tb_lower (default from TF_CONFIG)")
    ap.add_argument("--horizon", type=int, default=None,
                    help="Override horizon (default from TF_CONFIG)")
    args = ap.parse_args()

    run_experiment(
        symbol=args.symbol, interval=args.interval,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        weighted=not args.no_weights,
        label=args.label,
        tb_upper=args.tb_upper, tb_lower=args.tb_lower,
        horizon=args.horizon,
    )


if __name__ == "__main__":
    main()
