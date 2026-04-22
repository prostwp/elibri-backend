"""iter1_wfcv.py — Walk-forward CV verification of a promising config.

After finding a working hypothesis via single-holdout (e.g. exp4h_sym with
Sharpe +1.23), this script repeats it across 3-5 sequential time folds to
check stability. Avoids overfit-to-holdout.

Usage:
  python3 iter1_wfcv.py --interval 4h --tb-upper 1.5 --tb-lower 1.5 \
         --label wfcv_4h --n-folds 5 --oos-months 3

Output: logs/iter1_wfcv_<label>.json with per-fold + aggregate metrics.
"""
import argparse
import json
import math
import pathlib
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iter1_exp_b_sample_weights import (
    TIER, FEE_PER_SIDE, SLIPPAGE, HORIZON_EXIT,
    regime_weight_year_tiered, train_weighted_3class, ensemble_predict,
    simulate,
)
from feature_engine import (
    FEATURE_NAMES, build_features,
    make_target_triple_class,
    _atr,
)
from data_fetcher import fetch_or_cache
from train import TF_CONFIG


def run_wfcv(symbol="BTCUSDT", interval="4h",
             tb_upper=None, tb_lower=None, horizon=None,
             n_folds=5, oos_months=3, label="wfcv",
             initial_train_months=36, weighted=True):
    """Sliding walk-forward CV.

    Fold 0: train = 2017-08 → start_test_0 (initial_train_months back),
            test  = [start_test_0, start_test_0 + oos_months)
    Fold k: train ends at start_test_k = start_test_0 + k*oos_months
            test  = [start_test_k, start_test_k + oos_months)

    Final fold's test_end = initial_train_end + n_folds * oos_months.
    """
    t0 = time.time()
    tf_cfg = dict(TF_CONFIG[interval])
    if tb_upper is not None:
        tf_cfg["tb_upper"] = tb_upper
    if tb_lower is not None:
        tf_cfg["tb_lower"] = tb_lower
    if horizon is not None:
        tf_cfg["horizon"] = horizon
    horizon_v = tf_cfg["horizon"]
    tb_u = tf_cfg["tb_upper"]
    tb_l = tf_cfg["tb_lower"]

    print(f"\n=== WFCV {label}: {symbol} {interval}, tb={tb_u}/{tb_l}, h={horizon_v}, folds={n_folds}, oos={oos_months}mo ===")

    df = fetch_or_cache(symbol, interval, years=8.0)
    feat = build_features(df)
    atr_arr = _atr(df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(), period=14)
    close_arr = df["close"].to_numpy()
    high_arr = df["high"].to_numpy()
    low_arr = df["low"].to_numpy()
    target = make_target_triple_class(
        high_arr, low_arr, close_arr, atr_arr, horizon=horizon_v,
        upper_mult=tb_u, lower_mult=tb_l,
    )

    mask_labeled = target >= 0
    feat_L = feat.loc[mask_labeled].reset_index(drop=True)
    target_L = target[mask_labeled]
    atr_L = atr_arr[mask_labeled]
    df_L = df.loc[mask_labeled].reset_index(drop=True)

    ot = pd.to_datetime(feat_L["open_time"], utc=True)
    ts_min = ot.min()
    ts_max = ot.max()

    # First fold starts at ts_min + initial_train_months
    start_test_0 = (ts_min + pd.Timedelta(days=initial_train_months * 30)).normalize()
    print(f"  data range: {ts_min.date()} → {ts_max.date()}")
    print(f"  first test fold starts: {start_test_0.date()}")

    per_fold = []
    all_trades_combined = []

    for fold in range(n_folds):
        test_start = start_test_0 + pd.Timedelta(days=fold * oos_months * 30)
        test_end = test_start + pd.Timedelta(days=oos_months * 30)
        if test_end > ts_max:
            print(f"  fold {fold}: test_end {test_end.date()} > data end {ts_max.date()}, skip")
            continue

        train_mask = ot < test_start
        test_mask = (ot >= test_start) & (ot < test_end)

        if train_mask.sum() < 500 or test_mask.sum() < 50:
            print(f"  fold {fold}: insufficient rows {train_mask.sum()}/{test_mask.sum()}, skip")
            continue

        print(f"\n  -- fold {fold}: train to {test_start.date()} ({int(train_mask.sum())}), test {test_start.date()}→{test_end.date()} ({int(test_mask.sum())})")

        X_train = feat_L.loc[train_mask, FEATURE_NAMES].to_numpy()
        y_train = target_L[train_mask.to_numpy()]
        train_dates = feat_L.loc[train_mask, "open_time"].to_numpy()

        if weighted:
            sw = regime_weight_year_tiered(train_dates)
        else:
            sw = None

        hold_pct = float((y_train == 0).mean())
        long_pct = float((y_train == 1).mean())
        short_pct = float((y_train == 2).mean())
        print(f"    class dist: hold={hold_pct:.1%}, long={long_pct:.1%}, short={short_pct:.1%}")

        tf_overrides = {"n_est": tf_cfg.get("n_est", 200), "xgb_depth": tf_cfg.get("xgb_depth", 5)}
        tf0 = time.time()
        xgb, lgbm, rf = train_weighted_3class(X_train, y_train, sw, tf_overrides)
        print(f"    train time: {time.time() - tf0:.1f}s")

        # Test.
        X_test = feat_L.loc[test_mask, FEATURE_NAMES].to_numpy()
        proba_test = ensemble_predict(xgb, lgbm, rf, X_test)

        test_feat = feat_L.loc[test_mask].reset_index(drop=True)
        test_df = df_L.loc[test_mask.to_numpy()].reset_index(drop=True)
        test_atr = atr_L[test_mask.to_numpy()]

        # Backtest at thr=0.40 (best from single-holdout; also try 0.50).
        results_thr = {}
        for thr in [0.50, 0.40]:
            trades, rej = simulate(test_feat, test_df, proba_test, test_atr,
                                    long_thr=thr, short_thr=thr, interval=interval)
            if not trades:
                results_thr[f"thr_{int(thr*100):02d}"] = {
                    "threshold": thr, "n_trades": 0, "wr": 0, "sharpe": 0, "net_pct": 0,
                }
                continue
            longs = [t for t in trades if t["direction"] == 1]
            shorts = [t for t in trades if t["direction"] == -1]
            wins = sum(1 for t in trades if t["pnl_dollars"] > 0)
            net_dollars = sum(t["pnl_dollars"] for t in trades)
            net_pct = net_dollars / 10000.0 * 100
            returns = np.array([t["pnl_dollars"] / 10000.0 for t in trades])
            avg_bars = float(np.mean([t["bars_held"] for t in trades]))
            bars_per_year = {"5m": 365*24*12, "15m": 365*24*4, "1h": 365*24,
                              "4h": 365*6, "1d": 252}.get(interval, 252)
            trades_per_year = bars_per_year / avg_bars if avg_bars > 0 else 0
            sharpe = returns.mean() / returns.std() * math.sqrt(trades_per_year) if returns.std() > 1e-9 else 0.0
            results_thr[f"thr_{int(thr*100):02d}"] = {
                "threshold": thr,
                "n_trades": len(trades),
                "longs": len(longs),
                "shorts": len(shorts),
                "wr": wins / max(1, len(trades)),
                "net_pct": net_pct,
                "sharpe": sharpe,
                "trades": trades[:30],  # sample
            }
            # Save for combined aggregation.
            for t in trades:
                t_copy = dict(t)
                t_copy["fold"] = fold
                t_copy["threshold"] = thr
                all_trades_combined.append(t_copy)
            print(f"    thr={thr}: {len(trades)} trades ({len(longs)}L/{len(shorts)}S), WR={wins}/{len(trades)}={wins/max(1,len(trades))*100:.1f}%, Sharpe={sharpe:+.2f}, net={net_pct:+.2f}%")

        per_fold.append({
            "fold": fold,
            "train_end": str(test_start),
            "test_start": str(test_start),
            "test_end": str(test_end),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "class_dist": {"hold": hold_pct, "long": long_pct, "short": short_pct},
            "results_by_thr": results_thr,
        })

    # Aggregate.
    agg = {"thr_50": {"trades": 0, "wr_sum": 0, "sharpe_sum": 0, "net_sum": 0, "folds": 0},
           "thr_40": {"trades": 0, "wr_sum": 0, "sharpe_sum": 0, "net_sum": 0, "folds": 0}}
    for f in per_fold:
        for k, r in f["results_by_thr"].items():
            if r["n_trades"] > 0:
                agg[k]["trades"] += r["n_trades"]
                agg[k]["wr_sum"] += r["wr"]
                agg[k]["sharpe_sum"] += r["sharpe"]
                agg[k]["net_sum"] += r["net_pct"]
                agg[k]["folds"] += 1

    summary = {
        "label": label,
        "symbol": symbol,
        "interval": interval,
        "tb_upper": tb_u,
        "tb_lower": tb_l,
        "horizon": horizon_v,
        "n_folds": n_folds,
        "folds_used": len(per_fold),
        "oos_months": oos_months,
        "initial_train_months": initial_train_months,
        "weighted": weighted,
        "per_fold": per_fold,
        "aggregate": agg,
    }

    out_path = ROOT / "logs" / f"iter1_wfcv_{label}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  saved: {out_path}")
    print(f"  total time: {time.time() - t0:.1f}s")

    # Print aggregate summary.
    print("\n=== AGGREGATE over folds ===")
    for k, v in agg.items():
        if v["folds"] > 0:
            avg_wr = v["wr_sum"] / v["folds"] * 100
            avg_sharpe = v["sharpe_sum"] / v["folds"]
            avg_net = v["net_sum"] / v["folds"]
            total_net = v["net_sum"]
            print(f"  {k}: {v['folds']} folds, total trades={v['trades']}, "
                  f"avg WR={avg_wr:.1f}%, avg Sharpe={avg_sharpe:+.2f}, "
                  f"avg net/fold={avg_net:+.2f}%, total net={total_net:+.2f}%")

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="wfcv_4h")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="4h")
    ap.add_argument("--tb-upper", type=float, default=None)
    ap.add_argument("--tb-lower", type=float, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--oos-months", type=int, default=3)
    ap.add_argument("--initial-train-months", type=int, default=36)
    ap.add_argument("--no-weights", action="store_true")
    args = ap.parse_args()
    run_wfcv(
        symbol=args.symbol, interval=args.interval,
        tb_upper=args.tb_upper, tb_lower=args.tb_lower, horizon=args.horizon,
        n_folds=args.n_folds, oos_months=args.oos_months,
        initial_train_months=args.initial_train_months,
        label=args.label,
        weighted=not args.no_weights,
    )


if __name__ == "__main__":
    main()
