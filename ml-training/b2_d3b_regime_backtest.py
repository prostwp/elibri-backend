"""
b2_d3b_regime_backtest.py — Session B.2 d3b regime-split backtest.

Phase 3 backtest_v2 used default 70/30 split → OOS holdout = 2024-02..2026-03
= 100% bull. That window is useless for regime-robustness validation (can't
prove the model survives a bear).

This helper runs a custom backtest using the full Phase 1-trained BTCUSDT_4h
model (already in models/latest.json), feeds it 2022-05-01..2024-01-01 candles
(covers BEAR 2022 and CHOP 2023), applies the same tier gates + fees as
backtest_v2, and reports per-regime Sharpe/WR/direction balance.

Key difference from backtest_v2:
  - Uses production-trained model (loads saved JSON), NOT retrain on a split.
  - Fixed test window = 2022-05 .. 2024-01 (20 months of bear+chop).
  - Tier pipeline identical to backtest_v2 (same vol gate, label, HC, Turtle).

This gives the FIRST HONEST look at how the current 4h model behaves outside
bull market.
"""
import json
import math
import pathlib
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = pathlib.Path("/root/elibri-backend/ml-training")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_engine import FEATURE_NAMES, build_features, _atr, make_target_triple_barrier
from data_fetcher import fetch_or_cache
from risk_tiers_config import TIERS
from backtest_v2 import (
    align_1d_direction, label_signal, apply_gates,
    sim_trade, aggregate, _annual_factor,
    HORIZON_EXIT, FALLBACK_HC, load_best_thresholds,
)
from train import TF_CONFIG, train_ensemble, ensemble_predict


SYMBOL = "BTCUSDT"
REGIME_WINDOWS = {
    "BEAR 2022":  ("2022-05-01", "2022-12-31"),  # LUNA to FTX blow-up
    "CHOP 2023":  ("2023-01-01", "2023-12-31"),  # 2023 full year
    "BULL 2024":  ("2024-01-01", "2024-06-30"),  # bull 2024 H1 peak 73k
}


def train_model_on_window(symbol: str, interval: str, train_end: str):
    """Train 4h model using all data BEFORE train_end. Simulates 'production
    knowledge state as of train_end'."""
    df = fetch_or_cache(symbol, interval, years=8.0)
    cfg = TF_CONFIG[interval]
    horizon = cfg["horizon"]
    feat = build_features(df)
    close_arr = df["close"].to_numpy()
    high_arr = df["high"].to_numpy()
    low_arr = df["low"].to_numpy()
    atr_arr = _atr(high_arr, low_arr, close_arr, period=14)
    target = make_target_triple_barrier(
        high_arr, low_arr, close_arr, atr_arr,
        horizon=horizon,
        upper_mult=cfg["tb_upper"],
        lower_mult=cfg["tb_lower"],
    )
    mask = target >= 0
    feat_m = feat.loc[mask].reset_index(drop=True)
    target = target[mask]
    df_m = df[mask].reset_index(drop=True)

    train_end_ts = pd.Timestamp(train_end, tz="UTC")
    train_mask = pd.to_datetime(feat_m["open_time"], utc=True) < train_end_ts
    train_idx = int(train_mask.sum())
    if train_idx < 200:
        raise RuntimeError(f"too few train rows before {train_end}: {train_idx}")

    embargo = max(horizon, 4)
    actual_train_end = max(0, train_idx - embargo)
    X_train = feat_m[FEATURE_NAMES].iloc[:actual_train_end].to_numpy()
    y_train = target[:actual_train_end]
    xgb, lgbm, rf, meta = train_ensemble(X_train, y_train, quick=False)
    print(f"  trained 4h on {actual_train_end} rows before {train_end}, embargo={embargo}")
    return xgb, lgbm, rf, meta, feat_m, df_m


def slice_window(feat, df, start: str, end: str):
    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end = pd.Timestamp(end, tz="UTC")
    times = pd.to_datetime(feat["open_time"], utc=True)
    mask = (times >= ts_start) & (times < ts_end)
    idx = np.where(mask.to_numpy())[0]
    if len(idx) == 0:
        raise RuntimeError(f"no data in [{start}, {end})")
    return idx


def build_daily_predictions(symbol: str, train_end: str):
    """Train 1d model up to train_end, return proba + dates for MTF filter."""
    df = fetch_or_cache(symbol, "1d", years=8.0)
    cfg = TF_CONFIG["1d"]
    feat = build_features(df)
    close_arr = df["close"].to_numpy()
    high_arr = df["high"].to_numpy()
    low_arr = df["low"].to_numpy()
    atr_arr = _atr(high_arr, low_arr, close_arr, period=14)
    target = make_target_triple_barrier(
        high_arr, low_arr, close_arr, atr_arr,
        horizon=cfg["horizon"],
        upper_mult=cfg["tb_upper"],
        lower_mult=cfg["tb_lower"],
    )
    mask = target >= 0
    feat_m = feat.loc[mask].reset_index(drop=True)
    target = target[mask]

    ts = pd.Timestamp(train_end, tz="UTC")
    train_mask = pd.to_datetime(feat_m["open_time"], utc=True) < ts
    train_idx = int(train_mask.sum())
    embargo = max(cfg["horizon"], 4)
    actual_train_end = max(0, train_idx - embargo)
    X_train = feat_m[FEATURE_NAMES].iloc[:actual_train_end].to_numpy()
    y_train = target[:actual_train_end]
    xgb, lgbm, rf, meta = train_ensemble(X_train, y_train, quick=False)
    # Predict on ENTIRE post-train range for alignment. Needs to cover the 4h
    # test windows we'll slice.
    X_all = feat_m[FEATURE_NAMES].iloc[actual_train_end:].to_numpy()
    proba_all, _ = ensemble_predict(xgb, lgbm, rf, meta, X_all)
    dates_all = pd.to_datetime(feat_m["open_time"].iloc[actual_train_end:].values)
    print(f"  1d MTF model: trained on {actual_train_end} rows, "
          f"predicting {len(dates_all)} OOS days")
    return proba_all, dates_all


def run_regime(tier_name, cfg, feat, df_4h, proba_4h, hc, idx_slice,
               daily_proba, daily_dates, fee=0.001, slippage=0.0003):
    """Run one tier over a regime window. Returns trades list + rejections."""
    closes = feat["close"].iloc[idx_slice].to_numpy().astype(float)
    highs = df_4h["high"].iloc[idx_slice].to_numpy().astype(float)
    lows = df_4h["low"].iloc[idx_slice].to_numpy().astype(float)
    atr = _atr(df_4h["high"].to_numpy(),
               df_4h["low"].to_numpy(),
               df_4h["close"].to_numpy(), 14)[idx_slice].astype(float)
    dates = pd.to_datetime(feat["open_time"].iloc[idx_slice].values)
    proba = np.asarray(proba_4h[idx_slice], dtype=float)
    feats_test = feat[["rsi_14", "bb_position", "adx_14", "atr_norm_14"]].iloc[idx_slice].reset_index(drop=True)

    dir_1d = align_1d_direction(dates, daily_dates, daily_proba)

    trades = []
    rejected = defaultdict(int)
    trades_today = defaultdict(int)
    equity = 10000.0
    horizon_bars = HORIZON_EXIT["4h"]
    n = len(proba)
    i = 0
    while i < n:
        p = float(proba[i])
        if p > hc:
            direction = 1
        elif p < (1.0 - hc):
            direction = -1
        else:
            rejected["hc_threshold"] += 1
            i += 1
            continue

        row = feats_test.iloc[i]
        label = label_signal(
            rsi=float(row["rsi_14"]),
            bb_pos=float(row["bb_position"]),
            adx_scaled=float(row["adx_14"]),
            signal_dir=direction,
            dir_1d=int(dir_1d[i]),
            interval="4h",
        )
        day_key = pd.Timestamp(dates[i]).date()
        passed, reason = apply_gates(
            proba=p,
            features_row=row,
            tier_cfg=cfg,
            hc_threshold=hc,
            interval="4h",
            label=label,
            trades_today=trades_today[day_key],
        )
        if not passed:
            rejected[reason] += 1
            i += 1
            continue

        trade = sim_trade(
            i=i, direction=direction, entry_ref=float(closes[i]),
            atr=float(atr[i]), tier_cfg=cfg, equity=equity,
            fee=fee, slippage=slippage,
            horizon_bars=horizon_bars,
            highs=highs, lows=lows, closes=closes, dates=dates,
        )
        trade["label"] = label
        trade["probability"] = p
        trades.append(trade)
        equity = trade["equity_after"]
        trades_today[day_key] += 1
        i = trade["exit_idx"] + 1

    return trades, rejected


def main():
    train_end = "2022-05-01"  # train only on 2017-08 .. 2022-04, test after
    print(f"=== B.2 d3b — regime split backtest ===")
    print(f"Train end: {train_end} (all data before = training)")
    print(f"Test windows:")
    for reg, (s, e) in REGIME_WINDOWS.items():
        print(f"  {reg}: {s} .. {e}")
    print()

    # Train 4h model once.
    print("Training 4h model (TF_CONFIG horizon=12 tb 3.0/2.0)...")
    xgb, lgbm, rf, meta, feat, df_4h = train_model_on_window(SYMBOL, "4h", train_end)
    # Predict on all 4h data (train + test) for slicing.
    X_all = feat[FEATURE_NAMES].to_numpy()
    proba_all, _ = ensemble_predict(xgb, lgbm, rf, meta, X_all)
    print(f"4h proba range: [{proba_all.min():.3f}, {proba_all.max():.3f}] mean={proba_all.mean():.3f}")

    # Train 1d MTF filter.
    print("Training 1d MTF filter...")
    daily_proba, daily_dates = build_daily_predictions(SYMBOL, train_end)

    # HC threshold from Phase 2 sweep.
    thresholds = load_best_thresholds()
    hc = thresholds.get(f"{SYMBOL}_4h", FALLBACK_HC["4h"])
    print(f"Using HC threshold: {hc}")

    # --- Run each tier × each regime ---
    print()
    print(f"{'tier':<14} {'regime':<12} {'n':<4} {'long/short':<12} "
          f"{'WR':<7} {'ret%':<8} {'Sharpe':<8}")
    print("-" * 80)

    summary = {}
    for tier_name in ("conservative", "balanced", "aggressive"):
        cfg = TIERS[tier_name]
        for reg, (s, e) in REGIME_WINDOWS.items():
            try:
                idx = slice_window(feat, df_4h, s, e)
            except RuntimeError as exc:
                print(f"  skip {tier_name}/{reg}: {exc}")
                continue
            trades, rej = run_regime(
                tier_name, cfg, feat, df_4h, proba_all, hc, idx,
                daily_proba, daily_dates,
            )
            longs = sum(1 for t in trades if t["direction"] == 1)
            shorts = sum(1 for t in trades if t["direction"] == -1)
            wins = sum(1 for t in trades if t["pnl_dollars"] > 0)
            wr = wins / max(1, len(trades))
            ret_pct = (sum(t["pnl_dollars"] for t in trades) / 10000.0) * 100
            returns = np.array([t["pnl_pct"] for t in trades], dtype=float)
            avg_bars = float(np.mean([t["bars_held"] for t in trades])) if trades else 1.0
            if len(returns) >= 2 and returns.std() > 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(_annual_factor("4h", avg_bars))
            else:
                sharpe = 0.0
            print(f"{tier_name:<14} {reg:<12} {len(trades):<4} "
                  f"{longs}/{shorts:<10} {wr*100:5.1f}%  "
                  f"{ret_pct:+6.2f}% {sharpe:+6.2f}")
            summary[(tier_name, reg)] = {
                "n": len(trades), "longs": longs, "shorts": shorts,
                "wr": wr, "ret": ret_pct, "sharpe": sharpe,
                "rejected": dict(rej),
            }

    # Save detailed summary.
    out = ROOT / "logs" / "b2_d3b_regime_summary.json"
    out.write_text(json.dumps({
        str(k): v for k, v in summary.items()
    }, indent=2, default=str))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
