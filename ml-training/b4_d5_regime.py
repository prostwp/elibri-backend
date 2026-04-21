"""
b4_d5_regime.py — B.4 d5: 15m regime split. Clone of b3_d5 but for 15m.
"""
import json
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
    sim_trade, _annual_factor,
    HORIZON_EXIT, FALLBACK_HC, load_best_thresholds,
)
from train import TF_CONFIG, train_ensemble, ensemble_predict


SYMBOL = "BTCUSDT"
REGIME_WINDOWS = {
    "BEAR 2022": ("2022-05-01", "2022-12-31"),
    "CHOP 2023": ("2023-01-01", "2023-12-31"),
    "BULL 2024": ("2024-01-01", "2024-06-30"),
}


def train_model(symbol, interval, train_end):
    df = fetch_or_cache(symbol, interval, years=8.0)
    cfg = TF_CONFIG[interval]
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
    df_m = df[mask].reset_index(drop=True)

    ts = pd.Timestamp(train_end, tz="UTC")
    tm = pd.to_datetime(feat_m["open_time"], utc=True) < ts
    train_idx = int(tm.sum())
    embargo = max(cfg["horizon"], 4)
    end = max(0, train_idx - embargo)
    X_train = feat_m[FEATURE_NAMES].iloc[:end].to_numpy()
    y_train = target[:end]
    xgb, lgbm, rf, meta = train_ensemble(X_train, y_train, quick=False)
    print(f"  {interval}: trained on {end} rows (embargo {embargo})")
    return xgb, lgbm, rf, meta, feat_m, df_m


def slice_idx(feat, start, end):
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC")
    times = pd.to_datetime(feat["open_time"], utc=True)
    m = (times >= s) & (times < e)
    return np.where(m.to_numpy())[0]


def run_regime(tier_name, cfg, feat, df_tf, proba_all, hc, idx,
                daily_proba, daily_dates, tf, fee=0.001, slippage=0.0003):
    closes = feat["close"].iloc[idx].to_numpy().astype(float)
    highs = df_tf["high"].iloc[idx].to_numpy().astype(float)
    lows = df_tf["low"].iloc[idx].to_numpy().astype(float)
    atr_full = _atr(df_tf["high"].to_numpy(), df_tf["low"].to_numpy(),
                    df_tf["close"].to_numpy(), 14)
    atr = atr_full[idx].astype(float)
    dates = pd.to_datetime(feat["open_time"].iloc[idx].values)
    proba = np.asarray(proba_all[idx], dtype=float)
    feats_test = feat[["rsi_14", "bb_position", "adx_14", "atr_norm_14"]].iloc[idx].reset_index(drop=True)
    dir_1d = align_1d_direction(dates, daily_dates, daily_proba)

    trades = []
    rejected = defaultdict(int)
    trades_today = defaultdict(int)
    equity = 10000.0
    horizon_bars = HORIZON_EXIT[tf]
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
            rsi=float(row["rsi_14"]), bb_pos=float(row["bb_position"]),
            adx_scaled=float(row["adx_14"]), signal_dir=direction,
            dir_1d=int(dir_1d[i]), interval=tf,
        )
        day_key = pd.Timestamp(dates[i]).date()
        passed, reason = apply_gates(
            proba=p, features_row=row, tier_cfg=cfg,
            hc_threshold=hc, interval=tf, label=label,
            trades_today=trades_today[day_key],
        )
        if not passed:
            rejected[reason] += 1
            i += 1
            continue
        t = sim_trade(i=i, direction=direction, entry_ref=float(closes[i]),
                      atr=float(atr[i]), tier_cfg=cfg, equity=equity,
                      fee=fee, slippage=slippage,
                      horizon_bars=horizon_bars,
                      highs=highs, lows=lows, closes=closes, dates=dates)
        t["label"] = label
        trades.append(t)
        equity = t["equity_after"]
        trades_today[day_key] += 1
        i = t["exit_idx"] + 1
    return trades, rejected


def main():
    train_end = "2022-05-01"
    TF = "15m"
    print(f"=== B.4 d5 — {TF} regime split, train_end={train_end} ===")
    for reg, (s, e) in REGIME_WINDOWS.items():
        print(f"  {reg}: {s} .. {e}")
    print()

    print(f"Training {TF}...")
    xgb, lgbm, rf, meta, feat, df_tf = train_model(SYMBOL, TF, train_end)
    X_all = feat[FEATURE_NAMES].to_numpy()
    proba_all, _ = ensemble_predict(xgb, lgbm, rf, meta, X_all)
    print(f"{TF} proba: min={proba_all.min():.3f} max={proba_all.max():.3f} mean={proba_all.mean():.3f}")

    print("Training 1d MTF filter...")
    xgb_d, lgbm_d, rf_d, meta_d, feat_1d, _ = train_model(SYMBOL, "1d", train_end)
    X_1d = feat_1d[FEATURE_NAMES].to_numpy()
    proba_1d, _ = ensemble_predict(xgb_d, lgbm_d, rf_d, meta_d, X_1d)
    daily_dates = pd.to_datetime(feat_1d["open_time"].values)

    thresholds = load_best_thresholds()
    hc = thresholds.get(f"{SYMBOL}_{TF}", FALLBACK_HC[TF])
    print(f"HC: {hc}")

    print()
    print(f"{'tier':<14} {'regime':<12} {'n':<5} {'long/short':<12} "
          f"{'WR':<7} {'ret%':<8} {'Sharpe':<8}")
    print("-" * 80)

    summary = {}
    for tier_name in ("conservative", "balanced", "aggressive"):
        cfg = TIERS[tier_name]
        for reg, (s, e) in REGIME_WINDOWS.items():
            idx = slice_idx(feat, s, e)
            if len(idx) == 0:
                continue
            trades, rej = run_regime(tier_name, cfg, feat, df_tf, proba_all, hc, idx,
                                     proba_1d, daily_dates, TF)
            longs = sum(1 for t in trades if t["direction"] == 1)
            shorts = sum(1 for t in trades if t["direction"] == -1)
            wins = sum(1 for t in trades if t["pnl_dollars"] > 0)
            wr = wins / max(1, len(trades))
            ret = (sum(t["pnl_dollars"] for t in trades) / 10000.0) * 100
            returns = np.array([t["pnl_pct"] for t in trades], dtype=float)
            avg_bars = float(np.mean([t["bars_held"] for t in trades])) if trades else 1.0
            if len(returns) >= 2 and returns.std() > 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(_annual_factor(TF, avg_bars))
            else:
                sharpe = 0.0
            print(f"{tier_name:<14} {reg:<12} {len(trades):<5} "
                  f"{longs}/{shorts:<10} {wr*100:5.1f}%  "
                  f"{ret:+6.2f}% {sharpe:+6.2f}")
            summary[(tier_name, reg)] = {
                "n": len(trades), "longs": longs, "shorts": shorts,
                "wr": wr, "ret": ret, "sharpe": sharpe,
            }

    out = ROOT / "logs" / "b4_d5_regime_summary.json"
    out.write_text(json.dumps({str(k): v for k, v in summary.items()}, indent=2, default=str))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
