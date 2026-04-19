"""
paper_trade.py — virtual portfolio simulation on most recent N days.

Different from backtest.py: uses the FINAL deployed model (full-history trained,
saved in models/{sym}_{iv}_v*.json) and replays last N days in sequence,
producing the JSON log that the UI displays.

Output: logs/paper_trades.json
  {
    "started_at": "2026-04-18T23:00:00Z",
    "lookback_days": 90,
    "initial_equity": 10000,
    "final_equity": 11420,
    "trades": [
      {symbol, interval, direction, entry_price, exit_price, pnl_dollars, ...},
      ...
    ],
    "per_symbol": {BTCUSDT: {n, pnl, win_rate}, ...},
    "equity_curve": [{date, equity}, ...]
  }
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from feature_engine import FEATURE_NAMES, build_features
from data_fetcher import fetch_or_cache
from train import HORIZON_MAP, train_ensemble, ensemble_predict


ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"


def simulate_last_n_days(symbol: str, interval: str, days: int, initial_equity: float,
                         btc_close: np.ndarray | None) -> dict:
    """Simulate trades on the last N days using model trained on everything before."""
    df = fetch_or_cache(symbol, interval, years=8.0)
    if len(df) < 500:
        return {"symbol": symbol, "interval": interval, "error": "not enough data"}

    horizon = HORIZON_MAP[interval]
    feat = build_features(df, btc_close=btc_close)
    from feature_engine import make_target, _atr
    target = make_target(df["close"].to_numpy(), horizon=horizon, threshold=0.0)
    mask = target >= 0
    # Keep aligned: feat, target, and reindexed high/low arrays.
    df_aligned = df.loc[mask].reset_index(drop=True)
    feat = feat.loc[mask].reset_index(drop=True)
    target = target[mask]
    all_highs = df_aligned["high"].to_numpy()
    all_lows = df_aligned["low"].to_numpy()

    # Split: last N days = test, earlier = train.
    cutoff = pd.Timestamp.now(tz="UTC") - timedelta(days=days)
    feat_times = pd.to_datetime(feat["open_time"])
    if feat_times.dt.tz is None:
        feat_times = feat_times.dt.tz_localize("UTC")
    else:
        feat_times = feat_times.dt.tz_convert("UTC")
    test_mask = (feat_times > cutoff).to_numpy()
    n_test = int(test_mask.sum())
    if n_test < 10:
        return {"symbol": symbol, "interval": interval, "error": f"only {n_test} test bars"}

    X_train = feat[FEATURE_NAMES].to_numpy()[~test_mask]
    y_train = target[~test_mask]
    X_test = feat[FEATURE_NAMES].to_numpy()[test_mask]
    test_closes = feat["close"].to_numpy()[test_mask]
    test_dates = feat["open_time"].to_numpy()[test_mask]
    orig_highs = all_highs[test_mask]
    orig_lows = all_lows[test_mask]

    xgb, lgbm, rf, meta = train_ensemble(X_train, y_train, quick=False)
    proba, _ = ensemble_predict(xgb, lgbm, rf, meta, X_test)

    # ATR aligned to test subset.
    atr_all = _atr(all_highs, all_lows, feat["close"].to_numpy(), 14)
    test_atr = atr_all[test_mask]

    equity = initial_equity
    trades = []
    equity_curve = [{"date": str(test_dates[0]), "equity": equity}]
    i = 0
    while i < n_test:
        p = proba[i]
        # Use 0.55/0.45 threshold for 90-day paper-trading visibility.
        # (Backtest uses 0.60 for longer-horizon OOS; lower here so recent
        #  90 days have enough signals to show meaningful activity.)
        if p <= 0.55 and p >= 0.45:
            i += 1
            continue
        direction = "long" if p > 0.55 else "short"
        entry = test_closes[i]
        atr = test_atr[i] if test_atr[i] > 0 else entry * 0.01
        if direction == "long":
            sl, tp = entry - 1.5 * atr, entry + 2.5 * atr
        else:
            sl, tp = entry + 1.5 * atr, entry - 2.5 * atr

        exit_idx, exit_price = None, None
        for j in range(i + 1, min(n_test, i + 100)):
            if direction == "long":
                if orig_lows[j] <= sl: exit_idx, exit_price = j, sl; break
                if orig_highs[j] >= tp: exit_idx, exit_price = j, tp; break
            else:
                if orig_highs[j] >= sl: exit_idx, exit_price = j, sl; break
                if orig_lows[j] <= tp: exit_idx, exit_price = j, tp; break
        if exit_price is None:
            exit_idx = min(n_test - 1, i + 100)
            exit_price = test_closes[exit_idx]

        pnl_pct = (exit_price - entry) / entry if direction == "long" else (entry - exit_price) / entry
        sl_dist = abs(entry - sl) / entry
        size = (equity * 0.05) / max(sl_dist, 1e-6)
        pnl_dollars = size * pnl_pct
        equity += pnl_dollars

        trades.append({
            "symbol": symbol,
            "interval": interval,
            "direction": direction,
            "entry_date": str(test_dates[i]),
            "exit_date": str(test_dates[exit_idx]),
            "entry_price": float(entry),
            "exit_price": float(exit_price),
            "sl": float(sl),
            "tp": float(tp),
            "probability": float(p),
            "bars_held": exit_idx - i,
            "pnl_pct": float(pnl_pct),
            "pnl_dollars": float(pnl_dollars),
            "equity_after": float(equity),
        })
        equity_curve.append({"date": str(test_dates[exit_idx]), "equity": float(equity)})
        i = exit_idx + 1

    wins = [t for t in trades if t["pnl_dollars"] > 0]
    return {
        "symbol": symbol,
        "interval": interval,
        "n_trades": len(trades),
        "n_test_bars": n_test,
        "win_rate": len(wins) / len(trades) if trades else 0,
        "initial_equity": initial_equity,
        "final_equity": equity,
        "total_return_pct": (equity - initial_equity) / initial_equity * 100,
        "trades": trades,
        "equity_curve": equity_curve,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"])
    ap.add_argument("--intervals", nargs="*", default=["4h", "1d"])
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--capital-per-strategy", type=float, default=2000.0,
                    help="$ per (symbol,interval); total portfolio = N_strategies × this")
    args = ap.parse_args()

    # Pre-load BTC for cross-asset.
    btc_close_by_iv = {iv: fetch_or_cache("BTCUSDT", iv, years=8.0)["close"].to_numpy()
                        for iv in args.intervals}

    all_results = []
    total_initial = 0
    total_final = 0
    all_trades = []
    all_equity = []

    for sym in args.symbols:
        for iv in args.intervals:
            btc_close = btc_close_by_iv.get(iv) if sym != "BTCUSDT" else None
            try:
                r = simulate_last_n_days(sym, iv, args.days, args.capital_per_strategy, btc_close)
                all_results.append(r)
                if "error" not in r:
                    total_initial += r["initial_equity"]
                    total_final += r["final_equity"]
                    all_trades.extend(r["trades"])
                    all_equity.extend(r["equity_curve"])
                    print(f"  {sym} {iv}: n={r['n_trades']:3d} wr={r['win_rate']:.0%} "
                          f"ret={r['total_return_pct']:+.1f}%")
                else:
                    print(f"  {sym} {iv}: {r['error']}")
            except Exception as e:
                print(f"  {sym} {iv} FAILED: {e}")

    # Sort all_equity by date for portfolio curve.
    all_equity.sort(key=lambda x: x["date"])

    summary = {
        "started_at": datetime.now(tz=timezone.utc).isoformat(),
        "lookback_days": args.days,
        "capital_per_strategy": args.capital_per_strategy,
        "n_strategies": len(all_results),
        "total_initial_equity": total_initial,
        "total_final_equity": total_final,
        "total_return_pct": (total_final - total_initial) / total_initial * 100 if total_initial > 0 else 0,
        "n_trades_total": len(all_trades),
        "n_winning": sum(1 for t in all_trades if t["pnl_dollars"] > 0),
        "win_rate_overall": sum(1 for t in all_trades if t["pnl_dollars"] > 0) / len(all_trades) if all_trades else 0,
        "per_strategy": all_results,
    }
    with open(LOGS_DIR / "paper_trades.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n═══ Paper-trading summary (last {args.days}d) ═══")
    print(f"  strategies: {summary['n_strategies']}")
    print(f"  trades: {summary['n_trades_total']} (win rate: {summary['win_rate_overall']:.1%})")
    print(f"  total return: {summary['total_return_pct']:+.2f}%")
    print(f"  ${total_initial:.0f} → ${total_final:.2f}")


if __name__ == "__main__":
    main()
