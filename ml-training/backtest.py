"""
backtest.py — out-of-sample walk-forward backtest of the trained ensemble.

Strategy:
  - For each bar: predict next-horizon direction using model trained on data
    strictly before that bar (respects walk-forward from train.py).
  - Enter trade IF probability > HC_THRESHOLD (default 0.80) or < (1 - HC_THRESHOLD).
  - Position: long if prob > 0.80, short if prob < 0.20.
  - SL: entry ± 1.5 × ATR(14).  TP: entry ± 2.5 × ATR(14).
  - Exit: first to hit SL/TP, or close at end of horizon.
  - Position sizing: 5% of current equity at risk per trade.
  - Start capital: $10,000.

Outputs:
  logs/backtest_{sym}_{iv}.json    — equity curve + stats
  logs/backtest_summary.json       — aggregated across all pairs
  logs/backtest_{sym}_{iv}.png     — equity curve plot (if matplotlib available)

Regime detection:
  bull  = BTCUSDT close > 50d EMA and 50d EMA rising
  bear  = BTCUSDT close < 50d EMA and 50d EMA falling
  range = otherwise
Metrics are broken down per regime.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from feature_engine import FEATURE_NAMES, build_features, make_target
from data_fetcher import fetch_or_cache
from train import HORIZON_MAP, train_ensemble, ensemble_predict, walk_forward_split


ROOT = Path(__file__).parent
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)


@dataclass
class Trade:
    entry_date: str
    exit_date: str
    direction: str          # long | short
    entry_price: float
    exit_price: float
    sl: float
    tp: float
    bars_held: int
    pnl_pct: float          # unleveraged % return
    pnl_dollars: float
    equity_after: float
    regime: str             # bull | bear | range
    probability: float


@dataclass
class BacktestResult:
    symbol: str
    interval: str
    horizon: int
    start_date: str
    end_date: str
    initial_equity: float
    final_equity: float
    total_return_pct: float
    n_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe: float
    sortino: float
    avg_bars_held: float
    # Per-regime breakdown
    regime_stats: dict
    trades: List[Trade]
    equity_curve: List[dict]  # [{date, equity, trade_idx}]


def classify_regime(btc_df: pd.DataFrame) -> pd.Series:
    """Label each bar in BTC df as bull/bear/range based on 50d EMA trend."""
    close = btc_df["close"]
    ema = close.ewm(span=50, adjust=False).mean()
    ema_slope = ema.diff(10)  # 10-bar change of EMA

    bull = (close > ema) & (ema_slope > 0)
    bear = (close < ema) & (ema_slope < 0)

    regime = pd.Series("range", index=btc_df.index)
    regime[bull] = "bull"
    regime[bear] = "bear"
    return regime


def simulate_trades(
    symbol: str,
    interval: str,
    test_proba: np.ndarray,
    test_df: pd.DataFrame,
    regimes: pd.Series,
    # Default 0.60 matches observed meta-learner spread (probs concentrate
    # in [0.40, 0.67]); analyze_thresholds.py may pick a better per-model
    # value and downstream threshold files override.
    hc_threshold: float = 0.60,
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 2.5,
    risk_per_trade: float = 0.05,
    initial_equity: float = 10000.0,
) -> BacktestResult:
    """Walk through test bars, open trades on HC signals, track PnL."""
    closes = test_df["close"].to_numpy()
    highs = test_df["high"].to_numpy()
    lows = test_df["low"].to_numpy()
    dates = test_df["open_time"].values

    # Compute ATR on test window.
    from feature_engine import _atr
    atr_series = _atr(highs, lows, closes, 14)

    equity = initial_equity
    trades: List[Trade] = []
    equity_curve = [{"date": str(dates[0]), "equity": equity, "trade_idx": -1}]

    # Iterate forward. When open position, skip signals (no pyramiding).
    in_trade = False
    i = 0
    n = len(closes)

    while i < n:
        if not in_trade:
            prob = test_proba[i]
            if prob > hc_threshold:
                direction = "long"
            elif prob < 1 - hc_threshold:
                direction = "short"
            else:
                i += 1
                continue

            entry_price = closes[i]
            atr = atr_series[i] if atr_series[i] > 0 else entry_price * 0.01
            if direction == "long":
                sl = entry_price - sl_atr_mult * atr
                tp = entry_price + tp_atr_mult * atr
            else:
                sl = entry_price + sl_atr_mult * atr
                tp = entry_price - tp_atr_mult * atr

            # Walk forward until SL or TP hit, or end of data.
            exit_price = None
            exit_idx = None
            for j in range(i + 1, min(n, i + 100)):
                if direction == "long":
                    if lows[j] <= sl:
                        exit_price = sl
                        exit_idx = j
                        break
                    if highs[j] >= tp:
                        exit_price = tp
                        exit_idx = j
                        break
                else:  # short
                    if highs[j] >= sl:
                        exit_price = sl
                        exit_idx = j
                        break
                    if lows[j] <= tp:
                        exit_price = tp
                        exit_idx = j
                        break
            if exit_price is None:
                # Close at last available bar.
                exit_idx = min(n - 1, i + 100)
                exit_price = closes[exit_idx]

            # PnL.
            if direction == "long":
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price

            # Risk-based sizing: risk 5% equity on SL distance.
            sl_dist_pct = abs(entry_price - sl) / entry_price
            position_size = (equity * risk_per_trade) / max(sl_dist_pct, 1e-6)
            pnl_dollars = position_size * pnl_pct
            equity += pnl_dollars

            regime = str(regimes.iloc[min(i, len(regimes) - 1)]) if len(regimes) > 0 else "range"

            trades.append(Trade(
                entry_date=str(dates[i]),
                exit_date=str(dates[exit_idx]),
                direction=direction,
                entry_price=float(entry_price),
                exit_price=float(exit_price),
                sl=float(sl),
                tp=float(tp),
                bars_held=exit_idx - i,
                pnl_pct=float(pnl_pct),
                pnl_dollars=float(pnl_dollars),
                equity_after=float(equity),
                regime=regime,
                probability=float(prob),
            ))
            equity_curve.append({
                "date": str(dates[exit_idx]),
                "equity": float(equity),
                "trade_idx": len(trades) - 1,
            })
            i = exit_idx + 1
            in_trade = False
        else:
            i += 1

    # Compute stats.
    n_trades = len(trades)
    if n_trades == 0:
        return BacktestResult(
            symbol=symbol, interval=interval, horizon=HORIZON_MAP[interval],
            start_date=str(dates[0]), end_date=str(dates[-1]),
            initial_equity=initial_equity, final_equity=equity,
            total_return_pct=0.0, n_trades=0,
            win_rate=0.0, avg_win_pct=0.0, avg_loss_pct=0.0,
            profit_factor=0.0, max_drawdown_pct=0.0,
            sharpe=0.0, sortino=0.0, avg_bars_held=0.0,
            regime_stats={}, trades=[], equity_curve=equity_curve,
        )

    wins = [t for t in trades if t.pnl_dollars > 0]
    losses = [t for t in trades if t.pnl_dollars <= 0]
    win_rate = len(wins) / n_trades
    avg_win = float(np.mean([t.pnl_pct for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([t.pnl_pct for t in losses])) if losses else 0.0
    total_wins = sum(t.pnl_dollars for t in wins)
    total_losses = abs(sum(t.pnl_dollars for t in losses))
    profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

    # Drawdown from equity curve.
    eq_vals = [p["equity"] for p in equity_curve]
    peak = eq_vals[0]
    max_dd = 0.0
    for v in eq_vals:
        peak = max(peak, v)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)

    # Sharpe: per-trade PnL%.
    returns = np.array([t.pnl_pct for t in trades])
    sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0
    neg_returns = returns[returns < 0]
    sortino = float(returns.mean() / neg_returns.std() * np.sqrt(252)) if len(neg_returns) > 1 and neg_returns.std() > 0 else 0.0

    # Per-regime stats.
    regime_stats = {}
    for reg in ("bull", "bear", "range"):
        reg_trades = [t for t in trades if t.regime == reg]
        if not reg_trades:
            regime_stats[reg] = {"n": 0, "win_rate": 0, "total_pnl_pct": 0}
            continue
        reg_wins = [t for t in reg_trades if t.pnl_dollars > 0]
        regime_stats[reg] = {
            "n": len(reg_trades),
            "win_rate": len(reg_wins) / len(reg_trades),
            "total_pnl_pct": float(sum(t.pnl_pct for t in reg_trades)),
            "total_pnl_dollars": float(sum(t.pnl_dollars for t in reg_trades)),
            "avg_pnl_pct": float(np.mean([t.pnl_pct for t in reg_trades])),
        }

    return BacktestResult(
        symbol=symbol, interval=interval, horizon=HORIZON_MAP[interval],
        start_date=str(dates[0]), end_date=str(dates[-1]),
        initial_equity=initial_equity, final_equity=equity,
        total_return_pct=(equity - initial_equity) / initial_equity * 100,
        n_trades=n_trades,
        win_rate=win_rate,
        avg_win_pct=avg_win * 100,
        avg_loss_pct=avg_loss * 100,
        profit_factor=profit_factor,
        max_drawdown_pct=max_dd * 100,
        sharpe=sharpe,
        sortino=sortino,
        avg_bars_held=float(np.mean([t.bars_held for t in trades])),
        regime_stats=regime_stats,
        trades=trades,
        equity_curve=equity_curve,
    )


def backtest_one(symbol: str, interval: str, years: float, btc_close: np.ndarray | None,
                 btc_df: pd.DataFrame | None) -> BacktestResult | None:
    """Run OOS backtest: train on first 70% of data, test on last 30%."""
    t0 = time.time()
    print(f"\n━━━ BACKTEST {symbol} {interval} ━━━")

    df = fetch_or_cache(symbol, interval, years=years)
    if len(df) < 500:
        print(f"  skip: only {len(df)} candles")
        return None

    horizon = HORIZON_MAP[interval]
    feat = build_features(df, btc_close=btc_close)
    target = make_target(df["close"].to_numpy(), horizon=horizon, threshold=0.0)
    mask = target >= 0
    feat = feat.loc[mask].reset_index(drop=True)
    target = target[mask]

    # Align btc_df to feature frame for regime labelling.
    if btc_df is not None:
        btc_aligned = btc_df.set_index("open_time").reindex(feat["open_time"].values).reset_index(drop=True)
        # Forward-fill any gaps.
        btc_aligned = btc_aligned.ffill().bfill()
        regimes = classify_regime(btc_aligned)
    else:
        regimes = classify_regime(df.loc[mask].reset_index(drop=True))

    split = int(len(feat) * 0.7)
    X_train = feat[FEATURE_NAMES].iloc[:split].to_numpy()
    y_train = target[:split]
    X_test = feat[FEATURE_NAMES].iloc[split:].to_numpy()
    # Align high/low from original df using the same mask (target != sentinel).
    # `mask` is a numpy bool array — use directly with DataFrame .loc.
    df_masked = df.loc[mask].reset_index(drop=True) if hasattr(mask, 'index') else df[mask].reset_index(drop=True)
    test_df = pd.DataFrame({
        "close": feat["close"].iloc[split:].to_numpy(),
        "high":  df_masked["high"].iloc[split:].to_numpy(),
        "low":   df_masked["low"].iloc[split:].to_numpy(),
        "open_time": feat["open_time"].iloc[split:].values,
    })

    train_start = pd.Timestamp(feat['open_time'].iloc[split])
    train_end = pd.Timestamp(feat['open_time'].iloc[-1])
    print(f"  train={split} test={len(feat) - split} ({str(train_start)[:10]} → {str(train_end)[:10]})")
    print("  training…")
    xgb, lgbm, rf, meta = train_ensemble(X_train, y_train, quick=False)
    y_proba, _ = ensemble_predict(xgb, lgbm, rf, meta, X_test)
    print(f"  predictions: min={y_proba.min():.3f} max={y_proba.max():.3f}  "
          f"hc_count={((y_proba > 0.8) | (y_proba < 0.2)).sum()}")

    regimes_test = regimes.iloc[split:].reset_index(drop=True) if len(regimes) >= len(feat) else pd.Series(["range"] * len(X_test))
    result = simulate_trades(symbol, interval, y_proba, test_df, regimes_test)

    print(f"  n_trades={result.n_trades} win_rate={result.win_rate:.1%} "
          f"total_return={result.total_return_pct:+.1f}% "
          f"max_dd={result.max_drawdown_pct:.1f}% sharpe={result.sharpe:+.2f}")
    for reg, rs in result.regime_stats.items():
        if rs.get("n", 0) > 0:
            print(f"    {reg:5}: n={rs['n']:3d} win={rs['win_rate']:.0%} "
                  f"total_pnl={rs['total_pnl_dollars']:+.2f}$")

    # Save JSON.
    out = asdict(result)
    out["trades"] = [asdict(t) for t in result.trades]
    path = LOGS_DIR / f"backtest_{symbol}_{interval}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    # Save equity-curve PNG.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        dates = [pd.Timestamp(p["date"]) for p in result.equity_curve]
        eq = [p["equity"] for p in result.equity_curve]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dates, eq, lw=1.2, color="#6366f1")
        ax.axhline(result.initial_equity, ls=":", color="#666")
        ax.set_title(f"{symbol} {interval} — equity curve ({result.n_trades} trades, ret={result.total_return_pct:+.1f}%)")
        ax.set_ylabel("Equity $")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(LOGS_DIR / f"backtest_{symbol}_{interval}.png", dpi=100)
        plt.close(fig)
    except ImportError:
        pass

    print(f"  ⏱  {time.time() - t0:.1f}s")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"])
    ap.add_argument("--intervals", nargs="*", default=["4h", "1d"])
    ap.add_argument("--years", type=float, default=8.0)
    args = ap.parse_args()

    # Preload BTC for cross-asset + regime.
    btc_dfs = {iv: fetch_or_cache("BTCUSDT", iv, years=args.years) for iv in args.intervals}

    results = []
    for sym in args.symbols:
        for iv in args.intervals:
            btc_df = btc_dfs.get(iv)
            btc_close = btc_df["close"].to_numpy() if (sym != "BTCUSDT" and btc_df is not None) else None
            try:
                r = backtest_one(sym, iv, args.years, btc_close, btc_df)
                if r:
                    results.append(r)
            except Exception as e:
                print(f"  {sym} {iv} FAILED: {e}")

    # Summary.
    summary = {
        "total_strategies": len(results),
        "avg_total_return_pct": float(np.mean([r.total_return_pct for r in results])) if results else 0,
        "avg_win_rate": float(np.mean([r.win_rate for r in results])) if results else 0,
        "avg_sharpe": float(np.mean([r.sharpe for r in results])) if results else 0,
        "best": max(((r.symbol, r.interval, r.total_return_pct) for r in results), key=lambda x: x[2], default=None),
        "worst": min(((r.symbol, r.interval, r.total_return_pct) for r in results), key=lambda x: x[2], default=None),
        "results": [
            {
                "symbol": r.symbol, "interval": r.interval,
                "n_trades": r.n_trades, "win_rate": r.win_rate,
                "total_return_pct": r.total_return_pct,
                "max_drawdown_pct": r.max_drawdown_pct,
                "sharpe": r.sharpe,
                "profit_factor": r.profit_factor,
            }
            for r in results
        ],
    }
    with open(LOGS_DIR / "backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n═══ Backtest summary ═══")
    print(f"  strategies: {summary['total_strategies']}")
    print(f"  avg return: {summary['avg_total_return_pct']:+.1f}%")
    print(f"  avg win rate: {summary['avg_win_rate']:.1%}")
    print(f"  avg sharpe: {summary['avg_sharpe']:+.2f}")
    if summary["best"]:
        print(f"  BEST:  {summary['best'][0]} {summary['best'][1]}: {summary['best'][2]:+.1f}%")
    if summary["worst"]:
        print(f"  WORST: {summary['worst'][0]} {summary['worst'][1]}: {summary['worst'][2]:+.1f}%")


if __name__ == "__main__":
    main()
