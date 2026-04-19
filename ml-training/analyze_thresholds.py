"""
analyze_thresholds.py — post-training sweep of HC thresholds per model.

Loads every trained model, re-runs it on a held-out tail of the data,
and reports HC precision at thresholds [0.55, 0.60, 0.65, 0.70, 0.80]
plus the quantile-based top-10%.

Picks the best (threshold, precision) trade-off and saves to
logs/best_thresholds.json — consumed by the Go API to tune HC behaviour
without retraining.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engine import FEATURE_NAMES, build_features, make_target
from data_fetcher import fetch_or_cache
from train import HORIZON_MAP, train_ensemble, ensemble_predict, compute_hc_table


ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"


def analyze(symbol: str, interval: str, btc_close: np.ndarray | None) -> dict:
    df = fetch_or_cache(symbol, interval, years=8.0)
    if len(df) < 500:
        return {"symbol": symbol, "interval": interval, "error": "insufficient data"}

    horizon = HORIZON_MAP[interval]
    feat = build_features(df, btc_close=btc_close)
    target = make_target(df["close"].to_numpy(), horizon=horizon)
    mask = target >= 0
    feat = feat.loc[mask].reset_index(drop=True)
    target = target[mask]

    # Holdout = last 25% of data.
    split = int(len(feat) * 0.75)
    X_train, y_train = feat[FEATURE_NAMES].iloc[:split].to_numpy(), target[:split]
    X_test, y_test = feat[FEATURE_NAMES].iloc[split:].to_numpy(), target[split:]
    xgb, lgbm, rf, meta = train_ensemble(X_train, y_train, quick=False)
    proba, _ = ensemble_predict(xgb, lgbm, rf, meta, X_test)

    hc_table = compute_hc_table(y_test, proba)

    # Pick best: highest precision with >= 30 signals (meaningful sample).
    best = None
    for key, stats in hc_table.items():
        if stats["n_signals"] >= 30:
            if best is None or stats["precision"] > best["precision"]:
                best = {"key": key, **stats}
    if best is None:
        # Fall back to top-10% if nothing else had enough signals.
        best = {"key": "top_10pct", **hc_table.get("top_10pct", {"precision": 0, "n_signals": 0, "fraction": 0})}

    return {
        "symbol": symbol, "interval": interval, "horizon": horizon,
        "n_test": int(len(y_test)),
        "proba_min": float(proba.min()),
        "proba_max": float(proba.max()),
        "proba_mean": float(proba.mean()),
        "proba_std": float(proba.std()),
        "hc_table": hc_table,
        "best": best,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"])
    ap.add_argument("--intervals", nargs="*", default=["4h", "1d"])
    args = ap.parse_args()

    btc_by_iv = {iv: fetch_or_cache("BTCUSDT", iv, years=8.0)["close"].to_numpy() for iv in args.intervals}

    results = []
    for sym in args.symbols:
        for iv in args.intervals:
            btc_close = btc_by_iv.get(iv) if sym != "BTCUSDT" else None
            print(f"  analyzing {sym} {iv}…")
            r = analyze(sym, iv, btc_close)
            results.append(r)
            if "error" not in r:
                b = r["best"]
                print(f"    proba range [{r['proba_min']:.3f}, {r['proba_max']:.3f}] std={r['proba_std']:.3f}")
                print(f"    BEST: {b['key']} precision={b['precision']:.1%} on {b['n_signals']} signals ({b['fraction']:.1%} of bars)")

    # Save.
    out = {
        "generated_at": str(pd.Timestamp.now()),
        "results": results,
    }
    LOGS_DIR.mkdir(exist_ok=True)
    with open(LOGS_DIR / "best_thresholds.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    # Summary table.
    print("\n═══ Threshold sweep summary ═══")
    for r in results:
        if "error" in r:
            continue
        b = r["best"]
        print(f"  {r['symbol']:<10} {r['interval']:<4} σ={r['proba_std']:.3f}  "
              f"best: {b['key']:<10} precision={b['precision']:>5.1%} "
              f"({b['n_signals']} trades, {b['fraction']:.1%} of bars)")


if __name__ == "__main__":
    main()
