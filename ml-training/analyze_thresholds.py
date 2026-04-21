"""
analyze_thresholds.py — post-training sweep of HC thresholds per model.

Re-trains a quick ensemble on 75% of the data, evaluates on 25% holdout,
sweeps HC thresholds on a fine grid (0.55-0.90 step 0.025), and picks
the BEST (highest-threshold, quality-over-quantity) point that satisfies:

    precision >= MIN_PRECISION  AND
    MIN_SIG_PER_DAY <= signals_per_day <= MAX_SIG_PER_DAY[interval]

We take the MAXIMUM threshold inside the window (max quality while keeping
trade frequency realistic). Fallback 1: max threshold with precision>=0.55
AND sig/day>=0.3 (for 1d where 60% is unreachable). Fallback 2: max
precision with >=30 signals.

Saves to logs/best_thresholds.json — consumed by Go API to tune HC.

Supports tb_atr target mode for symmetric comparison with the
triple-barrier trained models.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engine import FEATURE_NAMES, build_features, make_target, make_target_triple_barrier, _atr
from data_fetcher import fetch_or_cache
from train import HORIZON_MAP, TF_CONFIG, train_ensemble, ensemble_predict


ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"

BARS_PER_DAY = {"5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}
MIN_PRECISION = 0.60
MIN_SIG_PER_DAY = 1.0
# Upper cap on trade frequency per interval — keeps signals realistic.
MAX_SIG_PER_DAY = {"5m": 10, "15m": 5, "1h": 3, "4h": 2, "1d": 1}

# Relaxed fallback thresholds (for 1d or very selective models).
FALLBACK_MIN_PRECISION = 0.55
FALLBACK_MIN_SIG_PER_DAY = 0.3

# Threshold grid: fine sweep from 0.55 to 0.90.
THRESHOLDS = [round(0.55 + 0.025 * i, 3) for i in range(15)]  # 0.55, 0.575, ..., 0.9


def sweep_thresholds(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Compute precision + n_signals at every threshold in THRESHOLDS."""
    out = {}
    for t in THRESHOLDS:
        mask = (proba > t) | (proba < 1 - t)
        n = int(mask.sum())
        if n == 0:
            out[f"thr_{t:.3f}"] = {"threshold": t, "precision": 0.0, "n_signals": 0, "fraction": 0.0}
            continue
        pred = (proba[mask] > 0.5).astype(int)
        correct = int((pred == y_true[mask]).sum())
        out[f"thr_{t:.3f}"] = {
            "threshold": t,
            "precision": correct / n,
            "n_signals": n,
            "fraction": n / len(y_true),
        }
    return out


def pick_best(hc_table: dict, interval: str) -> dict:
    """Highest threshold inside the [MIN_SIG_PER_DAY, MAX_SIG_PER_DAY] window
    satisfying precision>=0.60. Quality-over-quantity: we want the strictest
    gate that still produces a realistic trade frequency for the interval.
    """
    bpd = BARS_PER_DAY[interval]
    max_spd = MAX_SIG_PER_DAY.get(interval, 10)

    # Primary: max threshold with precision>=0.60 AND spd in [min, max].
    ok = []
    for _, stats in hc_table.items():
        spd = stats["fraction"] * bpd
        if (stats["precision"] >= MIN_PRECISION
                and MIN_SIG_PER_DAY <= spd <= max_spd):
            ok.append((stats["threshold"], stats, spd))
    if ok:
        ok.sort(key=lambda x: x[0], reverse=True)
        thr, stats, spd = ok[0]
        return {
            "key": f"thr_{thr:.3f}",
            "policy": "max_threshold_within_window",
            "sig_per_day": spd,
            **stats,
        }

    # Fallback 1: relaxed precision + relaxed min spd, still capped at max.
    # Used mainly for 1d models where 60% precision is often unreachable.
    relaxed = []
    for _, stats in hc_table.items():
        spd = stats["fraction"] * bpd
        if (stats["precision"] >= FALLBACK_MIN_PRECISION
                and FALLBACK_MIN_SIG_PER_DAY <= spd <= max_spd):
            relaxed.append((stats["threshold"], stats, spd))
    if relaxed:
        relaxed.sort(key=lambda x: x[0], reverse=True)
        thr, stats, spd = relaxed[0]
        return {
            "key": f"thr_{thr:.3f}",
            "policy": "fallback_relaxed_max_threshold",
            "sig_per_day": spd,
            **stats,
        }

    # Fallback 2: highest precision with >= 30 signals (original safety net).
    fb = None
    for _, stats in hc_table.items():
        if stats["n_signals"] >= 30 and stats["precision"] >= MIN_PRECISION:
            if fb is None or stats["precision"] > fb["precision"]:
                fb = stats
    if fb:
        return {
            "key": f"thr_{fb['threshold']:.3f}",
            "policy": "fallback_max_precision",
            "sig_per_day": fb["fraction"] * bpd,
            **fb,
        }
    return {"key": None, "policy": "no_valid_threshold", "sig_per_day": 0.0, "precision": 0.0, "n_signals": 0, "fraction": 0.0}


def analyze(symbol: str, interval: str, btc_close: np.ndarray | None,
            target_mode: str = "binary", tb_upper: float = 1.5, tb_lower: float = 1.0) -> dict:
    df = fetch_or_cache(symbol, interval, years=8.0)
    if len(df) < 500:
        return {"symbol": symbol, "interval": interval, "error": "insufficient data"}

    # Patch 2H: read horizon from TF_CONFIG (same dict train.py uses).
    # HORIZON_MAP is legacy (e.g. 1d: 10) while TF_CONFIG 1d: 5. Threshold
    # sweeps on a different horizon than the deployed model are meaningless.
    tf_cfg = TF_CONFIG.get(interval, {"horizon": HORIZON_MAP.get(interval, 10)})
    horizon = tf_cfg["horizon"]
    feat = build_features(df, btc_close=btc_close)
    close_arr = df["close"].to_numpy()

    if target_mode == "tb_atr":
        high_arr = df["high"].to_numpy()
        low_arr = df["low"].to_numpy()
        atr_arr = _atr(high_arr, low_arr, close_arr, period=14)
        target = make_target_triple_barrier(
            high_arr, low_arr, close_arr, atr_arr, horizon=horizon,
            upper_mult=tb_upper, lower_mult=tb_lower,
        )
    else:
        target = make_target(close_arr, horizon=horizon)

    mask = target >= 0
    feat = feat.loc[mask].reset_index(drop=True)
    target = target[mask]

    # Holdout = last 25% of data.
    split = int(len(feat) * 0.75)
    X_tr, y_tr = feat[FEATURE_NAMES].iloc[:split].to_numpy(), target[:split]
    X_te, y_te = feat[FEATURE_NAMES].iloc[split:].to_numpy(), target[split:]

    xgb, lgbm, rf, meta = train_ensemble(X_tr, y_tr, quick=False)
    proba, _ = ensemble_predict(xgb, lgbm, rf, meta, X_te)

    hc_table = sweep_thresholds(y_te, proba)
    best = pick_best(hc_table, interval)

    return {
        "symbol": symbol, "interval": interval, "horizon": horizon,
        "target_mode": target_mode,
        "tb_upper": tb_upper if target_mode == "tb_atr" else None,
        "tb_lower": tb_lower if target_mode == "tb_atr" else None,
        "n_test": int(len(y_te)),
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
    ap.add_argument("--intervals", nargs="*", default=["5m", "15m", "1h", "4h", "1d"])
    ap.add_argument("--target-mode", choices=["binary", "tb_atr"], default="binary")
    ap.add_argument("--tb-upper", type=float, default=1.5)
    ap.add_argument("--tb-lower", type=float, default=1.0)
    ap.add_argument("--scalp-upper", type=float, default=2.5, help="Override upper_mult for 5m/15m")
    ap.add_argument("--scalp-lower", type=float, default=1.5, help="Override lower_mult for 5m/15m")
    args = ap.parse_args()

    btc_by_iv = {iv: fetch_or_cache("BTCUSDT", iv, years=8.0)["close"].to_numpy() for iv in args.intervals}

    results = []
    for sym in args.symbols:
        for iv in args.intervals:
            btc_close = btc_by_iv.get(iv) if sym != "BTCUSDT" else None
            # Use scalp barriers for short TFs matching training config.
            if args.target_mode == "tb_atr" and iv in ("5m", "15m"):
                upper, lower = args.scalp_upper, args.scalp_lower
            else:
                upper, lower = args.tb_upper, args.tb_lower
            print(f"  analyzing {sym} {iv} [{args.target_mode}, upper={upper}, lower={lower}]…")
            r = analyze(sym, iv, btc_close, target_mode=args.target_mode, tb_upper=upper, tb_lower=lower)
            results.append(r)
            if "error" not in r:
                b = r["best"]
                print(f"    proba std={r['proba_std']:.3f}")
                print(f"    BEST: {b['key']} policy={b['policy']} "
                      f"precision={b['precision']:.1%} sig/day={b['sig_per_day']:.2f} "
                      f"({b['n_signals']} trades, {b['fraction']:.1%} of bars)")

    out = {
        "generated_at": str(pd.Timestamp.now()),
        "target_mode": args.target_mode,
        "policy": {
            "min_precision": MIN_PRECISION,
            "min_sig_per_day": MIN_SIG_PER_DAY,
            "max_sig_per_day": MAX_SIG_PER_DAY,
            "fallback_min_precision": FALLBACK_MIN_PRECISION,
            "fallback_min_sig_per_day": FALLBACK_MIN_SIG_PER_DAY,
            "rule": "max threshold with precision>=0.60 AND MIN<=sig/day<=MAX[interval]; fallback1 = max threshold with precision>=0.55 AND sig/day>=0.3; fallback2 = max precision with >=30 signals",
        },
        "results": results,
    }
    LOGS_DIR.mkdir(exist_ok=True)
    with open(LOGS_DIR / "best_thresholds.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n═══ Threshold sweep summary ═══")
    print(f"{'Symbol':<10} {'TF':<4} {'σ':>5} {'threshold':>9} {'precision':>10} {'sig/day':>8} {'trades':>7} {'policy':<35}")
    for r in results:
        if "error" in r:
            continue
        b = r["best"]
        print(f"  {r['symbol']:<10} {r['interval']:<4} {r['proba_std']:>5.3f} "
              f"{b['key'] or '—':>9} {b['precision']:>10.1%} "
              f"{b['sig_per_day']:>8.2f} {b['n_signals']:>7} {b['policy']:<35}")


if __name__ == "__main__":
    main()
