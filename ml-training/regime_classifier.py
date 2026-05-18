"""regime_classifier.py — bull/chop/bear classifier for regime-aware CV.

Operates on a daily price series (or any TF, but the cutoffs below are tuned
on 1d). Produces two outputs:

  classify_bars(close, atr) → np.ndarray of strings in {"bull", "chop", "bear"}
       per-bar regime label, computed from EMA-200 slope normalized by ATR-14.
       Same primitive as the `regime_score` feature in feature_engine, but
       discretized for stratification rather than fed as a continuous signal.

  bucket_windows(timestamps, regimes) → list[dict]
       Builds maximal contiguous regime segments for use as test windows in
       walk-forward CV. Sample output:
         [{"start": "2022-06-01", "end": "2023-01-15", "regime": "bear",
           "n_bars": 184, "name": "bear_2022_h2"}, ...]

Why regime-aware CV?
  Pure temporal walk-forward (B-style: train 24mo / test 3mo expanding) makes
  the test set bull-heavy in 2024-2026 because crypto went vertical. Models
  that memorize "always long" win backtests but die in production once chop
  returns. Regime-aware CV ensures every fold either tests on a different
  regime than its train (true OOD) or explicitly reports per-regime metrics.

Reference: project_v3_layered_fusion.md §"Regime-aware methodology".
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
import pandas as pd


# Threshold tuning (BTC 1d, 2018-2026 empirical):
#   p25 ≈ -0.020, p75 ≈ +0.046, p10 ≈ -0.086, p90 ≈ +0.056.
#   Min/max observed: -0.19 (Jan 2023 capitulation) / +0.087 (Apr 2021 peak).
#
# Setting cutoffs at the 70th / 30th percentile gives a roughly 30/40/30
# bull/chop/bear split — meaningful regime windows without leaving "chop"
# overstuffed. Sanity-checked against canonical regime peaks:
#     +0.087 score ↔ 2021-04 BTC top  ✓ (clear bull)
#     -0.19 score  ↔ 2023-01 capitulation bottom  ✓ (clear bear)

DEFAULT_BULL_THRESHOLD = 0.04
DEFAULT_BEAR_THRESHOLD = -0.04

EMA_PERIOD = 200
SLOPE_WINDOW = 50  # bars used for the EMA slope numerator


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """Standard EMA (matches feature_engine._ema)."""
    if len(values) == 0:
        return values
    k = 2.0 / (period + 1.0)
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = values[i] * k + out[i - 1] * (1 - k)
    return out


def _wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder ATR (matches feature_engine._atr Patch 2N+2 byte-equal version)."""
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    if n < period + 1:
        return atr
    tr = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    atr[period] = tr[1: period + 1].mean()
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def compute_regime_score(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    ema_period: int = EMA_PERIOD, slope_window: int = SLOPE_WINDOW,
) -> np.ndarray:
    """ATR-normalized EMA slope. Same definition as feature_engine.regime_score
    but exposed standalone so consumers don't have to build the full feature
    matrix just to bucket regimes.

    Output is in (approximately) ATR-units per bar. Positive = trend rising,
    negative = falling. 0-200 warmup bars are zeroed.
    """
    ema = _ema(close, ema_period)
    atr = _wilder_atr(high, low, close, period=14)
    slope = np.zeros_like(close, dtype=np.float64)
    warmup = ema_period + slope_window
    for i in range(warmup, len(close)):
        dy = ema[i] - ema[i - slope_window]
        denom = slope_window * (atr[i] + 1e-12)
        slope[i] = dy / denom
    return np.clip(slope, -5.0, 5.0)


def classify_bars(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    bull_thr: float = DEFAULT_BULL_THRESHOLD,
    bear_thr: float = DEFAULT_BEAR_THRESHOLD,
) -> np.ndarray:
    """Per-bar regime label as a numpy array of dtype '<U4'.

    Warmup bars (regime_score == 0) are tagged "warm" so callers can drop them
    rather than misinterpret as "chop".
    """
    score = compute_regime_score(high, low, close)
    out = np.full(len(close), "warm", dtype="<U4")
    out[score >= bull_thr] = "bull"
    out[score <= bear_thr] = "bear"
    out[(score > bear_thr) & (score < bull_thr) & (score != 0.0)] = "chop"
    return out


# ─── Window bucketing ────────────────────────────────────────────────────────

@dataclass
class RegimeWindow:
    name: str            # human-readable label, e.g. "bull_2024_h1"
    regime: str          # "bull" | "chop" | "bear"
    start: str           # ISO date string (UTC)
    end: str             # ISO date string (exclusive)
    start_idx: int       # index into the parent timestamp array (inclusive)
    end_idx: int         # index into the parent timestamp array (exclusive)
    n_bars: int


def bucket_windows(
    timestamps: pd.DatetimeIndex, regimes: np.ndarray, min_window_bars: int = 20,
) -> List[RegimeWindow]:
    """Group consecutive same-regime bars into contiguous windows.

    Skips warmup bars and any segment shorter than `min_window_bars` (those are
    too noisy to use as a CV test fold — typical setup needs ≥30 days of test).

    Returns windows ordered by `start`. The `name` field disambiguates multiple
    same-regime windows by year+half (e.g. "bull_2024_h1", "bull_2025_h2").
    """
    if len(timestamps) != len(regimes):
        raise ValueError("timestamps and regimes must be same length")

    ts = pd.DatetimeIndex(timestamps)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")

    windows: List[RegimeWindow] = []
    if len(regimes) == 0:
        return windows

    # State machine: track the start of the current regime run.
    cur_regime = regimes[0]
    cur_start = 0
    for i in range(1, len(regimes)):
        if regimes[i] != cur_regime:
            if cur_regime not in ("warm",) and (i - cur_start) >= min_window_bars:
                windows.append(_make_window(cur_regime, cur_start, i, ts))
            cur_regime = regimes[i]
            cur_start = i
    # Flush trailing run.
    if cur_regime not in ("warm",) and (len(regimes) - cur_start) >= min_window_bars:
        windows.append(_make_window(cur_regime, cur_start, len(regimes), ts))

    # Disambiguate names: bull_2024 + bull_2024 → bull_2024_a / bull_2024_b.
    # (Still rare given the half-year suffix; here as belt-and-braces.)
    seen: dict = {}
    for w in windows:
        if w.name in seen:
            seen[w.name] += 1
            w.name = f"{w.name}_{chr(96 + seen[w.name])}"  # _b, _c, ...
        else:
            seen[w.name] = 1
    return windows


def _make_window(regime: str, start_idx: int, end_idx: int, ts: pd.DatetimeIndex) -> RegimeWindow:
    start = ts[start_idx]
    end = ts[end_idx - 1] + pd.Timedelta(seconds=1)  # exclusive
    half = "h1" if start.month <= 6 else "h2"
    name = f"{regime}_{start.year}_{half}"
    return RegimeWindow(
        name=name, regime=regime,
        start=start.isoformat(), end=end.isoformat(),
        start_idx=int(start_idx), end_idx=int(end_idx),
        n_bars=int(end_idx - start_idx),
    )


# ─── Five-window picker (used by wfcv_regime_aware) ──────────────────────────

def pick_five_test_windows(windows: List[RegimeWindow]) -> List[RegimeWindow]:
    """Choose 5 distinct test windows covering as many regimes as possible.

    Strategy:
      1. Take the largest window of each regime (bull, chop, bear) — ensures
         all 3 regimes are represented if data has them. (3 windows max.)
      2. Fill the remaining slots with the next-largest windows of any regime
         that still leave the train/test ratio sensible.

    Returns a list of up to 5 RegimeWindow ordered by `start` (so wfcv prints
    them chronologically).
    """
    if not windows:
        return []
    by_regime = {}
    for w in windows:
        by_regime.setdefault(w.regime, []).append(w)
    for r in by_regime:
        by_regime[r].sort(key=lambda w: -w.n_bars)

    chosen: List[RegimeWindow] = []
    # Step 1: largest per regime.
    for r in ("bull", "bear", "chop"):
        if by_regime.get(r):
            chosen.append(by_regime[r][0])
    # Step 2: fill up to 5 with next-largest of any regime, no duplicates.
    pool = sorted(
        [w for r, ws in by_regime.items() for w in ws if w not in chosen],
        key=lambda w: -w.n_bars,
    )
    for w in pool:
        if len(chosen) >= 5:
            break
        chosen.append(w)
    chosen.sort(key=lambda w: w.start)
    return chosen[:5]


# ─── CLI: classify a parquet and dump regime manifest ────────────────────────

def main() -> None:
    import argparse, json, pathlib
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/BTCUSDT_1d.parquet",
                    help="Path to OHLCV parquet (1d recommended for regime classification)")
    ap.add_argument("--output", default="data/regime_manifest_BTCUSDT_1d.json")
    ap.add_argument("--bull-thr", type=float, default=DEFAULT_BULL_THRESHOLD)
    ap.add_argument("--bear-thr", type=float, default=DEFAULT_BEAR_THRESHOLD)
    ap.add_argument("--min-window", type=int, default=20,
                    help="Min bars for a regime window to be retained")
    args = ap.parse_args()

    path = pathlib.Path(args.parquet)
    if not path.is_absolute():
        path = pathlib.Path(__file__).parent / path
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} bars from {path}: "
          f"{df['open_time'].min()} → {df['open_time'].max()}")

    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    regimes = classify_bars(high, low, close, args.bull_thr, args.bear_thr)
    counts = {r: int((regimes == r).sum()) for r in ("warm", "bull", "chop", "bear")}
    print(f"Regime distribution (bars): {counts}")

    windows = bucket_windows(df["open_time"], regimes, min_window_bars=args.min_window)
    print(f"\nDetected {len(windows)} contiguous regime windows "
          f"(min {args.min_window} bars):")
    for w in windows:
        print(f"  {w.name:25s} {w.regime:5s} {w.start[:10]} → {w.end[:10]}  ({w.n_bars} bars)")

    five = pick_five_test_windows(windows)
    print(f"\nFive picked test windows for WFCV (regime-stratified):")
    for w in five:
        print(f"  · {w.name:25s} {w.regime:5s} {w.n_bars:4d} bars")

    out_path = pathlib.Path(args.output)
    if not out_path.is_absolute():
        out_path = pathlib.Path(__file__).parent / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "source": str(path),
            "bull_threshold": args.bull_thr,
            "bear_threshold": args.bear_thr,
            "ema_period": EMA_PERIOD,
            "slope_window": SLOPE_WINDOW,
            "regime_counts": counts,
            "windows": [asdict(w) for w in windows],
            "five_picked": [asdict(w) for w in five],
        }, f, indent=2)
    print(f"\nManifest → {out_path}")


if __name__ == "__main__":
    main()
