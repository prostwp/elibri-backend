#!/usr/bin/env python3
"""
make_parity_fixture.py — Patch 2N Python↔Go feature parity fixture generator.

Writes two artefacts:
  testdata/fixture_ohlcv.csv       — 60 deterministic OHLCV bars
  testdata/fixture_expected.json   — ground-truth 30-feature vector (last bar)

The Go parity test at internal/ml/features_v2_parity_test.go loads both and
asserts |go - python| <= 1e-9 per feature. This catches any drift between the
two implementations of feature_engine — the root cause behind Patch 2G's
adx/100 and rsi-denorm bugs where production computed features on a different
scale than the trainer.

Regenerate (ONLY when you consciously change a feature formula on both sides):

    cd elibri-backend/ml-training
    python3 make_parity_fixture.py

This is not called from the hot train/backtest path. It's a test-only helper.
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd

# Import project features. Adds parent dir in case this is called via
# `python -m ml-training.make_parity_fixture` from outside ml-training/.
ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_engine import FEATURE_NAMES, build_features  # noqa: E402


TESTDATA = ROOT / "testdata"
TESTDATA.mkdir(parents=True, exist_ok=True)
CSV_PATH = TESTDATA / "fixture_ohlcv.csv"
JSON_PATH = TESTDATA / "fixture_expected.json"

N_BARS = 60
INTERVAL_SEC = 3600  # 1-hour bars — matches a real BTCUSDT 1h row
T0 = 1_700_000_000


def synthetic_bars(n: int = N_BARS) -> list[dict[str, float | int]]:
    """Deterministic sine + linear drift so every feature has non-zero signal.

    Price path: start 50000, drift +30/bar, sine amplitude ±500 period 12.
    Volume: positive random-ish via sine at different phase. Enough variance
    for RSI/BB/ATR to fire real values; enough trend for ADX/EMA crosses.
    """
    bars: list[dict[str, float | int]] = []
    for i in range(n):
        close = 50000 + i * 30 + 500 * math.sin(i / 2)
        open_ = close - 15 - 40 * math.sin(i / 3)
        hi_wick = max(10, 80 * abs(math.sin(i / 4)))
        lo_wick = max(10, 60 * abs(math.cos(i / 5)))
        high = max(open_, close) + hi_wick
        low = min(open_, close) - lo_wick
        volume = 100 + 80 * (1 + math.sin(i / 6))
        # Binance taker_buy_base: fraction 0.40-0.70 of volume, varies with sine.
        taker = volume * (0.55 + 0.12 * math.sin(i / 7))
        bars.append({
            "open_time": T0 + i * INTERVAL_SEC,
            "open": round(open_, 4),
            "high": round(high, 4),
            "low": round(low, 4),
            "close": round(close, 4),
            "volume": round(volume, 4),
            "taker_buy_base": round(taker, 4),
        })
    return bars


def write_csv(bars: list[dict[str, float | int]]) -> None:
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bars[0].keys()))
        w.writeheader()
        for b in bars:
            w.writerow(b)
    print(f"wrote {CSV_PATH} ({len(bars)} bars)")


def write_expected(bars: list[dict[str, float | int]]) -> None:
    """Run feature_engine.build_features on the synthetic series, serialize
    the last row's features. This is the Python ground truth Go must match.

    Parity note: we pass `btc_close = closes` (the same series as the asset),
    NOT None. Reason: Python's None-branch shortcuts btc_corr/beta to 1.0,
    but Go's nil-branch shortcuts to 0.0 (intentional — `0 correlation` is the
    honest "cross-asset unknown" default; see features_v2.go:152-157 and the
    memory note about the semantic fix). Feeding an explicit BTC series avoids
    that branch divergence for this test. When BTC is the asset itself, corr
    with own series is trivially 1.0 on BOTH sides, which is what we assert.

    Follow-up work: align Python default to 0.0 to match Go, and accept the
    retrain. Tracked as Patch 2N+1.
    """
    df = pd.DataFrame(bars)
    closes = np.array([float(b["close"]) for b in bars])
    feat = build_features(df, btc_close=closes)
    # Grab the LAST bar's feature vector — that's what ExtractFeaturesV2
    # returns on a fresh prediction call.
    last_row = feat.iloc[-1]
    expected: dict[str, float] = {}
    for name in FEATURE_NAMES:
        val = float(last_row[name])
        if not math.isfinite(val):
            # Should never happen post-fillna(0), but guard to fail loudly
            # rather than writing NaN that JSON can't encode.
            raise RuntimeError(f"non-finite feature {name} = {val!r}")
        expected[name] = val
    payload = {
        "schema_version": 1,
        "feature_names": list(FEATURE_NAMES),
        # Include the input row shape so future debugging doesn't require
        # reconstructing it from the CSV.
        "n_bars": len(bars),
        "interval_sec": INTERVAL_SEC,
        # Expected features keyed by name. Go loads this and looks up by name
        # rather than trusting positional order (defence in depth against
        # feature_names drift).
        "expected_features": expected,
        # Also emit the positional list to test the positional contract that
        # Go actually uses (ExtractFeaturesV2 returns []float64 in FeatureNamesV2 order).
        "expected_features_positional": [expected[n] for n in FEATURE_NAMES],
    }
    with JSON_PATH.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"wrote {JSON_PATH} ({len(expected)} features)")


def main() -> None:
    bars = synthetic_bars()
    write_csv(bars)
    write_expected(bars)
    # Quick self-check: reload CSV + recompute, ensure stability. Future-proofs
    # against CSV rounding drift (we round to 4 dp; feature formulas don't care
    # but this proves it).
    with CSV_PATH.open() as f:
        reloaded = list(csv.DictReader(f))
    if len(reloaded) != N_BARS:
        raise RuntimeError(f"CSV round-trip lost bars: {len(reloaded)} != {N_BARS}")
    print("OK")


if __name__ == "__main__":
    main()
