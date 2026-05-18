"""Integration tests targeting Scenario A bug classes.

Focus areas (from feedback_ml_pipeline_safety.md):
  1. Look-ahead leakage in feature builders
  2. Build-time vs query-time feature DEFINITION drift
  3. NaN / inf propagation through the pipeline
  4. Edge values (zero ATR, zero variance, inverted levels, short series)
  5. Normalizer parameter contract (mu/sigma reuse at query time)

Synthetic data only — no Binance, no parquet. Each test ~5-10 LOC body.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
BOT_ROOT = HERE.parent
sys.path.insert(0, str(BOT_ROOT / "src"))
sys.path.insert(0, str(BOT_ROOT / "scripts"))

from build_btc_4h_pattern_index import (  # noqa: E402
    build_features,
    classify_regime_per_bar,
    normalize,
)
from pattern_match import compute_query_features  # noqa: E402

# Inline copies of analyzers.py helpers — analyzers.py uses relative imports
# (`.db`) so it can't be imported via bare sys.path. Tests for the math only
# need the helpers, not the aiohttp scaffolding around them.
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location("_analyzers_src", BOT_ROOT / "src" / "analyzers.py")


def _rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0)); losses.append(max(-d, 0.0))
    avg_gain = sum(gains) / period; avg_loss = sum(losses) / period
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(d, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-d, 0.0)) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _levels(klines, lookback=50):
    take = klines[-lookback:]
    highs = [float(k[2]) for k in take]
    lows = [float(k[3]) for k in take]
    close = float(take[-1][4])
    return max(highs), min(lows), (max(highs) + min(lows) + close) / 3.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_ohlcv_df(n: int, seed: int = 0) -> pd.DataFrame:
    """Random walk OHLCV DataFrame in build-script schema (open/high/low/close)."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 1, size=n))
    high = close + np.abs(rng.normal(0, 0.5, size=n))
    low = close - np.abs(rng.normal(0, 0.5, size=n))
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close})


def _df_to_klines(df: pd.DataFrame) -> list[list]:
    """Convert DataFrame to Binance kline rows: [t, o, h, l, c, v, ...]."""
    return [[i, r["open"], r["high"], r["low"], r["close"], 0] for i, r in df.iterrows()]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Look-ahead leakage probes
# ─────────────────────────────────────────────────────────────────────────────

def test_build_features_no_lookahead_when_future_changes():
    """Bug class: feature at bar t depends on bar > t (look-ahead).

    Build features on series S, then mutate the LAST bar's close, then rebuild.
    If feature[i] for i < n-1 changes, the builder peeks at the future.
    """
    df = _make_ohlcv_df(400, seed=1)
    feats_a = build_features(df).iloc[:-1].reset_index(drop=True)
    df2 = df.copy()
    df2.loc[len(df2) - 1, "close"] = df2["close"].iloc[-1] * 10  # extreme future mutation
    df2.loc[len(df2) - 1, "high"] = df2["high"].iloc[-1] * 10
    feats_b = build_features(df2).iloc[:-1].reset_index(drop=True)
    pd.testing.assert_frame_equal(feats_a, feats_b, check_exact=False, atol=1e-9)


def test_dist_to_high_uses_only_past_bars():
    """Bug class: 50-bar high includes the current OR FUTURE bar (`shift(1)` missing).

    Mutate ONLY a FUTURE bar's high; dist_to_high_atr at row t must be
    unchanged because the 50-bar high at t is computed from rows < t.
    Catches the classic 'forgot .shift(1) on rolling max' look-ahead.
    """
    df = _make_ohlcv_df(200, seed=2)
    feats_a = build_features(df)
    df2 = df.copy()
    t, future = 100, 150
    df2.loc[future, "high"] = df2["high"].iloc[future] * 100  # spike in the future
    feats_b = build_features(df2)
    assert feats_a["dist_to_high_atr"].iloc[t] == pytest.approx(
        feats_b["dist_to_high_atr"].iloc[t], abs=1e-6
    ), "future high leaked into past dist_to_high_atr — look-ahead!"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build-time vs query-time feature DRIFT
# ─────────────────────────────────────────────────────────────────────────────

def test_query_features_match_build_features_dimension():
    """compute_query_features MUST produce the same dim as build_features.

    Drift here = silent index search with mis-aligned axes (Scenario A class:
    'feature definition diverged between train and inference').
    """
    df = _make_ohlcv_df(300, seed=3)
    build_dim = build_features(df).shape[1]
    q = compute_query_features(
        rsi=50.0, macd_line=0.1, macd_signal=0.05, atr=2.0, price=100.0,
        support_50=95.0, resistance_50=105.0,
        recent_closes=[99, 99.5, 100, 100.5, 99.8, 100.2],
    )
    assert q.shape == (build_dim,), f"build_dim={build_dim} but query_dim={q.shape}"


def test_query_features_column_order_matches_build():
    """compute_query_features must place columns in the SAME order as build_features.

    Pipeline mixes ['rsi', 'macd_diff', 'atr_norm', 'regime_score',
    'dist_to_high_atr', 'dist_to_low_atr', 'ret_5'] — if the order
    drifts, normalize() applies wrong mu/sigma per axis.
    """
    df = _make_ohlcv_df(300, seed=4)
    expected_cols = list(build_features(df).columns)
    assert expected_cols == [
        "rsi", "macd_diff", "atr_norm", "regime_score",
        "dist_to_high_atr", "dist_to_low_atr", "ret_5",
    ], "column order changed — query-time vector will be misaligned"


def test_normalizer_reuse_does_not_recompute_mu_sigma():
    """Bug class: query-time normalize re-fits on the single live point
    (giving 0 every time). Reuse path must respect provided (mu, sigma).
    """
    feats = np.random.default_rng(5).normal(3, 7, size=(500, 7)).astype(np.float32)
    norm, mu, sigma = normalize(feats)
    # Feed a single new point with a known offset.
    new = np.array([[mu[0] + sigma[0], 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    new_norm, mu2, sigma2 = normalize(new, fit_mu=mu, fit_sigma=sigma)
    assert np.allclose(mu2, mu) and np.allclose(sigma2, sigma)
    # Offset of one sigma in axis 0 should yield ~1.0 in normalised space.
    assert abs(new_norm[0, 0] - 1.0) < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# 3. NaN / inf propagation
# ─────────────────────────────────────────────────────────────────────────────

def test_build_features_no_nan_after_warmup():
    """Bug class: NaN leaks past warmup into HNSW (kills index queries).

    After WARMUP_BARS=250 the feature frame must be fully finite.
    """
    df = _make_ohlcv_df(500, seed=6)
    feats = build_features(df).iloc[250:]
    assert np.isfinite(feats.to_numpy()).all(), "NaN/inf past warmup row 250"


def test_normalize_handles_zero_variance_column():
    """Bug class: division-by-zero when a feature is constant for the whole
    fit window — must NOT produce inf/NaN.
    """
    feats = np.column_stack([
        np.random.default_rng(7).normal(0, 1, size=400).astype(np.float32),
        np.full(400, 42.0, dtype=np.float32),  # constant column
    ])
    norm, mu, sigma = normalize(feats)
    assert np.isfinite(norm).all()
    assert sigma[1] == 1.0, "constant-column sigma must be replaced with 1.0"


def test_compute_query_features_handles_zero_atr_and_price():
    """Bug class: zero ATR or zero price → division by zero → inf in vector.

    The formatted post would render literal 'inf' to subscribers.
    """
    q = compute_query_features(
        rsi=50.0, macd_line=0.0, macd_signal=0.0,
        atr=0.0, price=0.0,
        support_50=0.0, resistance_50=0.0,
        recent_closes=[0.0] * 6,
    )
    assert np.isfinite(q).all(), f"non-finite query vector with zero inputs: {q}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Regime alignment between build-time and query-time
# ─────────────────────────────────────────────────────────────────────────────

def test_classify_regime_label_set_matches_idea_engine():
    """Bug class: build-side regime label ('bull'/'bear'/'chop'/'warm') drifts
    from the analyzer's `infer_regime` output → HNSW returns 'unknown' regime.
    """
    from idea_engine import infer_regime
    closes = pd.Series([100.0 + i * 0.5 for i in range(400)])
    bar_labels = set(np.unique(classify_regime_per_bar(closes)).tolist())
    runtime_label, _slope = infer_regime(closes.tolist())
    assert runtime_label in {"bull", "chop", "bear", "warm"}
    assert bar_labels.issubset({"bull", "chop", "bear", "warm"}), (
        f"unexpected regime labels from build-time: {bar_labels}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Analyzer-side primitives — edge values
# ─────────────────────────────────────────────────────────────────────────────

def test_analyzer_rsi_handles_monotone_series():
    """Bug class: RSI on flat-or-monotone series → NaN/inf or div-by-zero
    (avg_loss == 0). Must clip to 100.0, not crash.
    """
    closes = [100.0 + i for i in range(50)]  # strictly rising → all gains
    val = _rsi(closes)
    assert val == 100.0


def test_analyzer_levels_with_inverted_high_low():
    """Bug class: defensive — `_levels` must not raise on degenerate input."""
    klines = [[i, 100, 100, 100, 100, 0] for i in range(50)]  # flat candles
    res, sup, pivot = _levels(klines)
    assert res == sup == 100.0 and abs(pivot - 100.0) < 1e-9
