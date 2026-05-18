"""Tests for pattern_match — written BEFORE implementation (TDD).

The pattern matcher loads a pre-built HNSW index + forward-returns array
and answers: "find K historical bars whose features are nearest to this
current bar; report what happened in the next H bars across them."

These tests use synthetic numpy fixtures, no real BTC parquet — fast.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pattern_match import (  # noqa: E402
    PatternIndex,
    PatternSummary,
    aggregate,
    format_summary_text,
)


@pytest.fixture
def synth_index(tmp_path):
    """Build a tiny synthetic index in tmp_path:
       1000 bars × 5 features, with forward returns drawn from N(0, 1%)
    so half are wins.
    """
    rng = np.random.default_rng(42)
    n, dim = 1000, 5
    features = rng.normal(0, 1, size=(n, dim)).astype(np.float32)
    fwd_returns = rng.normal(0, 0.01, size=n).astype(np.float32)
    # Regimes alternating: half bull, half bear, some chop in middle.
    regimes = np.array(
        ["bull"] * 400 + ["chop"] * 200 + ["bear"] * 400, dtype="<U4"
    )

    idx = PatternIndex(symbol="TEST", interval="4h", dim=dim)
    idx.build(features, fwd_returns, regimes)
    idx.save(tmp_path / "test_index")
    return tmp_path / "test_index"


# ─── Loading + querying ──────────────────────────────────────────────────────

def test_index_loads_after_save(synth_index):
    idx = PatternIndex.load(synth_index)
    assert idx.dim == 5
    assert idx.n_items == 1000


def test_query_returns_topk(synth_index):
    idx = PatternIndex.load(synth_index)
    rng = np.random.default_rng(99)
    query = rng.normal(0, 1, size=5).astype(np.float32)
    indices, distances = idx.query(query, top_k=50)
    assert len(indices) == 50
    assert len(distances) == 50
    assert distances[0] <= distances[-1], "distances must be sorted ascending"


def test_query_with_topk_larger_than_index(synth_index):
    """Asking for more neighbours than the index has → returns whatever exists."""
    idx = PatternIndex.load(synth_index)
    query = np.zeros(5, dtype=np.float32)
    indices, _ = idx.query(query, top_k=5000)
    assert len(indices) == 1000  # capped at index size


# ─── Aggregation math ────────────────────────────────────────────────────────

def test_aggregate_empty_returns_safe():
    """Empty input → safe summary (no crash)."""
    summary = aggregate(
        np.array([], dtype=np.float32),
        np.array([], dtype="<U4"),
        direction="long",
    )
    assert summary.n_matches == 0
    assert summary.win_rate == 0.0


def test_aggregate_long_direction_winrate():
    """Long: positive forward returns count as wins. 60% positive → win_rate 0.6."""
    fwd = np.array([0.01, 0.02, 0.005, -0.01, -0.005,
                    0.01, 0.02, -0.01, 0.001, 0.003], dtype=np.float32)
    regs = np.array(["bull"] * 10, dtype="<U4")
    summary = aggregate(fwd, regs, direction="long")
    assert summary.n_matches == 10
    assert abs(summary.win_rate - 0.7) < 1e-6  # 7 positive of 10
    assert summary.mean_return > 0


def test_aggregate_short_direction_winrate():
    """Short: NEGATIVE forward returns count as wins."""
    fwd = np.array([0.01, 0.02, -0.005, -0.01, -0.005,
                    -0.01, 0.02, -0.01, -0.001, 0.003], dtype=np.float32)
    regs = np.array(["bear"] * 10, dtype="<U4")
    summary = aggregate(fwd, regs, direction="short")
    assert summary.n_matches == 10
    assert abs(summary.win_rate - 0.6) < 1e-6  # 6 negative of 10


def test_aggregate_regime_breakdown():
    """Regime split helps spot regime-specific edges."""
    fwd = np.array([0.01, 0.02, -0.01, -0.02], dtype=np.float32)
    regs = np.array(["bull", "bull", "bear", "chop"], dtype="<U4")
    summary = aggregate(fwd, regs, direction="long")
    assert summary.regime_counts["bull"] == 2
    assert summary.regime_counts["bear"] == 1
    assert summary.regime_counts["chop"] == 1


# ─── Text formatting ─────────────────────────────────────────────────────────

def test_format_text_includes_n_and_winrate():
    summary = PatternSummary(
        n_matches=87, win_rate=0.62, mean_return=0.018,
        median_return=0.012, max_drawdown=-0.03,
        regime_counts={"bull": 50, "chop": 25, "bear": 12},
    )
    txt = format_summary_text(summary)
    assert "87" in txt
    assert "62" in txt or "62%" in txt
    assert "1.8" in txt or "1.80" in txt or "+1.8" in txt


def test_format_text_handles_zero_matches():
    """Empty summary → graceful 'недостаточно данных' message."""
    summary = PatternSummary(
        n_matches=0, win_rate=0.0, mean_return=0.0,
        median_return=0.0, max_drawdown=0.0, regime_counts={},
    )
    txt = format_summary_text(summary)
    assert "недостаточно" in txt.lower() or "no" in txt.lower() or "—" in txt


# ─── Round-trip (build → save → load → query → aggregate) ────────────────────

def test_enrich_returns_none_when_index_missing(tmp_path):
    """Non-existent index dir → enrich returns None (graceful fallback)."""
    from pattern_match import enrich_with_pattern
    out = enrich_with_pattern(
        "NONEXIST", "4h", np.zeros(7, dtype=np.float32),
        direction="long", base_dir=tmp_path,
    )
    assert out is None


def test_compute_query_features_shape():
    from pattern_match import compute_query_features
    vec = compute_query_features(
        rsi=50.0, macd_line=0.1, macd_signal=0.05,
        atr=2.0, price=100.0,
        support_50=95.0, resistance_50=105.0,
        recent_closes=[99, 99.5, 100, 100.5, 99.8, 100.2],
    )
    assert vec.shape == (7,)
    assert vec.dtype == np.float32


def test_round_trip_smoke(synth_index):
    """End-to-end: load index, query, aggregate, format. No crash."""
    idx = PatternIndex.load(synth_index)
    rng = np.random.default_rng(7)
    query = rng.normal(0, 1, size=5).astype(np.float32)
    indices, _ = idx.query(query, top_k=50)
    fwd = idx.forward_returns[indices]
    regs = idx.regimes[indices]
    summary = aggregate(fwd, regs, direction="long")
    txt = format_summary_text(summary)
    assert isinstance(txt, str) and len(txt) > 20
