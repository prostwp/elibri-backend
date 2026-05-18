"""Tests for idea_engine — written BEFORE implementation (TDD).

Per `feedback_ml_pipeline_safety.md`: any new ML/analytics file ships with
2-3 unit tests minimum, written first. This file covers the pilot for
Crypto ML Trader (BTC 4h).

Run from telegram-bot/:
    .venv/bin/pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from idea_engine import (  # noqa: E402
    Setup,
    SignalContext,
    build_setup,
    infer_regime,
    score_confidence,
)


# ─── Confidence scoring (rule-based, deterministic) ─────────────────────────

def test_confidence_zero_when_all_signals_neutral():
    """No conviction signals → confidence ≤ 25 (low band)."""
    ctx = SignalContext(
        rsi=50.0, macd_line=0.0, macd_signal=0.0,
        trend="sideways", price=100.0, support=95.0, resistance=105.0, atr=2.0,
        regime="chop",
    )
    score = score_confidence(ctx, direction="long")
    assert 0 <= score <= 25, f"neutral signals gave {score}, expected ≤25"


def test_confidence_high_when_oversold_uptrend():
    """RSI<30 + uptrend + MACD bull-cross + price near support → confidence ≥ 65."""
    ctx = SignalContext(
        rsi=25.0, macd_line=0.5, macd_signal=0.2,
        trend="uptrend", price=96.0, support=95.0, resistance=110.0, atr=2.0,
        regime="bull",
    )
    score = score_confidence(ctx, direction="long")
    assert score >= 65, f"strong long setup gave {score}, expected ≥65"


def test_confidence_high_when_overbought_downtrend():
    """RSI>70 + downtrend + MACD bear-cross + price near resistance → high short conviction."""
    ctx = SignalContext(
        rsi=78.0, macd_line=-0.5, macd_signal=-0.2,
        trend="downtrend", price=109.0, support=90.0, resistance=110.0, atr=2.0,
        regime="bear",
    )
    score = score_confidence(ctx, direction="short")
    assert score >= 65, f"strong short setup gave {score}, expected ≥65"


def test_confidence_in_range_0_to_100():
    """Confidence is bounded — never negative, never above 100."""
    extreme_long = SignalContext(
        rsi=10.0, macd_line=2.0, macd_signal=0.1,
        trend="uptrend", price=95.5, support=95.0, resistance=120.0, atr=1.0,
        regime="bull",
    )
    score = score_confidence(extreme_long, direction="long")
    assert 0 <= score <= 100


def test_confidence_penalises_against_regime():
    """Long in bear regime → lower confidence than long in bull regime (same setup)."""
    base = dict(rsi=40.0, macd_line=0.3, macd_signal=0.1,
                trend="uptrend", price=100.0, support=95.0, resistance=110.0, atr=2.0)
    s_bull = score_confidence(SignalContext(**base, regime="bull"), direction="long")
    s_bear = score_confidence(SignalContext(**base, regime="bear"), direction="long")
    assert s_bull > s_bear, f"bull={s_bull} should beat bear={s_bear} for same long setup"


# ─── Trade idea construction (always returns a Setup) ───────────────────────

def test_build_setup_always_returns_setup():
    """Per requirement: idea is ALWAYS given, even when neutral. Confidence reflects strength."""
    neutral = SignalContext(
        rsi=50.0, macd_line=0.0, macd_signal=0.0,
        trend="sideways", price=100.0, support=95.0, resistance=105.0, atr=2.0,
        regime="chop",
    )
    setup = build_setup(neutral)
    assert isinstance(setup, Setup)
    assert setup.direction in ("long", "short")
    assert setup.entry > 0
    assert setup.stop_loss > 0
    assert setup.take_profit > 0


def test_long_setup_has_sl_below_entry_tp_above():
    """Long → SL below entry, TP above entry. Risk/reward sane."""
    bull_ctx = SignalContext(
        rsi=28.0, macd_line=0.4, macd_signal=0.1,
        trend="uptrend", price=100.0, support=95.0, resistance=115.0, atr=2.0,
        regime="bull",
    )
    setup = build_setup(bull_ctx)
    assert setup.direction == "long"
    assert setup.stop_loss < setup.entry, "long SL must be below entry"
    assert setup.take_profit > setup.entry, "long TP must be above entry"
    # R/R ≥ 1 sanity
    risk = setup.entry - setup.stop_loss
    reward = setup.take_profit - setup.entry
    assert risk > 0 and reward > 0


def test_short_setup_has_sl_above_entry_tp_below():
    """Short → SL above entry, TP below."""
    bear_ctx = SignalContext(
        rsi=75.0, macd_line=-0.4, macd_signal=-0.1,
        trend="downtrend", price=109.0, support=90.0, resistance=110.0, atr=2.0,
        regime="bear",
    )
    setup = build_setup(bear_ctx)
    assert setup.direction == "short"
    assert setup.stop_loss > setup.entry
    assert setup.take_profit < setup.entry


def test_setup_carries_reasoning_list():
    """Setup explains WHY the direction was picked (for the bot's analysis section)."""
    ctx = SignalContext(
        rsi=28.0, macd_line=0.4, macd_signal=0.1,
        trend="uptrend", price=100.0, support=95.0, resistance=115.0, atr=2.0,
        regime="bull",
    )
    setup = build_setup(ctx)
    assert isinstance(setup.reasons, list)
    assert len(setup.reasons) > 0
    # Reasons should mention triggering signals — non-empty strings.
    for r in setup.reasons:
        assert isinstance(r, str) and len(r) > 0


# ─── Edge cases (defensive — won't kill bot) ────────────────────────────────

def test_setup_handles_zero_atr():
    """ATR = 0 (flat candle, rare but real) → SL/TP fall back to support/resistance."""
    ctx = SignalContext(
        rsi=50.0, macd_line=0.0, macd_signal=0.0,
        trend="sideways", price=100.0, support=95.0, resistance=105.0, atr=0.0,
        regime="chop",
    )
    setup = build_setup(ctx)
    assert setup.entry > 0 and setup.stop_loss > 0 and setup.take_profit > 0


def test_infer_regime_warm_on_short_history():
    """< 250 bars → 'warm' (not enough data for EMA200)."""
    label, slope = infer_regime([100.0] * 50)
    assert label == "warm"
    assert slope == 0.0


def test_infer_regime_bull_on_uptrend():
    """Steady uptrend over 300 bars → 'bull'."""
    closes = [100.0 + i * 1.0 for i in range(300)]
    label, slope = infer_regime(closes)
    assert label == "bull"
    assert slope > 0


def test_infer_regime_bear_on_downtrend():
    closes = [400.0 - i * 1.0 for i in range(300)]
    label, slope = infer_regime(closes)
    assert label == "bear"
    assert slope < 0


def test_infer_regime_chop_on_flat():
    """Flat oscillation → 'chop'."""
    closes = [100.0 + (i % 5 - 2) * 0.3 for i in range(300)]
    label, slope = infer_regime(closes)
    assert label in ("chop", "warm")
    # Slope bounded in [-5, 5]
    assert -5.0 <= slope <= 5.0


def test_setup_handles_inverted_levels():
    """Sometimes support > resistance briefly (level recompute lag) — must not crash."""
    bad_ctx = SignalContext(
        rsi=50.0, macd_line=0.0, macd_signal=0.0,
        trend="sideways", price=100.0, support=110.0, resistance=90.0, atr=2.0,
        regime="chop",
    )
    setup = build_setup(bad_ctx)
    # Just doesn't raise — values may be approximate.
    assert setup is not None
