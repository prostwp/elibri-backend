"""Tests for templates — written before implementation (TDD)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from idea_engine import Setup, SignalContext  # noqa: E402
from templates import (  # noqa: E402
    AUTHOR_TEMPLATES,
    PostContext,
    render_post,
)


@pytest.fixture
def sample_ctx() -> PostContext:
    sig = SignalContext(
        rsi=28.0, macd_line=0.4, macd_signal=0.1,
        trend="uptrend", price=100.0, support=95.0, resistance=115.0, atr=2.0,
        regime="bull",
    )
    setup = Setup(
        direction="long", entry=100.0, stop_loss=97.0, take_profit=104.0,
        confidence=72,
        reasons=["RSI=28 — перепроданность", "восходящий тренд по EMA"],
    )
    return PostContext(
        author_name="TA Trader (BTC)",
        symbol="BTCUSDT",
        interval="4h",
        price_now=100.0,
        change_24h_pct=1.5,
        signal_ctx=sig,
        setup=setup,
    )


def test_render_includes_all_three_blocks(sample_ctx):
    """Output must have ИДЕЯ, АНАЛИЗ, ЧТЕНИЕ РЫНКА sections (per Денис's spec)."""
    out = render_post(sample_ctx, template_key="ta_default")
    assert "ИДЕЯ" in out
    assert "АНАЛИЗ" in out
    assert "ЧТЕНИЕ РЫНКА" in out


def test_render_substitutes_setup_values(sample_ctx):
    """Confidence, entry, SL, TP must appear in output as numbers."""
    out = render_post(sample_ctx, template_key="ta_default")
    assert "72" in out                              # confidence
    assert "100" in out                             # entry
    assert "97" in out or "97.0" in out             # SL
    assert "104" in out or "104.0" in out           # TP


def test_render_has_direction_label(sample_ctx):
    """LONG/SHORT must appear (in Russian or English)."""
    out = render_post(sample_ctx, template_key="ta_default")
    assert "LONG" in out or "ЛОНГ" in out or "long" in out.lower()


def test_render_handles_short_direction(sample_ctx):
    sample_ctx.setup = Setup(
        direction="short", entry=109.0, stop_loss=112.0, take_profit=104.0,
        confidence=68, reasons=["RSI=78 — перекупленность"],
    )
    out = render_post(sample_ctx, template_key="ta_default")
    assert "SHORT" in out or "ШОРТ" in out or "short" in out.lower()


def test_render_includes_reasons(sample_ctx):
    """Reason strings from Setup are surfaced in АНАЛИЗ section."""
    out = render_post(sample_ctx, template_key="ta_default")
    assert "RSI" in out
    assert "тренд" in out.lower()


def test_unknown_template_falls_back(sample_ctx):
    """Asking for a template that doesn't exist yet → graceful fallback to default."""
    out = render_post(sample_ctx, template_key="nonexistent_template")
    assert "ИДЕЯ" in out  # default template was used


def test_template_handles_missing_optional_fields():
    """Some authors have no `bio` or `change_24h_pct` — must not crash."""
    sig = SignalContext(
        rsi=50.0, macd_line=0.0, macd_signal=0.0,
        trend="sideways", price=100.0, support=95.0, resistance=105.0, atr=2.0,
        regime="chop",
    )
    setup = Setup(direction="long", entry=100, stop_loss=98, take_profit=103,
                  confidence=20, reasons=["слабый сигнал"])
    ctx = PostContext(
        author_name="Test", symbol="X", interval="4h",
        price_now=100.0, change_24h_pct=None,
        signal_ctx=sig, setup=setup,
    )
    out = render_post(ctx, template_key="ta_default")
    assert len(out) > 100


def test_author_templates_registry_has_pilot():
    """The pilot author key is registered."""
    assert "ta_default" in AUTHOR_TEMPLATES
    assert "crypto_ml" in AUTHOR_TEMPLATES  # pilot integration


# ─── render_simple_post (for non-TA authors) ────────────────────────────────

def test_render_simple_post_includes_three_sections():
    """Astro/Currency/Index posts share the 3-section structure with TA posts."""
    from templates import render_simple_post
    out = render_simple_post(
        author_name="Astro Trader",
        header="BTC + космический фон",
        idea_block="Полнолуние ⇒ ожидайте разворота",
        analysis_block="Меркурий ретроградный — повышенный риск",
        market_block="Глобально: bear regime BTC, S&P -1.2%",
    )
    assert "ИДЕЯ" in out and "АНАЛИЗ" in out and "ЧТЕНИЕ РЫНКА" in out
    assert "Astro Trader" in out


def test_render_simple_post_renders_html_safely():
    """Output is HTML-formatted for aiogram parse_mode='HTML'."""
    from templates import render_simple_post
    out = render_simple_post(
        author_name="X", header="h", idea_block="i",
        analysis_block="a", market_block="m",
    )
    assert "<b>X</b>" in out  # bold author
    assert out.count("<b>") >= 4  # author + 3 section headers
