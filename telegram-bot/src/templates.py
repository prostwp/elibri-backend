"""Post templates — every author's bot output goes through here.

Section structure (per Денис's spec):
    📊 [header: символ, TF, цена]
    🎯 ИДЕЯ — entry/SL/TP/direction/Confidence
    📈 АНАЛИЗ — почему такая идея (reasons из Setup)
    🌐 ЧТЕНИЕ РЫНКА — текущий контекст (regime, ключевые драйверы)

Templates are simple Python f-string-like strings with `{var}` placeholders
filled by `render_post`. Per-author style differs by KEY (more concise for
индикаторщик, more verbose for macromind), but the THREE-BLOCK structure
is invariant.

Per `feedback_ml_pipeline_safety.md`: tests live in tests/test_templates.py
and were written first.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    # Package import path (used by analyzers.py at runtime).
    from .idea_engine import Setup, SignalContext
except ImportError:
    # Test import path: tests/ injects src/ into sys.path.
    from idea_engine import Setup, SignalContext  # type: ignore[no-redef]


@dataclass
class PostContext:
    """Bundle of everything a template might want."""
    author_name: str
    symbol: str
    interval: str
    price_now: float
    change_24h_pct: Optional[float]
    signal_ctx: SignalContext
    setup: Setup
    bio: Optional[str] = None    # for authors that show their tagline
    market_note: Optional[str] = None  # injected by analyzer (e.g. macro context)
    # Bug #1 fix: human-readable trend description (separate from
    # signal_ctx.trend which is the literal `uptrend/downtrend/sideways`
    # used by idea_engine for equality checks).
    trend_human: Optional[str] = None


# ─── Direction label helpers ────────────────────────────────────────────────

def _dir_emoji(direction: str) -> str:
    return "🟢 LONG" if direction == "long" else "🔴 SHORT"


def _confidence_band(c: int) -> str:
    """Verbal qualifier for confidence number — used in opening line."""
    if c >= 70:
        return "сильный setup"
    if c >= 50:
        return "умеренный setup"
    if c >= 30:
        return "слабый setup"
    return "очень слабый setup"


def _trend_word_ru(trend: str) -> str:
    return {"uptrend": "восходящий", "downtrend": "нисходящий",
            "sideways": "боковой"}.get(trend, trend)


def _regime_word_ru(regime: str) -> str:
    return {"bull": "бычий", "bear": "медвежий", "chop": "боковик",
            "warm": "не определён (мало данных)"}.get(regime, regime)


# ─── Templates per author key ───────────────────────────────────────────────

DEFAULT_TEMPLATE = """\
<b>{author_name}</b>
<i>{symbol} {interval} · цена <code>{price_str}</code>{change_str}</i>

🎯 <b>ИДЕЯ</b>
{dir_label} от <code>{entry_str}</code>
SL: <code>{sl_str}</code> · TP: <code>{tp_str}</code>
Confidence: <b>{confidence}%</b> ({confidence_band})

📈 <b>АНАЛИЗ</b>
{reasons_block}
RSI(14): <b>{rsi:.0f}</b> · тренд: <i>{trend_ru}</i> · MACD: {macd_word}
Уровни: поддержка <code>{support_str}</code>, сопротивление <code>{resistance_str}</code>

🌐 <b>ЧТЕНИЕ РЫНКА</b>
Глобальный режим: <b>{regime_ru}</b>. {market_note}
ATR(14): <code>{atr_str}</code> — нормальная волатильность для данного TF.

<i>NodeVision · {timestamp}</i>"""


CRYPTO_ML_TEMPLATE = """\
<b>{author_name}</b>
<i>{symbol} {interval} · цена <code>{price_str}</code>{change_str}</i>

🎯 <b>ИДЕЯ</b>
{dir_label} от <code>{entry_str}</code>
SL: <code>{sl_str}</code> · TP: <code>{tp_str}</code> (R/R {rr})
Confidence: <b>{confidence}%</b> ({confidence_band})

📈 <b>АНАЛИЗ</b>
{reasons_block}
RSI(14)=<b>{rsi:.0f}</b>, MACD: {macd_word}, тренд: <i>{trend_ru}</i>.
Поддержка <code>{support_str}</code> · сопротивление <code>{resistance_str}</code>.

🌐 <b>ЧТЕНИЕ РЫНКА</b>
Глобальный режим — <b>{regime_ru}</b>.
{market_note}

<i>NodeVision · {timestamp}</i>"""


PRICE_ALERTS_TEMPLATE = """\
<b>{author_name}</b>
<i>{symbol} {interval} · цена <code>{price_str}</code>{change_str}</i>

🔔 <b>УРОВНИ И БРЕЙКАУТ</b>
Поддержка: <code>{support_str}</code>
Сопротивление: <code>{resistance_str}</code>
Pivot: цена <i>{trend_ru}</i>, ATR(14) <code>{atr_str}</code>

🎯 <b>ИДЕЯ</b>
{dir_label} от <code>{entry_str}</code> · SL <code>{sl_str}</code> · TP <code>{tp_str}</code>
Confidence: <b>{confidence}%</b> ({confidence_band})
{reasons_block}

🌐 <b>ЧТЕНИЕ РЫНКА</b>
Глобальный режим — <b>{regime_ru}</b>.
{market_note}

<i>NodeVision · {timestamp}</i>"""


GOLD_NEWS_TEMPLATE = """\
<b>{author_name}</b>
<i>Золото через PAXG/USDT (1 PAXG = 1 oz) · цена <code>${price_str}</code>{change_str}</i>

🎯 <b>ИДЕЯ</b>
{dir_label} от <code>{entry_str}</code>
SL: <code>{sl_str}</code> · TP: <code>{tp_str}</code> (R/R {rr})
Confidence: <b>{confidence}%</b> ({confidence_band})

📈 <b>АНАЛИЗ</b>
{reasons_block}
RSI(14)=<b>{rsi:.0f}</b>, тренд: <i>{trend_ru}</i>, MACD: {macd_word}.
Поддержка <code>${support_str}</code> · сопротивление <code>${resistance_str}</code>.

🌐 <b>ЧТЕНИЕ РЫНКА</b>
Глобальный режим золота — <b>{regime_ru}</b>.
{market_note}

<i>NodeVision · gold_silver · {timestamp}</i>"""


AUTHOR_TEMPLATES: dict[str, str] = {
    "ta_default": DEFAULT_TEMPLATE,
    "crypto_ml": CRYPTO_ML_TEMPLATE,
    "price_alerts": PRICE_ALERTS_TEMPLATE,
    "gold_news": GOLD_NEWS_TEMPLATE,
}


# ─── Standalone post renderer for analyzers WITHOUT TA pipeline ─────────────
# (Astro / Currency / Index / Oil — these don't have a Setup, so we expose a
# simpler render function that just bakes the 3-section layout from raw
# string blocks the analyzer prepared.)


def render_simple_post(
    author_name: str,
    header: str,           # subtitle line under author name
    idea_block: str,       # what to write in 🎯 ИДЕЯ section
    analysis_block: str,   # 📈 АНАЛИЗ
    market_block: str,     # 🌐 ЧТЕНИЕ РЫНКА
    footer_tag: str = "NodeVision",
) -> str:
    """Three-section layout for non-TA authors (astro / currency / placeholders).

    Same outer shape as TA posts so subscribers see consistent structure
    regardless of which author button they tapped.
    """
    import datetime as _dt
    return (
        f"<b>{author_name}</b>\n"
        f"<i>{header}</i>\n\n"
        f"🎯 <b>ИДЕЯ</b>\n{idea_block}\n\n"
        f"📈 <b>АНАЛИЗ</b>\n{analysis_block}\n\n"
        f"🌐 <b>ЧТЕНИЕ РЫНКА</b>\n{market_block}\n\n"
        f"<i>{footer_tag} · {_dt.datetime.utcnow():%H:%M UTC}</i>"
    )


# ─── Renderer ────────────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    """Compact price formatting: 2 decimals over $1, 6 decimals below."""
    if abs(v) >= 1:
        return f"{v:,.2f}".replace(",", " ")
    return f"{v:.6f}"


def render_post(ctx: PostContext, template_key: str = "ta_default") -> str:
    """Render full post (3 sections). Unknown template_key → falls back to default."""
    import datetime as _dt

    tpl = AUTHOR_TEMPLATES.get(template_key, DEFAULT_TEMPLATE)
    setup = ctx.setup
    sig = ctx.signal_ctx

    macd_word = ("бычье пересечение" if sig.macd_line > sig.macd_signal
                 else "медвежье пересечение" if sig.macd_line < sig.macd_signal
                 else "—")
    change_str = f" ({ctx.change_24h_pct:+.2f}% за 24ч)" if ctx.change_24h_pct is not None else ""
    reasons_block = "\n".join(f"  • {r}" for r in setup.reasons) if setup.reasons else "  • —"
    rr_val = setup.risk_reward
    rr_str = f"{rr_val:.2f}" if rr_val > 0 else "—"

    # Prefer human-readable trend description from analyzer; fall back to
    # short Russian word if analyzer didn't supply it.
    trend_text = ctx.trend_human or _trend_word_ru(sig.trend)

    return tpl.format(
        author_name=ctx.author_name,
        symbol=ctx.symbol,
        interval=ctx.interval,
        price_str=_fmt(ctx.price_now),
        change_str=change_str,
        dir_label=_dir_emoji(setup.direction),
        entry_str=_fmt(setup.entry),
        sl_str=_fmt(setup.stop_loss),
        tp_str=_fmt(setup.take_profit),
        confidence=setup.confidence,
        confidence_band=_confidence_band(setup.confidence),
        reasons_block=reasons_block,
        rsi=sig.rsi,
        macd_word=macd_word,
        trend_ru=trend_text,
        regime_ru=_regime_word_ru(sig.regime),
        support_str=_fmt(sig.support),
        resistance_str=_fmt(sig.resistance),
        atr_str=_fmt(sig.atr),
        rr=rr_str,
        market_note=ctx.market_note or "Существенных макро-триггеров на горизонте 12-24ч не зафиксировано.",
        timestamp=_dt.datetime.utcnow().strftime("%H:%M UTC"),
    )
