"""Default formatter — neutral, indicator-style. Used until per-author formatters land."""
from typing import Any


def format_signal(payload: dict[str, Any]) -> str:
    """
    Render a Redis Stream entry into a Telegram-ready message.

    Expected payload keys (Redis stream values arrive as strings):
        symbol, interval, direction, entry, stop_loss, take_profit, confidence, label, bar_time
    """
    symbol = payload.get("symbol", "?")
    interval = payload.get("interval", "?")
    direction = payload.get("direction", "?").upper()
    entry = _fmt_price(payload.get("entry"))
    sl = _fmt_price(payload.get("stop_loss"))
    tp = _fmt_price(payload.get("take_profit"))
    confidence_pct = _fmt_pct(payload.get("confidence"))
    label = payload.get("label", "")

    arrow = "🔻" if direction == "SELL" else ("🚀" if direction == "BUY" else "•")

    risk_reward = _risk_reward(payload)
    rr_line = f"R/R: {risk_reward}\n" if risk_reward else ""
    label_line = f"Setup: {label}\n" if label else ""

    return (
        f"{arrow} <b>{symbol} {interval} {direction}</b>\n"
        f"\n"
        f"Entry: <code>{entry}</code>\n"
        f"Stop:  <code>{sl}</code>\n"
        f"Target: <code>{tp}</code>\n"
        f"\n"
        f"{rr_line}"
        f"Confidence: {confidence_pct}\n"
        f"{label_line}"
        f"\n"
        f"<i>NodeVision · paper trade · not financial advice</i>"
    )


def _fmt_price(raw: Any) -> str:
    if raw is None:
        return "—"
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return str(raw)
    if v >= 1000:
        return f"{v:,.2f}".replace(",", " ")
    return f"{v:.4f}"


def _fmt_pct(raw: Any) -> str:
    if raw is None:
        return "—"
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return str(raw)
    # Confidence stored either as 0..1 or 0..100; normalize to percent.
    if v <= 1.0:
        v *= 100
    return f"{v:.1f}%"


def _risk_reward(payload: dict[str, Any]) -> str | None:
    try:
        entry = float(payload["entry"])
        sl = float(payload["stop_loss"])
        tp = float(payload["take_profit"])
    except (KeyError, TypeError, ValueError):
        return None
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk <= 0:
        return None
    return f"{reward / risk:.1f}:1"
