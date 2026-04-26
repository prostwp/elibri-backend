"""3-in-1 author response renderer: header + analytics + idea + trade.

No LLM, no fake data — every line comes from real DB / Binance ticker.
If a piece is unavailable (e.g. no alert ever fired), we say so honestly
instead of inventing a synthetic signal.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from .db import Alert, Author


def render_author_response(
    author: Author,
    alert: Alert | None,
    market: dict[str, Any] | None,
) -> str:
    parts: list[str] = []

    # ── Header ─────────────────────────────────────────────────
    premium_badge = "💎 " if author.is_premium else ""
    parts.append(f"<b>{premium_badge}{author.name}</b>")
    if author.bio:
        parts.append(f"<i>{author.bio}</i>")
    parts.append(f"")

    # ── 1. Аналитика рынка ─────────────────────────────────────
    parts.append("📊 <b>Аналитика</b>")
    if market is not None:
        price = market["price"]
        change = market["change_pct_24h"]
        change_arrow = "▲" if change >= 0 else "▼"
        parts.append(
            f"{author.symbol}: <code>{_fmt_price(price)}</code> "
            f"{change_arrow} {abs(change):.2f}% за 24ч"
        )
        parts.append(
            f"24ч диапазон: {_fmt_price(market['low_24h'])} — {_fmt_price(market['high_24h'])}"
        )
    else:
        parts.append(
            f"{author.symbol}: данные временно недоступны (Binance API не отвечает)"
        )
    parts.append(f"Таймфрейм: {author.interval}")
    parts.append("")

    # ── 2. Торговая идея ───────────────────────────────────────
    parts.append("💡 <b>Торговая идея</b>")
    if alert is None:
        parts.append(
            f"Нет активного сигнала. {_style_descriptor(author.style)} "
            f"ожидает подтверждения по {author.theme}."
        )
    else:
        direction = alert.direction.upper()
        side_word = "покупка" if direction == "BUY" else "продажа"
        rr = _risk_reward(alert)
        rr_line = f", R/R {rr}" if rr else ""
        parts.append(
            f"{_direction_arrow(direction)} <b>{side_word}</b> от "
            f"<code>{_fmt_price(alert.entry_price)}</code>"
        )
        parts.append(
            f"Стоп: <code>{_fmt_price(alert.stop_loss)}</code> · "
            f"Цель: <code>{_fmt_price(alert.take_profit)}</code>"
        )
        parts.append(
            f"Уверенность: <b>{_fmt_confidence(alert.confidence)}</b>"
            f"{rr_line} · Сетап: <i>{alert.label or '—'}</i>"
        )
    parts.append("")

    # ── 3. Текущая сделка ──────────────────────────────────────
    parts.append("📝 <b>Текущая сделка</b>")
    if alert is None:
        parts.append("Активной сделки нет — ждём первого сигнала.")
    else:
        # Compare current price to alert entry to show how the trade is going
        if market is not None and alert.entry_price > 0:
            current = market["price"]
            if alert.direction == "buy":
                pnl_pct = (current - alert.entry_price) / alert.entry_price * 100
            else:
                pnl_pct = (alert.entry_price - current) / alert.entry_price * 100
            status = (
                f"<b>+{pnl_pct:.2f}%</b> в плюсе"
                if pnl_pct > 0
                else f"<b>{pnl_pct:.2f}%</b> в минусе"
            )
            # Detect SL/TP hit (best-effort — runner is the source of truth)
            if alert.direction == "buy":
                if current <= alert.stop_loss:
                    status += " · стоп пробит"
                elif current >= alert.take_profit:
                    status += " · цель достигнута"
            else:
                if current >= alert.stop_loss:
                    status += " · стоп пробит"
                elif current <= alert.take_profit:
                    status += " · цель достигнута"
            parts.append(status)
            parts.append(
                f"Открыта {_fmt_age(alert.created_at)} назад "
                f"({alert.created_at.astimezone().strftime('%d.%m %H:%M')})"
            )
        else:
            parts.append(f"Сигнал от {alert.created_at.astimezone().strftime('%d.%m %H:%M')}")

    parts.append("")
    parts.append(
        f"<i>NodeVision · {author.theme} · {author.risk_tier} · paper trade · не финсовет</i>"
    )

    return "\n".join(parts)


def render_premium_paywall(author: Author) -> str:
    return (
        f"💎 <b>{author.name}</b>\n\n"
        f"Этот сценарий доступен в <b>премиум-подписке</b>.\n\n"
        f"Что внутри:\n"
        f"• ML-сигналы на {author.symbol} {author.interval}\n"
        f"• Aggressive-уровень риска (более частые сделки)\n"
        f"• Уведомления в Telegram при каждом сигнале\n"
        f"• Доступ к историческим данным проверки модели\n\n"
        f"<b>30 USD/мес</b>\n\n"
        f"Запросить подписку: напишите @{_admin_handle()} (пока ручная активация)."
    )


# ───── helpers ────────────────────────────────────────────────────

def _fmt_price(v: float) -> str:
    if v >= 1000:
        return f"{v:,.2f}".replace(",", " ")
    if v >= 1:
        return f"{v:.4f}"
    return f"{v:.6f}"


def _fmt_confidence(v: float) -> str:
    if v <= 1.0:
        v *= 100
    return f"{v:.1f}%"


def _direction_arrow(direction: str) -> str:
    return "🟢" if direction == "BUY" else "🔴"


def _risk_reward(alert: Alert) -> str | None:
    if alert.entry_price <= 0:
        return None
    risk = abs(alert.entry_price - alert.stop_loss)
    reward = abs(alert.take_profit - alert.entry_price)
    if risk <= 0:
        return None
    return f"{reward / risk:.1f}:1"


def _fmt_age(when: datetime) -> str:
    delta = datetime.now(when.tzinfo) - when
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{seconds}с"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}м"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}ч"
    return f"{hours // 24}д"


def _style_descriptor(style: str) -> str:
    return {
        "technical": "Технический анализ",
        "ml": "ML-сценарий",
        "news": "Новостной трейдер",
        "fundamental": "Фундаментальный анализ",
        "levels": "Сценарий уровней",
        "astro": "Астро-трейдер",
    }.get(style, "Сценарий")


def _admin_handle() -> str:
    # Hardcoded for now — move to settings when we add multi-admin support.
    return "denis_karnachev"
