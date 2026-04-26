"""Per-author analyzers — turn live market data into copy-ready text.

Each analyzer returns a finished Telegram-ready string (HTML formatting).
The handler picks the right one by author.style + author.theme.

Design rules:
  - REAL data only — every number from a live API call
  - Graceful degradation — if a feed is unreachable, say so honestly
    instead of inventing numbers (Илья принципиально просил "никаких
    заглушек" — see project_v4_authors_real memory)
  - Each analyzer is a pure async function (author, market_session) → str
  - No DB writes; analyzer is a read-side projection
"""
from __future__ import annotations

import datetime as dt
import logging
import math
from typing import Any

import aiohttp
import ephem  # type: ignore[import-not-found]

from .db import Author, Db

log = logging.getLogger(__name__)

UA = {"User-Agent": "Mozilla/5.0 NodeVision/1.0"}
TIMEOUT = aiohttp.ClientTimeout(total=8)


# ─────────────────────────────────────────────────────────────────────────
#  Provider helpers (cached per-call session)
# ─────────────────────────────────────────────────────────────────────────


async def _get_json(session: aiohttp.ClientSession, url: str) -> Any | None:
    try:
        async with session.get(url, headers=UA) as r:
            if r.status != 200:
                log.warning("GET %s → %s", url, r.status)
                return None
            return await r.json()
    except Exception:
        log.exception("GET %s failed", url)
        return None


async def _binance_24h(session: aiohttp.ClientSession, symbol: str) -> dict[str, Any] | None:
    data = await _get_json(
        session, f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
    )
    if not data or "lastPrice" not in data:
        return None
    return {
        "price": float(data["lastPrice"]),
        "change_pct": float(data["priceChangePercent"]),
        "high": float(data["highPrice"]),
        "low": float(data["lowPrice"]),
        "vol": float(data["quoteVolume"]),
    }


async def _binance_klines(
    session: aiohttp.ClientSession, symbol: str, interval: str = "4h", limit: int = 50
) -> list[list[Any]] | None:
    data = await _get_json(
        session,
        f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}",
    )
    return data if isinstance(data, list) else None


async def _coingecko_global(session: aiohttp.ClientSession) -> dict[str, Any] | None:
    data = await _get_json(session, "https://api.coingecko.com/api/v3/global")
    if not data or "data" not in data:
        return None
    return data["data"]


async def _frankfurter(session: aiohttp.ClientSession, base: str, targets: str) -> dict | None:
    # Try the .dev domain (current canonical), fall back to .app legacy.
    for url in (
        f"https://api.frankfurter.dev/v1/latest?base={base}&symbols={targets}",
        f"https://api.frankfurter.app/latest?base={base}&symbols={targets}",
    ):
        d = await _get_json(session, url)
        if d and "rates" in d:
            return d
    return None


# ─────────────────────────────────────────────────────────────────────────
#  Technical helpers (no API — math on the kline series we already fetched)
# ─────────────────────────────────────────────────────────────────────────


def _closes(klines: list[list[Any]]) -> list[float]:
    # Binance kline: [open_time, open, high, low, close, volume, ...]
    return [float(k[4]) for k in klines]


def _rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    gains = []
    losses = []
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(d, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-d, 0.0)) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _macd_signal(closes: list[float]) -> tuple[float, float] | None:
    if len(closes) < 35:
        return None

    def ema(values: list[float], period: int) -> list[float]:
        k = 2.0 / (period + 1.0)
        out = [values[0]]
        for v in values[1:]:
            out.append(v * k + out[-1] * (1 - k))
        return out

    ema12 = ema(closes, 12)
    ema26 = ema(closes, 26)
    macd_line = [a - b for a, b in zip(ema12, ema26)]
    signal_line = ema(macd_line, 9)
    return macd_line[-1], signal_line[-1]


def _levels(klines: list[list[Any]], lookback: int = 50) -> tuple[float, float, float]:
    """Return (resistance, support, recent_pivot). Simple but real:
    resistance = max high, support = min low, pivot = avg of those + close.
    """
    take = klines[-lookback:]
    highs = [float(k[2]) for k in take]
    lows = [float(k[3]) for k in take]
    close = float(take[-1][4])
    return max(highs), min(lows), (max(highs) + min(lows) + close) / 3.0


def _trend(closes: list[float]) -> str:
    if len(closes) < 20:
        return "недостаточно данных"
    sma20 = sum(closes[-20:]) / 20
    last = closes[-1]
    delta_pct = (last - sma20) / sma20 * 100
    if delta_pct > 1.5:
        return f"восходящий (цена на {delta_pct:.1f}% выше SMA-20)"
    if delta_pct < -1.5:
        return f"нисходящий (цена на {abs(delta_pct):.1f}% ниже SMA-20)"
    return f"боковой (цена около SMA-20, отклонение {delta_pct:+.1f}%)"


def _fmt_price(v: float) -> str:
    if v >= 1000:
        return f"{v:,.2f}".replace(",", " ")
    if v >= 1:
        return f"{v:.4f}"
    return f"{v:.6f}"


# ─────────────────────────────────────────────────────────────────────────
#  Analyzers (one per author style)
# ─────────────────────────────────────────────────────────────────────────


async def analyze_crypto_technical(
    session: aiohttp.ClientSession, author: Author, db: Db
) -> str:
    """For TA Trader (BTC), Technical Crypto (ETH), Price Alerts."""
    sym = author.symbol
    ticker = await _binance_24h(session, sym)
    klines = await _binance_klines(session, sym, author.interval, 60)
    if ticker is None or klines is None:
        return _err_block(author, "Binance API временно недоступен")

    closes = _closes(klines)
    rsi = _rsi(closes)
    macd = _macd_signal(closes)
    res, sup, pivot = _levels(klines)
    trend = _trend(closes)

    rsi_word = (
        "перекупленность" if rsi and rsi > 70
        else "перепроданность" if rsi and rsi < 30
        else "нейтрально"
    )
    macd_word = (
        "бычье пересечение" if macd and macd[0] > macd[1]
        else "медвежье пересечение" if macd else "—"
    )

    return (
        f"<b>{author.name}</b>\n"
        f"<i>Сырьё для поста — упакуйте в свой стиль и опубликуйте.</i>\n"
        f"\n"
        f"📊 <b>{sym} {author.interval}</b>\n"
        f"Цена: <code>{_fmt_price(ticker['price'])}</code> "
        f"({ticker['change_pct']:+.2f}% за 24ч)\n"
        f"24ч диапазон: {_fmt_price(ticker['low'])} — {_fmt_price(ticker['high'])}\n"
        f"\n"
        f"📈 <b>Тренд:</b> {trend}\n"
        f"\n"
        f"🎯 <b>Уровни (последние 50 свечей):</b>\n"
        f"   Сопротивление: <code>{_fmt_price(res)}</code>\n"
        f"   Поддержка:     <code>{_fmt_price(sup)}</code>\n"
        f"   Pivot:         <code>{_fmt_price(pivot)}</code>\n"
        f"\n"
        f"🔢 <b>Индикаторы:</b>\n"
        f"   RSI(14): <b>{rsi:.1f}</b> — {rsi_word}\n"
        f"   MACD: {macd_word}"
        + (f" (line {macd[0]:+.2f}, signal {macd[1]:+.2f})" if macd else "")
        + f"\n"
        f"\n"
        f"<i>Данные: Binance (live). NodeVision · {dt.datetime.utcnow():%H:%M UTC}</i>"
    )


async def analyze_crypto_ml(
    session: aiohttp.ClientSession, author: Author, db: Db
) -> str:
    """For Crypto ML Trader, Premium BTC ML — uses our model's latest alert."""
    sym = author.symbol
    ticker = await _binance_24h(session, sym)
    alert = await db.latest_alert(author.strategy_id)

    parts = [f"<b>{author.name}</b>"]
    if author.is_premium:
        parts[0] = f"💎 {parts[0]}"
    parts.append(f"<i>{author.bio}</i>" if author.bio else "")
    parts.append("")

    parts.append(f"📊 <b>{sym} {author.interval}</b>")
    if ticker:
        parts.append(
            f"Цена: <code>{_fmt_price(ticker['price'])}</code> "
            f"({ticker['change_pct']:+.2f}% за 24ч)"
        )
    parts.append("")

    parts.append("🤖 <b>Сигнал ML модели:</b>")
    if alert is None:
        parts.append(
            f"Активного сигнала нет. Модель обучена на 8 годах истории "
            f"{sym}, ждём подтверждённого setup на {author.interval}."
        )
    else:
        direction = alert.direction.upper()
        parts.append(
            f"{'🟢 LONG' if direction == 'BUY' else '🔴 SHORT'} от "
            f"<code>{_fmt_price(alert.entry_price)}</code>"
        )
        parts.append(
            f"Стоп: <code>{_fmt_price(alert.stop_loss)}</code> · "
            f"Цель: <code>{_fmt_price(alert.take_profit)}</code>"
        )
        parts.append(f"Уверенность модели: {alert.confidence * 100 if alert.confidence < 1 else alert.confidence:.1f}%")
        parts.append(f"Сетап: <i>{alert.label or '—'}</i>")
        parts.append(f"Возраст сигнала: {(dt.datetime.now(alert.created_at.tzinfo) - alert.created_at).total_seconds() / 3600:.1f} ч")
    parts.append("")

    parts.append(f"<i>Данные: Binance + наша 4ч ML модель. Paper trade. Не финсовет.</i>")
    return "\n".join(p for p in parts if p is not None)


async def analyze_gold_news(
    session: aiohttp.ClientSession, author: Author, db: Db
) -> str:
    """Gold/Silver News — PAXG (tokenized gold 1:1) for price + GDELT for headlines."""
    paxg = await _binance_24h(session, "PAXGUSDT")

    parts = [f"<b>{author.name}</b>"]
    parts.append(f"<i>{author.bio}</i>" if author.bio else "")
    parts.append("")

    parts.append("🥇 <b>Золото (через PAXG/USDT, 1 PAXG = 1 oz)</b>")
    if paxg:
        parts.append(
            f"Цена: <code>${_fmt_price(paxg['price'])}</code> за oz "
            f"({paxg['change_pct']:+.2f}% за 24ч)"
        )
        parts.append(
            f"24ч диапазон: ${_fmt_price(paxg['low'])} — ${_fmt_price(paxg['high'])}"
        )
    else:
        parts.append("Цена временно недоступна (Binance API)")
    parts.append("")

    # GDELT news — best effort, often timeouts in some regions
    parts.append("📰 <b>Свежие новости по золоту:</b>")
    headlines = await _gdelt_headlines(session, "gold price", limit=3)
    if headlines:
        for h in headlines:
            parts.append(f"• <a href='{h['url']}'>{h['title'][:100]}</a>")
    else:
        parts.append(
            "Лента новостей временно недоступна. Возьмите заголовки с "
            "investing.com/gold или gold.org для копирайтинга."
        )
    parts.append("")

    parts.append(f"<i>NodeVision · gold_silver · {dt.datetime.utcnow():%H:%M UTC}</i>")
    return "\n".join(p for p in parts if p is not None)


async def analyze_currency(
    session: aiohttp.ClientSession,
    author: Author,
    db: Db,
    style: str,  # "news" or "fundamental"
) -> str:
    """Currency News / Currency Fundamental — Frankfurter rates + news/macro."""
    fx = await _frankfurter(session, "USD", "EUR,GBP,JPY,CHF,CAD,AUD")

    parts = [f"<b>{author.name}</b>"]
    parts.append(f"<i>{author.bio}</i>" if author.bio else "")
    parts.append("")

    parts.append("💱 <b>Текущие курсы (USD base, ECB):</b>")
    if fx:
        date = fx.get("date", "—")
        for ccy, rate in fx["rates"].items():
            parts.append(f"   USD/{ccy}: <code>{rate:.4f}</code>")
        parts.append(f"<i>Источник: ECB через Frankfurter, дата {date}</i>")
    else:
        parts.append("Frankfurter API временно недоступен — попробуйте через 1-2 минуты.")
    parts.append("")

    if style == "news":
        parts.append("📰 <b>Свежие FX новости:</b>")
        h = await _gdelt_headlines(session, "forex OR currency OR USD OR ECB OR Fed", limit=3)
        if h:
            for x in h:
                parts.append(f"• <a href='{x['url']}'>{x['title'][:100]}</a>")
        else:
            parts.append("Лента новостей временно недоступна.")
    else:
        parts.append("📈 <b>Фундаментальный фон:</b>")
        parts.append(
            "• Решения ФРС/ЕЦБ влияют на USD/EUR через дифференциал ставок\n"
            "• Релизы CPI/NFP — главные триггеры волатильности\n"
            "• Risk-on рынки → JPY слабеет, AUD/CAD укрепляются"
        )
        parts.append("")
        parts.append("<i>Календарь событий: investing.com/economic-calendar</i>")
    parts.append("")
    parts.append(f"<i>NodeVision · currencies · {dt.datetime.utcnow():%H:%M UTC}</i>")
    return "\n".join(p for p in parts if p is not None)


async def analyze_astro(
    session: aiohttp.ClientSession, author: Author, db: Db
) -> str:
    """Astro Trader — Moon phase, retrogrades, sun position via ephem."""
    now = dt.datetime.utcnow()
    moon = ephem.Moon(now)
    sun = ephem.Sun(now)
    mercury = ephem.Mercury(now)
    mars = ephem.Mars(now)
    venus = ephem.Venus(now)

    moon_pct = moon.moon_phase * 100
    if moon_pct > 95:
        moon_word = "🌕 Полнолуние"
    elif moon_pct < 5:
        moon_word = "🌑 Новолуние"
    elif moon_pct > 50:
        moon_word = f"🌔 Растущая Луна ({moon_pct:.0f}%)"
    else:
        moon_word = f"🌒 Убывающая Луна ({moon_pct:.0f}%)"

    # Next full / new moon
    next_full = ephem.next_full_moon(now)
    next_new = ephem.next_new_moon(now)

    # Mercury retrograde detection — compare today vs tomorrow heliocentric longitude
    mercury_tomorrow = ephem.Mercury(now + dt.timedelta(days=1))
    merc_retro = float(mercury_tomorrow.hlon) < float(mercury.hlon)

    # Sun sign (rough — by ecliptic longitude)
    sun_signs = [
        ("Овен", 0), ("Телец", 30), ("Близнецы", 60), ("Рак", 90),
        ("Лев", 120), ("Дева", 150), ("Весы", 180), ("Скорпион", 210),
        ("Стрелец", 240), ("Козерог", 270), ("Водолей", 300), ("Рыбы", 330),
    ]
    sun_lon_deg = math.degrees(float(sun.hlon)) % 360
    sun_sign = next(
        (name for name, start in reversed(sun_signs) if sun_lon_deg >= start),
        "—",
    )

    parts = [f"<b>{author.name}</b>"]
    parts.append(f"<i>{author.bio}</i>" if author.bio else "")
    parts.append("")
    parts.append(f"🌌 <b>Космический фон ({now:%d.%m.%Y %H:%M} UTC)</b>")
    parts.append("")
    parts.append(f"<b>Луна:</b> {moon_word}")
    parts.append(f"   Следующее полнолуние: {ephem.localtime(next_full):%d.%m %H:%M}")
    parts.append(f"   Следующее новолуние: {ephem.localtime(next_new):%d.%m %H:%M}")
    parts.append("")
    parts.append(f"<b>Меркурий:</b> {'🔄 Ретроградный' if merc_retro else '➡️ Директный'}")
    parts.append(f"<b>Солнце:</b> в знаке {sun_sign}")
    parts.append("")
    parts.append("📜 <b>Что писать в посте:</b>")
    if moon_pct > 95:
        parts.append("• Полнолуние = пик эмоций, частые развороты на рынках")
    elif moon_pct < 5:
        parts.append("• Новолуние = новые циклы, хорошее время для входов в тренд")
    elif moon_pct > 50:
        parts.append("• Растущая Луна = расширение объёмов, тренды продолжаются")
    else:
        parts.append("• Убывающая Луна = охлаждение, фиксация прибыли")
    if merc_retro:
        parts.append("• Меркурий ретроградный — повышенный риск ошибок исполнения")
    else:
        parts.append("• Меркурий директный — нормальный режим коммуникаций/исполнения")
    parts.append(f"• Солнце в {sun_sign} — секторная окраска месяца")
    parts.append("")
    parts.append(f"<i>Данные: Swiss Ephemeris (ephem). NodeVision · астро</i>")
    return "\n".join(parts)


async def analyze_index_or_oil(
    session: aiohttp.ClientSession, author: Author, db: Db
) -> str:
    """Index/Oil Fundamental — honest 'feed integration in progress' for now.

    Truthful placeholder: free-tier feeds for SPX/NASDAQ/Oil require API keys
    we haven't acquired yet. Returns useful structure (sector context +
    next steps) instead of fake numbers.
    """
    asset_word = {
        "indices": "индексы (SPX/NASDAQ/DJI)",
        "oil_gas": "нефть и газ (WTI, Brent, Henry Hub)",
    }.get(author.theme, "актив")

    return (
        f"<b>{author.name}</b>\n"
        f"<i>{author.bio}</i>\n"
        f"\n"
        f"📊 <b>{asset_word}</b>\n"
        f"\n"
        f"⚙️ <b>Интеграция в работе.</b>\n"
        f"Бесплатные источники цен для этого класса активов закрылись "
        f"(Yahoo Finance, Stooq) — подключаем платный фид (Twelve Data / "
        f"Finnhub Premium, ~$30/мес).\n"
        f"\n"
        f"📜 <b>Что писать сейчас (без бота):</b>\n"
        + (
            "• Влияние Fed на индексы через ставки + QT\n"
            "• Корреляция SPX с DXY (доллар вверх → SPX часто вниз)\n"
            "• Earnings season — главный драйвер волатильности\n"
            "• График: tradingview.com/chart/?symbol=SPX"
            if author.theme == "indices"
            else
            "• Геополитика (ОПЕК, Ближний Восток) → нефть\n"
            "• Складские запасы EIA по средам — главный триггер\n"
            "• Ралли доллара ослабляет нефть\n"
            "• График: tradingview.com/chart/?symbol=USOIL"
        )
        + f"\n\n<i>NodeVision · {author.theme}</i>"
    )


# ─────────────────────────────────────────────────────────────────────────
#  News provider helpers
# ─────────────────────────────────────────────────────────────────────────


async def _gdelt_headlines(
    session: aiohttp.ClientSession, query: str, limit: int = 3
) -> list[dict[str, str]]:
    """GDELT 2.0 doc API — free, no key, sometimes rate-limited."""
    from urllib.parse import quote

    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={quote(query)}&mode=ArtList&maxrecords={limit}"
        "&format=json&sort=DateDesc"
    )
    data = await _get_json(session, url)
    if not data or not isinstance(data, dict):
        return []
    arts = data.get("articles", [])
    return [
        {"title": a.get("title", "—"), "url": a.get("url", "")}
        for a in arts
        if a.get("title")
    ]


# ─────────────────────────────────────────────────────────────────────────
#  Public dispatch — handlers.py calls this
# ─────────────────────────────────────────────────────────────────────────


async def render_for_author(author: Author, db: Db) -> str:
    """Pick the right analyzer by author.theme + author.style."""
    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        try:
            # Crypto ML
            if author.style == "ml" and author.theme == "crypto":
                return await analyze_crypto_ml(session, author, db)

            # Crypto technical / levels
            if author.theme == "crypto":
                return await analyze_crypto_technical(session, author, db)

            # Gold / Silver
            if author.theme == "gold_silver":
                return await analyze_gold_news(session, author, db)

            # Currency news / fundamental
            if author.theme == "currencies":
                return await analyze_currency(session, author, db, author.style or "news")

            # Astro
            if author.style == "astro":
                return await analyze_astro(session, author, db)

            # Indices / Oil — honest placeholder
            if author.theme in ("indices", "oil_gas"):
                return await analyze_index_or_oil(session, author, db)

            # Fallback — should never trigger if seed is correct
            return _err_block(author, f"unknown analyzer for theme={author.theme} style={author.style}")
        except Exception:
            log.exception("analyzer crashed for author %s", author.slug)
            return _err_block(author, "сценарий упал — посмотрите логи")


def _err_block(author: Author, reason: str) -> str:
    return (
        f"<b>{author.name}</b>\n\n"
        f"⚠️ {reason}\n\n"
        f"<i>NodeVision · {author.theme}</i>"
    )
