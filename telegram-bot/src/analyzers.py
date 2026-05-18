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
    """Bug #6 fix: validate ALL required keys + catch parse errors so
    partial payloads (mini-ticker, transient Binance shape changes) return
    None cleanly instead of crashing analyzers with KeyError/ValueError.
    """
    data = await _get_json(
        session, f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
    )
    required = ("lastPrice", "priceChangePercent", "highPrice", "lowPrice", "quoteVolume")
    if not data or not all(k in data for k in required):
        return None
    try:
        return {
            "price": float(data["lastPrice"]),
            "change_pct": float(data["priceChangePercent"]),
            "high": float(data["highPrice"]),
            "low": float(data["lowPrice"]),
            "vol": float(data["quoteVolume"]),
        }
    except (TypeError, ValueError):
        log.warning("ticker/24hr for %s returned unparseable values", symbol)
        return None


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


def _trend(closes: list[float]) -> tuple[str, str]:
    """Returns (literal, human_label). The literal feeds SignalContext.trend
    (must match the Literal type checked in idea_engine); the human_label is
    rendered in templates only.

    Bug #1 fix (review 2026-04-30): prior version returned only the human
    string and SignalContext.trend silently mismatched against
    Literal["uptrend","downtrend","sideways"], dropping confidence by 5 on
    every post and disabling trend-alignment scoring entirely.
    """
    if len(closes) < 20:
        return "sideways", "недостаточно данных"
    sma20 = sum(closes[-20:]) / 20
    last = closes[-1]
    delta_pct = (last - sma20) / sma20 * 100
    if delta_pct > 1.5:
        return "uptrend", f"восходящий (цена на {delta_pct:.1f}% выше SMA-20)"
    if delta_pct < -1.5:
        return "downtrend", f"нисходящий (цена на {abs(delta_pct):.1f}% ниже SMA-20)"
    return "sideways", f"боковой (цена около SMA-20, отклонение {delta_pct:+.1f}%)"


def _fmt_price(v: float) -> str:
    if v >= 1000:
        return f"{v:,.2f}".replace(",", " ")
    if v >= 1:
        return f"{v:.4f}"
    return f"{v:.6f}"


# ─────────────────────────────────────────────────────────────────────────
#  Analyzers (one per author style)
# ─────────────────────────────────────────────────────────────────────────


async def _run_idea_pipeline(
    session: aiohttp.ClientSession,
    author: Author,
    template_key: str,
    pair_override: str | None = None,
    extra_market_note: str | None = None,
) -> str:
    """Shared pipeline for analyzers that share the TA → idea → pattern-match flow.

    Used by: crypto_technical (TA BTC, TA ETH, Price Alerts), crypto_ml,
    gold_news. Each caller passes its template_key (per-author voice/format)
    plus an optional override for the trading pair (gold uses PAXG even
    though author.symbol may be something else) and an optional extra
    market_note (e.g. news bullets for gold) prepended to pattern-match text.

    Returns finished HTML post or `_err_block` on data outage.
    """
    from .idea_engine import SignalContext, build_setup, infer_regime
    from .pattern_match import compute_query_features, enrich_with_pattern
    from .templates import PostContext, render_post

    sym = pair_override or author.symbol
    ticker = await _binance_24h(session, sym)
    klines = await _binance_klines(session, sym, author.interval, 60)
    klines_1d = await _binance_klines(session, sym, "1d", 300)

    # Bug #5 fix: empty list (Binance halt / rate-limit) passes `is None`
    # check but breaks _levels with IndexError. Use truthiness + length floor.
    if ticker is None or not klines:
        return _err_block(author, "Binance API временно недоступен")
    if len(klines) < 20:
        return _err_block(author, "Недостаточно баров для анализа")

    closes = _closes(klines)
    rsi = _rsi(closes)
    macd = _macd_signal(closes)
    res, sup, _pivot = _levels(klines)
    trend_lit, trend_human = _trend(closes)

    # Bug #4 fix: replicate `pandas .ewm(alpha=1/period, adjust=False).mean()`
    # used in build script (Wilder smoothing). Recurrence (matches pandas
    # adjust=False exactly):
    #     s[0] = x[0]
    #     s[i] = alpha * x[i] + (1 - alpha) * s[i-1]   where alpha = 1/period
    # We seed from the first available TR (same as pandas), accept some
    # initial drift, but with 60 input bars the seed influence on the last
    # bar is < 1.5% (decay = (1-1/14)^58 ≈ 0.014).
    atr = 0.0
    period = 14
    n = len(klines)
    if n >= 2:
        alpha = 1.0 / period
        trs: list[float] = []
        for i in range(1, n):
            high_i = float(klines[i][2])
            low_i = float(klines[i][3])
            prev_close = float(klines[i - 1][4])
            tr = max(high_i - low_i, abs(high_i - prev_close), abs(low_i - prev_close))
            trs.append(tr)
        smoothed = trs[0]
        for tr in trs[1:]:
            smoothed = alpha * tr + (1 - alpha) * smoothed
        atr = smoothed

    closes_1d = _closes(klines_1d) if klines_1d else []
    regime, regime_slope = infer_regime(closes_1d)

    sig = SignalContext(
        # Bug #3 fix: explicit None-check, not truthiness — preserves rsi=0.0
        # (max oversold) and macd_line=0.0 (real EMA crossover).
        rsi=float(rsi) if rsi is not None else 50.0,
        macd_line=float(macd[0]) if macd is not None else 0.0,
        macd_signal=float(macd[1]) if macd is not None else 0.0,
        trend=trend_lit,
        price=float(ticker["price"]),
        support=float(sup),
        resistance=float(res),
        atr=float(atr),
        regime=regime,
    )
    setup = build_setup(sig)

    pattern_text: str | None = None
    try:
        query_vec = compute_query_features(
            rsi=sig.rsi, macd_line=sig.macd_line, macd_signal=sig.macd_signal,
            atr=sig.atr, price=sig.price,
            support_50=sig.support, resistance_50=sig.resistance,
            recent_closes=closes[-6:] if len(closes) >= 6 else closes,
            # Bug #2 fix: pass real slope from infer_regime — not hardcoded 0.0.
            regime_score=regime_slope,
        )
        pattern_text = enrich_with_pattern(
            symbol=sym, interval=author.interval,
            query_vec=query_vec, direction=setup.direction, top_k=100,
        )
    except Exception:
        log.exception("pattern_match failed for %s %s", sym, author.interval)
        pattern_text = None

    market_note_parts = [p for p in (extra_market_note, pattern_text) if p]
    market_note = "\n".join(market_note_parts) if market_note_parts else None

    ctx = PostContext(
        author_name=author.name, symbol=sym, interval=author.interval,
        price_now=float(ticker["price"]),
        change_24h_pct=float(ticker["change_pct"]),
        signal_ctx=sig, setup=setup, bio=author.bio,
        market_note=market_note,
        trend_human=trend_human,
    )
    return render_post(ctx, template_key=template_key)


async def analyze_crypto_technical(
    session: aiohttp.ClientSession, author: Author, db: Db
) -> str:
    """For TA Trader (BTC), Technical Crypto (ETH), Price Alerts.

    Routes to per-author template:
      author.style="technical" → ta_default (TA Trader / Technical Crypto style)
      author.style="levels"    → price_alerts (focus on key-level break)
    """
    template_key = "price_alerts" if author.style == "levels" else "ta_default"
    return await _run_idea_pipeline(session, author, template_key=template_key)

async def analyze_crypto_ml(
    session: aiohttp.ClientSession, author: Author, db: Db
) -> str:
    """Pilot author — uses idea_engine + templates + HNSW pattern-match.

    Was the entry-point for the dropped 4h supervised model; now serves as
    pattern-based analyst (rule-based idea + historical context). The
    pipeline body is shared with TA Trader / Gold News via
    `_run_idea_pipeline`.
    """
    return await _run_idea_pipeline(session, author, template_key="crypto_ml")


async def analyze_gold_news(
    session: aiohttp.ClientSession, author: Author, db: Db
) -> str:
    """Gold/Silver News — PAXG (tokenized gold 1:1) + GDELT headlines.

    Uses the shared idea pipeline (PAXG OHLCV via Binance), then prepends
    fresh news headlines as the market_note. No HNSW yet (PAXG parquet
    not built); pattern-match silently skipped → fallback to rule-based.
    """
    headlines = await _gdelt_headlines(session, "gold price", limit=3)
    if headlines:
        news_lines = ["<b>Новости по золоту:</b>"] + [
            f"• <a href='{h['url']}'>{h['title'][:100]}</a>"
            for h in headlines
        ]
        news_note = "\n".join(news_lines)
    else:
        news_note = (
            "Лента новостей временно недоступна. Возьмите заголовки "
            "с investing.com/gold или gold.org."
        )

    return await _run_idea_pipeline(
        session, author,
        template_key="gold_news",
        pair_override="PAXGUSDT",
        extra_market_note=news_note,
    )


async def analyze_currency(
    session: aiohttp.ClientSession,
    author: Author,
    db: Db,
    style: str,  # "news" or "fundamental"
) -> str:
    """Currency News / Fundamental — Frankfurter rates + news/macro context.

    Uses `render_simple_post` (no TA pipeline) — FX historical parquet not
    in this repo, so no HNSW. Idea block is rate-direction directional cue,
    analysis is the rate table + news, market_block is macro context.
    """
    from .templates import render_simple_post

    fx = await _frankfurter(session, "USD", "EUR,GBP,JPY,CHF,CAD,AUD")

    if fx:
        rate_eur = float(fx["rates"].get("EUR", 0)) or None
        rate_lines = [f"USD/{ccy}: <code>{rate:.4f}</code>"
                      for ccy, rate in fx["rates"].items()]
        rates_block = "\n".join(rate_lines)
        date = fx.get("date", "—")
        idea_pair = "EUR/USD"
        idea_dir = (
            "Долгосрочный диапазон 1.05–1.15 — следите за пробоями. "
            "Confidence: <b>низкий</b> (rate-only, без TA)."
        )
        analysis = (
            f"<b>Курсы (USD base, ECB {date}):</b>\n{rates_block}"
        )
    else:
        idea_pair = "EUR/USD"
        idea_dir = "Frankfurter API временно недоступен — данные подгрузятся через 1-2 минуты."
        analysis = "—"

    if style == "news":
        h = await _gdelt_headlines(
            session, "forex OR currency OR USD OR ECB OR Fed", limit=3
        )
        if h:
            news_lines = [f"• <a href='{x['url']}'>{x['title'][:100]}</a>" for x in h]
            market_block = "<b>Свежие FX новости:</b>\n" + "\n".join(news_lines)
        else:
            market_block = (
                "Лента новостей временно недоступна. "
                "Возьмите заголовки с investing.com/economic-calendar."
            )
    else:  # fundamental
        market_block = (
            "<b>Фундаментальный фон:</b>\n"
            "• Решения ФРС/ЕЦБ влияют на USD/EUR через дифференциал ставок\n"
            "• Релизы CPI/NFP — главные триггеры волатильности\n"
            "• Risk-on рынки → JPY слабеет, AUD/CAD укрепляются\n"
            "<i>Календарь: investing.com/economic-calendar</i>"
        )

    return render_simple_post(
        author_name=author.name,
        header=f"{idea_pair} · валютный обзор",
        idea_block=idea_dir,
        analysis_block=analysis,
        market_block=market_block,
        footer_tag="NodeVision · currencies",
    )


async def analyze_astro(
    session: aiohttp.ClientSession, author: Author, db: Db
) -> str:
    """Astro Trader — Moon phase, Mercury retrograde, Sun sign via ephem.

    Renders through `render_simple_post` (no TA pipeline) so the post shape
    matches all other authors. Idea block = directional cue from moon phase;
    analysis = ephemeris details; market_block = trading lore for the post.
    """
    from .templates import render_simple_post

    now = dt.datetime.utcnow()
    moon = ephem.Moon(now)
    sun = ephem.Sun(now)
    mercury = ephem.Mercury(now)
    mercury_tomorrow = ephem.Mercury(now + dt.timedelta(days=1))

    moon_pct = moon.moon_phase * 100
    if moon_pct > 95:
        moon_word = "🌕 Полнолуние"
        moon_cue = "Полнолуние ⇒ пик эмоций, ожидайте разворотов"
        moon_dir = "флэт / ловите развороты"
    elif moon_pct < 5:
        moon_word = "🌑 Новолуние"
        moon_cue = "Новолуние ⇒ новые циклы, хорошее время для входов в тренд"
        moon_dir = "тренд-фоллоу со входом по импульсу"
    elif moon_pct > 50:
        moon_word = f"🌔 Растущая Луна ({moon_pct:.0f}%)"
        moon_cue = "Растущая Луна ⇒ расширение объёмов, тренды продолжаются"
        moon_dir = "long-bias на тренде"
    else:
        moon_word = f"🌒 Убывающая Луна ({moon_pct:.0f}%)"
        moon_cue = "Убывающая Луна ⇒ охлаждение, фиксация прибыли"
        moon_dir = "осторожный shorting / фиксация long-позиций"

    next_full = ephem.next_full_moon(now)
    next_new = ephem.next_new_moon(now)

    merc_retro = float(mercury_tomorrow.hlon) < float(mercury.hlon)
    merc_word = "🔄 Ретроградный" if merc_retro else "➡️ Директный"

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

    idea_block = (
        f"{moon_word} ⇒ {moon_dir}\n"
        f"Confidence: <b>низкий</b> (астро — статистический фон, не сигнал)."
    )
    analysis_block = (
        f"<b>Луна:</b> {moon_word}\n"
        f"  Следующее полнолуние: {ephem.localtime(next_full):%d.%m %H:%M}\n"
        f"  Следующее новолуние: {ephem.localtime(next_new):%d.%m %H:%M}\n"
        f"<b>Меркурий:</b> {merc_word}\n"
        f"<b>Солнце:</b> в знаке {sun_sign}"
    )
    market_lore = [moon_cue]
    if merc_retro:
        market_lore.append("Меркурий ретроградный — повышенный риск ошибок исполнения")
    else:
        market_lore.append("Меркурий директный — нормальный режим коммуникаций")
    market_lore.append(f"Солнце в {sun_sign} — секторная окраска месяца")
    market_block = "\n".join(f"• {x}" for x in market_lore)

    return render_simple_post(
        author_name=author.name,
        header=f"Космический фон ({now:%d.%m.%Y %H:%M} UTC)",
        idea_block=idea_block,
        analysis_block=analysis_block,
        market_block=market_block,
        footer_tag="NodeVision · астро (Swiss Ephemeris)",
    )


async def analyze_index_or_oil(
    session: aiohttp.ClientSession, author: Author, db: Db
) -> str:
    """Index/Oil Fundamental — honest 'feed integration in progress' placeholder.

    Free-tier feeds for SPX/NASDAQ/Oil require API keys we haven't acquired
    (Twelve Data / Finnhub Premium ~$30/mo). Returns the same 3-section
    layout as live authors so the post structure is consistent — but
    populated with sector context + integration status, no fake numbers.
    """
    from .templates import render_simple_post

    if author.theme == "indices":
        asset_word = "индексы (SPX/NASDAQ/DJI)"
        chart_url = "tradingview.com/chart/?symbol=SPX"
        market_lore = (
            "• Влияние Fed на индексы через ставки + QT\n"
            "• Корреляция SPX с DXY (доллар вверх → SPX часто вниз)\n"
            "• Earnings season — главный драйвер волатильности"
        )
    else:  # oil_gas
        asset_word = "нефть и газ (WTI, Brent, Henry Hub)"
        chart_url = "tradingview.com/chart/?symbol=USOIL"
        market_lore = (
            "• Геополитика (ОПЕК+, Ближний Восток) → нефть\n"
            "• Складские запасы EIA по средам — главный триггер\n"
            "• Ралли доллара ослабляет нефть"
        )

    return render_simple_post(
        author_name=author.name,
        header=f"{asset_word} — обзор",
        idea_block=(
            "⚙️ <b>Интеграция в работе.</b> Готовая trade idea появится после "
            "подключения платного фида (Twelve Data / Finnhub Premium). "
            "Confidence: <b>—</b> (нет данных)."
        ),
        analysis_block=(
            "Бесплатные источники цен для этого класса активов либо закрылись "
            "(Yahoo Finance), либо отдают сильно лагнутые данные (Stooq). "
            "Подключаем платный фид."
        ),
        market_block=f"{market_lore}\n<i>График сейчас: {chart_url}</i>",
        footer_tag=f"NodeVision · {author.theme}",
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


async def _live_render(session: aiohttp.ClientSession, author: Author, db: Db) -> str:
    """Live recompute via the analyzer matched by author.theme + author.style.

    Used by signal_worker to generate the post text that gets stored in
    alerts.meta.rendered_text. NOT called from the bot click handler.
    """
    if author.style == "ml" and author.theme == "crypto":
        return await analyze_crypto_ml(session, author, db)
    if author.theme == "crypto":
        return await analyze_crypto_technical(session, author, db)
    if author.theme == "gold_silver":
        return await analyze_gold_news(session, author, db)
    if author.theme == "currencies":
        return await analyze_currency(session, author, db, author.style or "news")
    if author.style == "astro":
        return await analyze_astro(session, author, db)
    if author.theme in ("indices", "oil_gas"):
        return await analyze_index_or_oil(session, author, db)
    return _err_block(author, f"unknown analyzer for theme={author.theme} style={author.style}")


async def compute_alert_payload(
    session: aiohttp.ClientSession, author: Author, db: Db,
) -> dict | None:
    """Public worker entry point. Renders + structures author's post.

    Returns dict with keys ready for `db.upsert_alert`:
        direction, confidence, entry_price, stop_loss, take_profit,
        rendered_text, label
    or None if generation failed at a level we shouldn't write to DB.

    For non-TA authors (Astro/Currency/Index/Oil) the structured fields
    are coarse defaults — direction='hold', no entry/SL/TP. The rendered
    text is what subscribers actually see.
    """
    try:
        text = await _live_render(session, author, db)
    except Exception:
        log.exception("live render failed for %s", author.slug)
        return None

    # For TA authors we'd ideally extract Setup back from the rendered post
    # to populate structured columns, but the simple route is to recompute
    # Setup once for storage purposes. Skipping that for now — non-essential
    # for the bot UX (subscribers see rendered_text, not the columns). Fill
    # safe defaults; columns can be promoted later if a dashboard needs them.
    return {
        "direction": "info",
        "rendered_text": text,
        "label": author.style,
        "confidence": None,
        "entry_price": None,
        "stop_loss": None,
        "take_profit": None,
    }


async def render_for_author(author: Author, db: Db) -> str:
    """Bot click handler. Reads stored rendered_text from alerts.

    The signal_worker writes new alerts every 5 minutes (bar_time-deduped),
    so subscribers see the SAME post across multiple clicks until the
    worker ships a fresh bar's update. Cold-start fallback shows a
    "waiting" message — never falls back to live recompute (that would
    re-introduce the "different idea on every click" symptom).
    """
    from .signal_worker import COLD_START_MESSAGE
    try:
        alert = await db.latest_alert(author.strategy_id)
    except Exception:
        log.exception("latest_alert query failed for %s", author.slug)
        return _err_block(author, "ошибка чтения сигнала из БД")
    if alert is None:
        return COLD_START_MESSAGE
    text = alert.meta.get("rendered_text") if alert.meta else None
    if not text:
        return COLD_START_MESSAGE
    return text


def _err_block(author: Author, reason: str) -> str:
    return (
        f"<b>{author.name}</b>\n\n"
        f"⚠️ {reason}\n\n"
        f"<i>NodeVision · {author.theme}</i>"
    )
