"""Signal worker — generates per-author posts on a schedule and stores
the rendered text in the alerts table.

Why this exists (per Денис 2026-04-30):
> "идеи должны фиксироваться в базе данных и даваться по мере поступления
>  от сигнала, я не должен нажимать каждый раз на кнопку и потом видеть
>  каждый раз новую идею"

Architecture:
- One async task in the bot process (no separate service to deploy).
- Loop: every `INTERVAL_SECONDS` (default 300 = 5 min) iterate through
  all active authors, run the live analyzer, UPSERT into `alerts` table
  with `bar_time` floored to the author's TF.
- UNIQUE (strategy_id, bar_time, direction) on the table dedupes — when
  the same bar is seen twice, the second insert silently DOES NOTHING
  via ON CONFLICT.
- Bot click handler reads `db.latest_alert(strategy_id)` → returns
  `alert.meta['rendered_text']` (which the worker already rendered).

Cold start: on first start, run_once is invoked synchronously before the
sleep loop so all authors have at least one alert ready.

This file is INTENTIONALLY simple — see Денис's "можно чуть проще":
no significant-change detection, no stale gating, no per-TF schedule
optimisation. Just: every 5 min, refresh all, dedup via UNIQUE.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import aiohttp

log = logging.getLogger(__name__)

DEFAULT_REFRESH_SECONDS = 300  # 5 min

# Map TF literal → seconds-per-bar. Used by bar_time_ms() to floor the
# current epoch time to the start of the current bar.
_TF_SECONDS = {
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
}

# Direction normalisation. Existing alerts table uses BUY/SELL strings
# (legacy from when the supervised model wrote them). We map our
# idea_engine vocabulary (long/short/hold) to that string set so the
# UNIQUE constraint and any historical dashboards keep working.
_DIRECTION_MAP = {
    "long": "BUY",
    "short": "SELL",
    "hold": "HOLD",
    "info": "INFO",
}


COLD_START_MESSAGE = (
    "⏳ <b>Готовим первый сигнал…</b>\n\n"
    "Воркер сейчас собирает данные с биржи и формирует пост — обычно занимает "
    "до минуты. Нажмите кнопку через 30-60 секунд, и сигнал будет здесь."
)


# ─── Bar-time helper ────────────────────────────────────────────────────────

def bar_time_ms(interval: str, now_ms: int | None = None) -> int:
    """Floor current epoch (ms) to the start of the current bar.

    Two consecutive worker passes within the same bar produce the same
    bar_time — the alerts UNIQUE constraint then silently dedupes the
    second INSERT, so we don't write the same row twice.
    """
    sec = _TF_SECONDS.get(interval)
    if sec is None:
        raise ValueError(f"unknown interval {interval!r}; expected one of {sorted(_TF_SECONDS)}")
    t = now_ms if now_ms is not None else int(time.time() * 1000)
    bar_ms = sec * 1000
    return (t // bar_ms) * bar_ms


def setup_to_alert_direction(direction: str) -> str:
    """Map idea_engine direction literal → alerts.direction VARCHAR."""
    return _DIRECTION_MAP.get(direction, direction.upper())


# ─── Refresh stats ──────────────────────────────────────────────────────────

@dataclass
class RefreshSummary:
    success: int = 0
    failed: int = 0
    skipped: int = 0  # e.g. analyzer reported "Binance temporarily unavailable"

    @property
    def total(self) -> int:
        return self.success + self.failed + self.skipped


# ─── One-shot refresh (called by run_forever AND once on cold start) ────────

async def refresh_one(
    session: aiohttp.ClientSession,
    db,
    author,
) -> bool:
    """Render an author's post via the live analyzer and UPSERT into alerts.

    Returns True on success (alert row written or already present), False on
    analyzer failure. Skipped (analyzer-reported degradation) returns True
    too — the post was rendered, just with a "data unavailable" body.
    """
    # Lazy imports to avoid circular dependency (analyzers imports db.Author).
    from .analyzers import compute_alert_payload

    payload = await compute_alert_payload(session, author, db)
    if payload is None:
        return False
    bar_t = bar_time_ms(author.interval)
    await db.upsert_alert(
        strategy_id=author.strategy_id,
        symbol=author.symbol,
        interval=author.interval,
        direction=setup_to_alert_direction(payload["direction"]),
        confidence=payload.get("confidence"),
        entry_price=payload.get("entry_price"),
        stop_loss=payload.get("stop_loss"),
        take_profit=payload.get("take_profit"),
        bar_time=bar_t,
        rendered_text=payload["rendered_text"],
        label=payload.get("label", ""),
    )
    return True


async def refresh_all(db, timeout_sec: int = 30) -> RefreshSummary:
    """One full sweep over all active authors. Failures are isolated per author."""
    summary = RefreshSummary()
    authors = await db.list_authors()
    log.info("signal_worker: refreshing %d authors", len(authors))
    timeout = aiohttp.ClientTimeout(total=timeout_sec)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for author in authors:
            try:
                ok = await refresh_one(session, db, author)
                if ok:
                    summary.success += 1
                else:
                    summary.skipped += 1
            except Exception:
                log.exception("signal_worker: author %s failed", author.slug)
                summary.failed += 1
    log.info(
        "signal_worker: %d ok, %d skipped, %d failed",
        summary.success, summary.skipped, summary.failed,
    )
    return summary


async def run_forever(db, every_seconds: int = DEFAULT_REFRESH_SECONDS) -> None:
    """Cold-start refresh + infinite loop. Run as asyncio.create_task in main."""
    log.info("signal_worker: cold-start refresh")
    try:
        await refresh_all(db)
    except Exception:
        log.exception("signal_worker: cold-start refresh failed")

    while True:
        try:
            await asyncio.sleep(every_seconds)
            await refresh_all(db)
        except asyncio.CancelledError:
            log.info("signal_worker: cancelled, exiting")
            raise
        except Exception:
            log.exception("signal_worker: cycle failed, sleeping until next tick")
