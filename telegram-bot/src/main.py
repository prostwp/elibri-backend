"""Bot entry point.

Two modes selectable via BOT_MODE env:

  interactive (default, MVP)
    - Long-polls Telegram for /start, /menu, /whoami, button clicks
    - On click: fetches latest alert + live ticker, renders 3-in-1 reply
    - Admin-gated; premium gate for premium-flagged authors
    - DOES NOT consume Redis stream (no channel fanout)

  fanout (legacy)
    - Consumes signals:btc:4h Redis stream
    - Posts every entry to TELEGRAM_DEFAULT_CHAT_ID
    - Used during the B.2 paper trade demo; kept for backwards compat

Most signal traffic in V4 lives in interactive mode: users browse the
author menu and pull signals on demand. Channel fanout returns when we
spin up per-author public channels with curated formatting.
"""
from __future__ import annotations

import asyncio
import logging
import signal as sig

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from .config import settings
from .db import Db
from .formatters import format_signal
from .handlers import register_handlers
from .market_data import MarketData
from .publisher import TelegramPublisher
from .redis_consumer import StreamConsumer

log = logging.getLogger(__name__)


async def run_interactive() -> None:
    """V4 mode: long-poll for commands + callbacks; no stream consumption."""
    bot = Bot(
        token=settings.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()

    db = Db(settings.postgres_url)
    market = MarketData()
    await db.connect()
    await market.connect()

    register_handlers(dp, db, market, settings.admin_chat_id_set)

    log.info(
        "Bot started in interactive mode: bot=%s admins=%s premium_check=db",
        settings.telegram_bot_username,
        sorted(settings.admin_chat_id_set) or "(none configured)",
    )

    try:
        await dp.start_polling(bot, handle_signals=True)
    finally:
        await market.close()
        await db.close()
        await bot.session.close()


async def run_fanout() -> None:
    """Legacy mode: stream consumer → default channel."""
    chat_id = settings.chat_id_int
    if chat_id is None:
        log.warning(
            "TELEGRAM_DEFAULT_CHAT_ID empty in fanout mode — entries will be "
            "consumed and acked but not posted. Set the channel ID in .env."
        )

    publisher = TelegramPublisher(settings.telegram_bot_token, dry_run=settings.dry_run)
    consumer = StreamConsumer(
        url=settings.redis_url,
        stream_key=settings.redis_stream_key,
        group=settings.redis_consumer_group,
        consumer=settings.redis_consumer_name,
    )
    await consumer.connect()
    log.info(
        "Bot started in fanout mode: stream=%s group=%s consumer=%s chat=%s",
        settings.redis_stream_key,
        settings.redis_consumer_group,
        settings.redis_consumer_name,
        chat_id or "(unset)",
    )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for s in (sig.SIGINT, sig.SIGTERM):
        loop.add_signal_handler(s, stop.set)

    try:
        async for entry_id, payload in consumer.stream():
            if stop.is_set():
                break
            try:
                text = format_signal(payload)
            except Exception:
                log.exception("formatter failed on %s", entry_id)
                await consumer.ack(entry_id)
                continue

            if chat_id is None:
                log.info("[no chat] entry %s text:\n%s", entry_id, text)
                await consumer.ack(entry_id)
                continue

            ok = await publisher.post(chat_id, text)
            if ok:
                await consumer.ack(entry_id)
                log.info("posted entry %s", entry_id)
            else:
                log.warning("delivery failed for %s; leaving un-acked", entry_id)
                await asyncio.sleep(5)
    finally:
        await consumer.close()
        await publisher.close()


async def run() -> None:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if settings.bot_mode == "interactive":
        await run_interactive()
    elif settings.bot_mode == "fanout":
        await run_fanout()
    else:
        raise SystemExit(f"unknown BOT_MODE={settings.bot_mode!r}")


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
