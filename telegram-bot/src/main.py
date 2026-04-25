"""Bot entry point.

Glues the Redis Streams consumer to the Telegram publisher:
  - read entry from stream
  - format via formatter
  - send to default channel (single-bot MVP)
  - ack on confirmed delivery; leave un-acked on failure for retry
"""
from __future__ import annotations

import asyncio
import logging
import signal

from .config import settings
from .formatters import format_signal
from .publisher import TelegramPublisher
from .redis_consumer import StreamConsumer

log = logging.getLogger(__name__)


async def run() -> None:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    chat_id = settings.chat_id_int
    if chat_id is None:
        log.warning(
            "TELEGRAM_DEFAULT_CHAT_ID is empty — bot will read from stream and log only. "
            "Set the channel ID in .env to enable real delivery."
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
        "Bot started: stream=%s group=%s consumer=%s chat=%s dry_run=%s",
        settings.redis_stream_key,
        settings.redis_consumer_group,
        settings.redis_consumer_name,
        chat_id or "(unset)",
        settings.dry_run,
    )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    try:
        async for entry_id, payload in consumer.stream():
            if stop_event.is_set():
                break

            try:
                text = format_signal(payload)
            except Exception:
                log.exception("formatter failed on entry %s payload=%r", entry_id, payload)
                # bad payload is unrecoverable — ack so we don't loop on it forever.
                await consumer.ack(entry_id)
                continue

            if chat_id is None:
                log.info("[no chat configured] would post:\n%s", text)
                await consumer.ack(entry_id)
                continue

            ok = await publisher.post(chat_id, text)
            if ok:
                await consumer.ack(entry_id)
                log.info("posted entry %s", entry_id)
            else:
                # Don't ack — Redis will redeliver after PEL idle timeout.
                log.warning("delivery failed for %s, leaving un-acked", entry_id)
                await asyncio.sleep(5)
    finally:
        await consumer.close()
        await publisher.close()


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
