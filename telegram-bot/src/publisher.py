"""Sends formatted messages to Telegram channels via aiogram."""
from __future__ import annotations

import logging

from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramRetryAfter, TelegramAPIError
from aiogram.client.default import DefaultBotProperties

log = logging.getLogger(__name__)


class TelegramPublisher:
    def __init__(self, token: str, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self._bot = Bot(token=token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    async def close(self) -> None:
        await self._bot.session.close()

    async def post(self, chat_id: int, text: str) -> bool:
        """Returns True on confirmed delivery (or dry-run), False on retryable failure."""
        if self.dry_run:
            log.info("[dry-run] would post to %s:\n%s", chat_id, text)
            return True
        try:
            await self._bot.send_message(chat_id=chat_id, text=text, disable_web_page_preview=True)
            return True
        except TelegramRetryAfter as e:
            # Telegram tells us to back off; mark as not delivered so we retry.
            log.warning("Telegram rate limited, retry after %ss", e.retry_after)
            return False
        except TelegramAPIError:
            log.exception("Telegram API error posting to %s", chat_id)
            return False
