"""Telegram command + callback handlers — bot interactive layer.

/start              — welcome + author menu (admin only for now)
/menu               — re-send the author menu
/whoami             — show your chat_id (debugging premium gating)
callback "author:X" — render the 3-in-1 response for author with slug X

Admin-only gate via ADMIN_CHAT_IDS in config. Premium authors check
users.is_premium_subscriber on click and show a paywall instead of
the response if the user isn't subscribed.
"""
from __future__ import annotations

import logging

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from .analyzers import render_for_author
from .db import Db
from .market_data import MarketData
from .render import render_premium_paywall

log = logging.getLogger(__name__)


def register_handlers(
    dp: Dispatcher,
    db: Db,
    market: MarketData,
    admin_chat_ids: set[int],
) -> None:
    @dp.message(Command("start"))
    async def cmd_start(msg: Message) -> None:
        if msg.from_user is None or msg.from_user.id not in admin_chat_ids:
            await msg.answer(
                "Бот пока в закрытом режиме. Доступ только администраторам."
            )
            log.info("denied /start to user_id=%s", msg.from_user.id if msg.from_user else None)
            return

        await msg.answer(
            "<b>NodeVision</b> · авторские сценарии\n\n"
            "Выбери автора, чтобы увидеть аналитику, торговую идею "
            "и текущее состояние сделки.",
            reply_markup=await _build_menu_keyboard(db),
        )

    @dp.message(Command("menu"))
    async def cmd_menu(msg: Message) -> None:
        if msg.from_user is None or msg.from_user.id not in admin_chat_ids:
            return
        await msg.answer(
            "Авторы:",
            reply_markup=await _build_menu_keyboard(db),
        )

    @dp.message(Command("whoami"))
    async def cmd_whoami(msg: Message) -> None:
        if msg.from_user is None:
            return
        is_admin = msg.from_user.id in admin_chat_ids
        is_premium = await db.user_is_premium(msg.from_user.id)
        await msg.answer(
            f"<b>Your IDs</b>\n"
            f"telegram user_id: <code>{msg.from_user.id}</code>\n"
            f"admin: {is_admin}\n"
            f"premium subscriber: {is_premium}"
        )

    @dp.callback_query(F.data.startswith("author:"))
    async def cb_author(cq: CallbackQuery) -> None:
        if cq.from_user is None or cq.data is None or not isinstance(cq.message, Message):
            await cq.answer()
            return
        if cq.from_user.id not in admin_chat_ids:
            await cq.answer("Доступ закрыт", show_alert=True)
            return

        slug = cq.data.split(":", 1)[1]
        author = await db.get_author_by_slug(slug)
        if author is None:
            await cq.answer("Автор не найден", show_alert=True)
            return

        # Premium gate
        if author.is_premium and not await db.user_is_premium(cq.from_user.id):
            text = render_premium_paywall(author)
            await cq.message.answer(text, disable_web_page_preview=True)
            await cq.answer()
            return

        # Per-style analyzer — fetches live data, returns finished message.
        text = await render_for_author(author, db)
        await cq.message.answer(text, disable_web_page_preview=True)
        await cq.answer()


async def _build_menu_keyboard(db: Db) -> InlineKeyboardMarkup:
    authors = await db.list_authors()
    if not authors:
        return InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text="(нет авторов)", callback_data="noop")]]
        )

    # Group: free authors first, premium last (separated). 2 buttons per row
    # for readability when there are 5+ authors.
    free = [a for a in authors if not a.is_premium]
    premium = [a for a in authors if a.is_premium]

    rows: list[list[InlineKeyboardButton]] = []
    pair: list[InlineKeyboardButton] = []
    for a in free:
        pair.append(_button(a))
        if len(pair) == 2:
            rows.append(pair)
            pair = []
    if pair:
        rows.append(pair)

    # Premium row(s) — always full-width to draw attention
    for a in premium:
        rows.append([_button(a)])

    return InlineKeyboardMarkup(inline_keyboard=rows)


def _button(a) -> InlineKeyboardButton:  # noqa: ANN001 — Author from db.py
    icon = _theme_icon(a.theme)
    label = f"{icon} {a.name}"
    return InlineKeyboardButton(text=label, callback_data=f"author:{a.slug}")


def _theme_icon(theme: str) -> str:
    return {
        "crypto": "₿",
        "gold_silver": "🥇",
        "currencies": "💱",
        "indices": "📊",
        "oil_gas": "⛽",
        "astro": "🌌",
        "multi": "🔔",
    }.get(theme, "📈")
