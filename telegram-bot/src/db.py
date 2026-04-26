"""Postgres helpers — fetch authors list, latest alert, user premium status.

Used by the bot's command + callback handlers to render the menu and the
3-in-1 author response. Read-only by design: the bot never writes to
postgres; the Go runner is the only writer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import asyncpg

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Author:
    slug: str
    name: str
    theme: str
    style: str
    bio: str
    position: int
    is_premium: bool
    symbol: str
    interval: str
    risk_tier: str
    strategy_id: str


@dataclass(frozen=True)
class Alert:
    id: str
    direction: str
    label: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    bar_time: int
    created_at: datetime


class Db:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        # Small pool — bot is a single process with low concurrency.
        self._pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=4)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def list_authors(self) -> list[Author]:
        """Authors visible in the bot menu, ordered by author_position.

        Filters: is_active=true, author_slug NOT NULL. Premium authors
        are returned too — paywall is enforced at click time, so the
        button is visible to all but only opens for premium subscribers.
        """
        assert self._pool is not None
        rows = await self._pool.fetch(
            """
            SELECT id::text AS strategy_id,
                   author_slug, author_name, author_theme, author_style,
                   COALESCE(author_bio, '') AS author_bio,
                   COALESCE(author_position, 99) AS author_position,
                   is_premium,
                   selected_pair AS symbol, interval, risk_tier
              FROM strategies
             WHERE is_active = true
               AND author_slug IS NOT NULL
             ORDER BY author_position, author_slug
            """
        )
        return [
            Author(
                slug=r["author_slug"],
                name=r["author_name"],
                theme=r["author_theme"],
                style=r["author_style"],
                bio=r["author_bio"],
                position=r["author_position"],
                is_premium=r["is_premium"],
                symbol=r["symbol"],
                interval=r["interval"],
                risk_tier=r["risk_tier"],
                strategy_id=r["strategy_id"],
            )
            for r in rows
        ]

    async def get_author_by_slug(self, slug: str) -> Author | None:
        assert self._pool is not None
        row = await self._pool.fetchrow(
            """
            SELECT id::text AS strategy_id,
                   author_slug, author_name, author_theme, author_style,
                   COALESCE(author_bio, '') AS author_bio,
                   COALESCE(author_position, 99) AS author_position,
                   is_premium,
                   selected_pair AS symbol, interval, risk_tier
              FROM strategies
             WHERE author_slug = $1 AND is_active = true
             LIMIT 1
            """,
            slug,
        )
        if row is None:
            return None
        return Author(
            slug=row["author_slug"],
            name=row["author_name"],
            theme=row["author_theme"],
            style=row["author_style"],
            bio=row["author_bio"],
            position=row["author_position"],
            is_premium=row["is_premium"],
            symbol=row["symbol"],
            interval=row["interval"],
            risk_tier=row["risk_tier"],
            strategy_id=row["strategy_id"],
        )

    async def latest_alert(self, strategy_id: str) -> Alert | None:
        """Most recent alert for a strategy. Returns None if never fired."""
        assert self._pool is not None
        row = await self._pool.fetchrow(
            """
            SELECT id::text, direction, label,
                   confidence::float8 AS confidence,
                   entry_price::float8 AS entry_price,
                   stop_loss::float8 AS stop_loss,
                   take_profit::float8 AS take_profit,
                   bar_time, created_at
              FROM alerts
             WHERE strategy_id = $1
             ORDER BY created_at DESC
             LIMIT 1
            """,
            strategy_id,
        )
        if row is None:
            return None
        return Alert(
            id=row["id"],
            direction=row["direction"],
            label=row["label"] or "",
            confidence=float(row["confidence"]) if row["confidence"] is not None else 0.0,
            entry_price=float(row["entry_price"]) if row["entry_price"] is not None else 0.0,
            stop_loss=float(row["stop_loss"]) if row["stop_loss"] is not None else 0.0,
            take_profit=float(row["take_profit"]) if row["take_profit"] is not None else 0.0,
            bar_time=row["bar_time"],
            created_at=row["created_at"],
        )

    async def user_is_premium(self, telegram_chat_id: int) -> bool:
        """True if the chat owner has a non-expired premium subscription."""
        assert self._pool is not None
        row = await self._pool.fetchrow(
            """
            SELECT is_premium_subscriber, premium_until
              FROM users
             WHERE telegram_chat_id = $1
             LIMIT 1
            """,
            telegram_chat_id,
        )
        if row is None or not row["is_premium_subscriber"]:
            return False
        # NULL premium_until = lifetime / forever-trial
        if row["premium_until"] is None:
            return True
        return row["premium_until"] > datetime.utcnow()
