"""Live market data — current price, 24h change. Free Binance public API.

No auth required, rate limit ~1200 req/min per IP. Plenty for our use:
the bot fetches one ticker per /author button click, ~10/minute peak.
"""
from __future__ import annotations

import logging
from typing import Any

import aiohttp

log = logging.getLogger(__name__)

BINANCE_24H = "https://api.binance.com/api/v3/ticker/24hr"
TIMEOUT = aiohttp.ClientTimeout(total=5)


class MarketData:
    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession(timeout=TIMEOUT)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def ticker_24h(self, symbol: str) -> dict[str, Any] | None:
        """Returns {price, change_pct_24h, high_24h, low_24h} or None if API fails.

        symbol: Binance pair like BTCUSDT.

        Failure is logged and returns None — caller can degrade gracefully
        (rendering message without the price line) instead of refusing to
        respond to the user.
        """
        assert self._session is not None
        try:
            async with self._session.get(
                BINANCE_24H, params={"symbol": symbol}
            ) as resp:
                if resp.status != 200:
                    log.warning("Binance 24h ticker %s: status %s", symbol, resp.status)
                    return None
                data = await resp.json()
        except Exception:
            log.exception("Binance 24h ticker %s failed", symbol)
            return None

        try:
            return {
                "price": float(data["lastPrice"]),
                "change_pct_24h": float(data["priceChangePercent"]),
                "high_24h": float(data["highPrice"]),
                "low_24h": float(data["lowPrice"]),
                "volume_quote_24h": float(data["quoteVolume"]),
            }
        except (KeyError, TypeError, ValueError):
            log.warning("Binance 24h ticker %s: unexpected payload %r", symbol, data)
            return None
