"""Reads signals from a Redis Stream via consumer-group semantics.

Why consumer groups:
- Survives bot restarts — Redis remembers what we acked, redelivers what we didn't.
- Survives bot crashes mid-send — unacked entries reappear after PEL idle timeout.
- Multiple consumers (workers) can read the same stream cooperatively later.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

import redis.asyncio as aioredis
from redis.exceptions import ResponseError

log = logging.getLogger(__name__)


class StreamConsumer:
    def __init__(
        self,
        url: str,
        stream_key: str,
        group: str,
        consumer: str,
        block_ms: int = 5_000,
        batch_size: int = 16,
    ) -> None:
        self.url = url
        self.stream_key = stream_key
        self.group = group
        self.consumer = consumer
        self.block_ms = block_ms
        self.batch_size = batch_size
        self._client: aioredis.Redis | None = None

    async def connect(self) -> None:
        self._client = aioredis.from_url(self.url, decode_responses=True)
        # Ensure stream + group exist. MKSTREAM lets us create both at once
        # even if the producer hasn't pushed anything yet.
        try:
            await self._client.xgroup_create(
                self.stream_key, self.group, id="$", mkstream=True
            )
            log.info("Created consumer group %s on stream %s", self.group, self.stream_key)
        except ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
            log.info("Consumer group %s already exists on %s", self.group, self.stream_key)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def stream(self) -> AsyncIterator[tuple[str, dict[str, str]]]:
        """
        Yield (entry_id, payload) tuples until cancelled.

        First yields any pending entries this consumer owns (recovery from crash),
        then switches to fresh entries blocked-read from the group.
        """
        if self._client is None:
            raise RuntimeError("call connect() first")

        # 1. Pending recovery — pick up what we owned but didn't ack last run.
        async for entry in self._iterate(start_id="0"):
            yield entry

        # 2. Live tail — block until new entries arrive.
        async for entry in self._iterate(start_id=">"):
            yield entry

    async def _iterate(self, start_id: str) -> AsyncIterator[tuple[str, dict[str, str]]]:
        assert self._client is not None
        while True:
            try:
                resp = await self._client.xreadgroup(
                    groupname=self.group,
                    consumername=self.consumer,
                    streams={self.stream_key: start_id},
                    count=self.batch_size,
                    block=self.block_ms,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("XREADGROUP failed; backing off 2s")
                await asyncio.sleep(2)
                continue

            if not resp:
                # Recovery loop is done once a > poll returns empty.
                if start_id == "0":
                    return
                continue

            # resp shape: [(stream_key, [(entry_id, {field: value}), ...])]
            for _stream, entries in resp:
                for entry_id, payload in entries:
                    yield entry_id, payload

            if start_id == "0":
                # We delivered all pending; exit pending phase.
                return

    async def ack(self, entry_id: str) -> None:
        assert self._client is not None
        await self._client.xack(self.stream_key, self.group, entry_id)
