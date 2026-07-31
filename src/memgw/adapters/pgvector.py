"""The self-hosted adapter. Postgres in production, SQLite for development.

Deliberately the *first* adapter rather than the last: it runs offline with no
API key and brings no magic of its own, so it is the honest baseline for whether
the contract can carry a provider that is only a store.

Search is exact -- every memory in scope is scored -- rather than approximate.
For a per-end-user store that is the right trade at this size, and it keeps every
declared capability literally true. Native ``pgvector`` ANN indexing (a ``vector``
column and ``<=>`` ordering) is a follow-up, and until it lands this adapter is
correct but scans. That limit is stated in the README, not discovered in
production.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Index,
    MetaData,
    String,
    Table,
    Text,
    delete,
    insert,
    select,
    update,
)
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from memgw import adapters
from memgw.capabilities import Capabilities
from memgw.catalog import new_id
from memgw.embedding import Embedder, Extractor, cosine
from memgw.errors import UnsupportedCapability
from memgw.types import Episode, HealthStatus, ProviderMemory, Scope, SearchQuery

_metadata = MetaData()

memories = Table(
    "memgw_memories",
    _metadata,
    Column("native_id", String(48), primary_key=True),
    Column("subject", String(256), nullable=False),
    Column("agent", String(256)),
    Column("session", String(256)),
    Column("labels", JSON, nullable=False),
    Column("content", Text, nullable=False),
    Column("embedding", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Index("ix_memgw_memories_scope", "subject", "agent", "session"),
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class PgvectorAdapter:
    name = "pgvector"

    def __init__(
        self,
        url: str = "sqlite+aiosqlite:///memgw_pgvector.db",
        *,
        embedder: Embedder,
        extractor: Extractor | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if url.endswith(":memory:"):
            kwargs = {"poolclass": StaticPool, "connect_args": {"check_same_thread": False}}
        self._engine = create_async_engine(url, **kwargs)
        self._embedder = embedder
        self._extractor = extractor

    async def init(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(_metadata.create_all)

    async def close(self) -> None:
        await self._engine.dispose()

    # -- introspection --------------------------------------------------------

    def capabilities(self) -> Capabilities:
        return Capabilities(
            # Configuration decides this, not the class: no extractor means this
            # instance genuinely cannot turn a conversation into memories.
            supports_ingest=self._extractor is not None,
            supports_upsert=True,
            supports_update=True,
            supports_delete=True,
            supports_delete_by_scope=True,
            search_modes=["semantic"],
            supports_score=True,
            max_limit=200,
            scope_dims=["subject", "agent", "session"],
            supports_labels=True,
            memory_model="flat_facts",
            dedup="none",
            supports_export=True,
            supports_import=True,
            consistency="read_after_write",
            metered_externally=False,
        )

    async def health(self) -> HealthStatus:
        try:
            async with self._engine.connect() as conn:
                await conn.execute(select(1))
            return HealthStatus(ok=True)
        except Exception as exc:  # noqa: BLE001
            return HealthStatus(ok=False, detail=str(exc))

    # -- write ----------------------------------------------------------------

    async def ingest(self, episode: Episode, scope: Scope) -> list[ProviderMemory]:
        if self._extractor is None:
            raise UnsupportedCapability(
                "this pgvector instance has no extractor configured; it can store "
                "ready-made facts but cannot decide what a conversation is worth"
            )
        facts = await self._extractor.extract(episode)
        if not facts:
            return []
        return await self.upsert(facts, scope)

    async def upsert(self, facts: list[str], scope: Scope) -> list[ProviderMemory]:
        vectors = await self._embedder.embed(facts)
        now = _now()
        rows = []
        for fact, vector in zip(facts, vectors):
            rows.append(
                {
                    "native_id": new_id("pv"),
                    "subject": scope.subject,
                    "agent": scope.agent,
                    "session": scope.session,
                    "labels": dict(scope.labels),
                    "content": fact,
                    "embedding": list(vector),
                    "created_at": now,
                    "updated_at": now,
                }
            )
        async with self._engine.begin() as conn:
            await conn.execute(insert(memories), rows)
        return [
            ProviderMemory(
                native_id=row["native_id"],
                content=row["content"],
                created_at=now,
                updated_at=now,
                raw={"adapter": "pgvector"},
            )
            for row in rows
        ]

    async def update(self, native_id: str, content: str) -> ProviderMemory:
        [vector] = await self._embedder.embed([content])
        now = _now()
        async with self._engine.begin() as conn:
            await conn.execute(
                update(memories)
                .where(memories.c.native_id == native_id)
                .values(content=content, embedding=list(vector), updated_at=now)
            )
        got = await self.get(native_id)
        if got is None:
            raise UnsupportedCapability(f"no memory {native_id!r} to update")
        return got

    # -- read -----------------------------------------------------------------

    async def search(self, query: SearchQuery, scope: Scope) -> list[ProviderMemory]:
        if query.mode != "semantic":
            raise UnsupportedCapability(
                f"pgvector adapter cannot serve {query.mode!r} search",
                details={"available": ["semantic"]},
            )

        [needle] = await self._embedder.embed([query.query])
        rows = await self._rows_in_scope(scope)

        scored = []
        for row in rows:
            score = cosine(needle, row["embedding"])
            if query.min_score is not None and score < query.min_score:
                continue
            scored.append((score, row))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [self._to_memory(row, score) for score, row in scored[: query.limit]]

    async def get(self, native_id: str) -> ProviderMemory | None:
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(select(memories).where(memories.c.native_id == native_id))
            ).mappings().one_or_none()
        return None if row is None else self._to_memory(row, None)

    # -- delete ---------------------------------------------------------------

    async def delete(self, native_id: str) -> bool:
        async with self._engine.begin() as conn:
            result = await conn.execute(delete(memories).where(memories.c.native_id == native_id))
        return result.rowcount > 0

    async def delete_scope(self, scope: Scope) -> int:
        rows = await self._rows_in_scope(scope)
        doomed = [row["native_id"] for row in rows]
        if not doomed:
            return 0
        async with self._engine.begin() as conn:
            await conn.execute(delete(memories).where(memories.c.native_id.in_(doomed)))
        return len(doomed)

    # -- internals ------------------------------------------------------------

    async def _rows_in_scope(self, scope: Scope) -> list[dict[str, Any]]:
        """Every dimension the caller set is applied. Dropping one would return
        another end-user's memories, so the filter is built from ``scope.dims()``
        rather than from whichever columns happen to be indexed."""
        conditions = [memories.c.subject == scope.subject]
        if scope.agent:
            conditions.append(memories.c.agent == scope.agent)
        if scope.session:
            conditions.append(memories.c.session == scope.session)

        async with self._engine.connect() as conn:
            rows = (await conn.execute(select(memories).where(*conditions))).mappings().all()

        if not scope.labels:
            return [dict(row) for row in rows]
        return [
            dict(row)
            for row in rows
            if all((row["labels"] or {}).get(k) == v for k, v in scope.labels.items())
        ]

    @staticmethod
    def _to_memory(row: Any, score: float | None) -> ProviderMemory:
        return ProviderMemory(
            native_id=row["native_id"],
            content=row["content"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            score=score,
            raw={"adapter": "pgvector"},
        )


adapters.register("pgvector", PgvectorAdapter)
