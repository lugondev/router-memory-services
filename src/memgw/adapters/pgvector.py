"""The self-hosted adapter. Postgres in production, SQLite for development.

Deliberately the *first* adapter rather than the last: it runs offline with no
API key and brings no magic of its own, so it is the honest baseline for whether
the contract can carry a provider that is only a store.

**On Postgres it uses pgvector, which it did not always do.** The embedding is a
real ``vector(n)`` column and ranking happens in SQL through ``<=>``. The version
before this stored embeddings as JSON and scored them in Python, which meant every
row in scope crossed the wire before ``limit`` could drop any of it: a subject with
ten thousand memories shipped ten thousand vectors to answer a query for three.

**On SQLite it still scans**, because SQLite has no vector type. That path is
unchanged and remains the right answer for development and for embedded mode.

Search is still **exact** either way. Ordering by ``<=>`` without an ANN index scans
too -- but it scans inside Postgres, in C, over data that never moves. An HNSW index
would make it approximate, which is a different promise, so it is a deliberate
follow-up rather than a default.
"""

from __future__ import annotations

import asyncio
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
    text,
    update,
)
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from memgw import adapters
from memgw.capabilities import Capabilities
from memgw.catalog import new_id
from memgw.embedding import Embedder, Extractor, cosine
from memgw.errors import MemoryNotFound, UnsupportedCapability
from memgw.types import Episode, HealthStatus, ProviderMemory, Scope, SearchQuery

_metadata = MetaData()

#: Deletes go one statement per chunk when ids are unavoidable. SQLite builds still
#: in the wild cap bind parameters at 999.
_CHUNK = 500


class _NoGate:
    """A real connection pool hands out distinct connections, so nothing needs
    serialising. Standing in for the lock keeps one code path instead of two."""

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exc: object) -> None:
        return None


DEFAULT_TABLE = "memgw_memories"


def build_table(name: str, metadata: MetaData, *, dimension: int | None) -> Table:
    """One table definition, two column types.

    ``dimension`` set means Postgres with the extension: the embedding is a real
    ``vector(n)``, so ordering happens inside the database. ``None`` means SQLite,
    where the embedding is JSON and scoring happens in Python. Everything else about
    the table is identical, which is what keeps one adapter honest about both.
    """
    if dimension is not None:
        from pgvector.sqlalchemy import Vector

        embedding: Any = Column("embedding", Vector(dimension), nullable=False)
    else:
        embedding = Column("embedding", JSON, nullable=False)

    return Table(
        name,
        metadata,
        Column("native_id", String(48), primary_key=True),
        # Leading the index because it is the first thing every query filters on, and
        # because a store shared by two tenants that both call an end-user "u_1" is
        # one store with one row and two owners.
        Column("tenant", String(128)),
        Column("subject", String(256), nullable=False),
        Column("agent", String(256)),
        Column("session", String(256)),
        Column("labels", JSON, nullable=False),
        Column("content", Text, nullable=False),
        embedding,
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("updated_at", DateTime(timezone=True), nullable=False),
        Index(f"ix_{name}_scope", "tenant", "subject", "agent", "session"),
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
        table: str = DEFAULT_TABLE,
    ) -> None:
        kwargs: dict[str, Any] = {}
        shared_connection = url.endswith(":memory:")
        if shared_connection:
            kwargs = {"poolclass": StaticPool, "connect_args": {"check_same_thread": False}}
        self._engine = create_async_engine(url, **kwargs)
        #: One shared DBAPI connection cannot carry two transactions at once; see the
        #: same note in :mod:`memgw.catalog`.
        self._gate = asyncio.Lock() if shared_connection else _NoGate()
        self._embedder = embedder
        self._extractor = extractor

        self.table_name = table
        #: Postgres gets a real ``vector`` column and orders in SQL; everything else
        #: keeps the JSON column and scores in Python. Decided from the URL because
        #: it has to be known before the table is defined.
        self.native_vectors = self._engine.dialect.name == "postgresql"
        self._metadata = MetaData()
        self._table = build_table(
            table,
            self._metadata,
            dimension=embedder.dimension if self.native_vectors else None,
        )

    @property
    def engine(self):
        return self._engine

    async def init(self) -> None:
        if self.native_vectors:
            await self._prepare_postgres()
        async with self._gate, self._engine.begin() as conn:
            await conn.run_sync(self._metadata.create_all)

    async def _prepare_postgres(self) -> None:
        """Install the extension, and refuse a table built for a different geometry.

        A ``vector(n)`` column has a fixed width, so changing the embedding model
        changes ``n``. Writing 3072-dimensional vectors into a ``vector(1536)`` column
        is an error Postgres will raise; writing them into a table built for a *third*
        model is not, and its only symptom is recall slowly getting worse. So the
        width is checked before anything is written.
        """
        async with self._gate, self._engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

            found = (
                await conn.execute(
                    text(
                        "select a.atttypmod from pg_attribute a "
                        "join pg_class c on c.oid = a.attrelid "
                        "where c.relname = :t and a.attname = 'embedding'"
                    ),
                    {"t": self.table_name},
                )
            ).scalar_one_or_none()

        wanted = self._embedder.dimension
        if found is not None and found != wanted:
            raise ValueError(
                f"table {self.table_name!r} stores {found}-dimension vectors and this "
                f"embedder produces {wanted}. Point at a different table, or re-embed: "
                "two geometries in one column is not a mismatch you would notice."
            )

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
            supports_list=True,
            search_modes=["semantic"],
            supports_score=True,
            max_limit=200,
            scope_dims=["tenant", "subject", "agent", "session"],
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
            async with self._gate, self._engine.connect() as conn:
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
        # strict: an embedder returning a different number of vectors than facts is a
        # bug that would otherwise silently drop memories.
        for fact, vector in zip(facts, vectors, strict=True):
            rows.append(
                {
                    "native_id": new_id("pv"),
                    "tenant": scope.tenant,
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
        async with self._gate, self._engine.begin() as conn:
            await conn.execute(insert(self._table), rows)
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
        async with self._gate, self._engine.begin() as conn:
            result = await conn.execute(
                update(self._table)
                .where(self._table.c.native_id == native_id)
                .values(content=content, embedding=list(vector), updated_at=now)
            )
        if result.rowcount == 0:
            # Absence, not incapacity: 422 here would tell a caller to stop trying a
            # verb that works fine, when the real answer is that the id is wrong.
            raise MemoryNotFound(f"no memory {native_id!r} to update")
        got = await self.get(native_id)
        assert got is not None
        return got

    # -- read -----------------------------------------------------------------

    async def search(self, query: SearchQuery, scope: Scope) -> list[ProviderMemory]:
        if query.mode != "semantic":
            raise UnsupportedCapability(
                f"pgvector adapter cannot serve {query.mode!r} search",
                details={"available": ["semantic"]},
            )

        [needle] = await self._embedder.embed([query.query])
        if self.native_vectors and not scope.labels:
            return await self._search_in_sql(query, scope, needle)
        return await self._search_in_python(query, scope, needle)

    async def _search_in_sql(
        self, query: SearchQuery, scope: Scope, needle: list[float]
    ) -> list[ProviderMemory]:
        """Rank and cut inside Postgres.

        The old path read every row in scope into Python before it could drop one, so
        ``limit`` saved nothing: a subject with ten thousand memories shipped ten
        thousand vectors across the wire to answer a query for three.

        ``<=>`` is cosine *distance* -- 0 means identical -- so it is turned back into
        a similarity before anyone sees it. Handing a distance out as a score would
        inverse every ranking, and ``min_score`` would filter the wrong end.
        """
        distance = self._table.c.embedding.cosine_distance(needle)
        similarity = (1 - distance).label("score")

        statement = (
            select(self._table, similarity)
            .where(*self._conditions(scope))
            .order_by(distance)
            .limit(query.limit)
        )
        if query.min_score is not None:
            statement = statement.where(similarity >= query.min_score)

        async with self._gate, self._engine.connect() as conn:
            rows = (await conn.execute(statement)).mappings().all()
        return [self._to_memory(row, row["score"]) for row in rows]

    async def _search_in_python(
        self, query: SearchQuery, scope: Scope, needle: list[float]
    ) -> list[ProviderMemory]:
        """SQLite, and the label-filtered Postgres case.

        Labels live in a JSON column and are matched in Python, so the limit cannot be
        pushed into SQL without truncating before the filter runs.
        """
        rows = await self._rows_in_scope(scope)

        scored = []
        for row in rows:
            score = cosine(needle, list(row["embedding"]))
            if query.min_score is not None and score < query.min_score:
                continue
            scored.append((score, row))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [self._to_memory(row, score) for score, row in scored[: query.limit]]

    async def explain_search(self, query: SearchQuery, scope: Scope) -> str:
        """The query plan, so a test can prove the database did the work."""
        [needle] = await self._embedder.embed([query.query])
        distance = self._table.c.embedding.cosine_distance(needle)
        statement = (
            select(self._table.c.native_id)
            .where(*self._conditions(scope))
            .order_by(distance)
            .limit(query.limit)
        )
        async with self._gate, self._engine.connect() as conn:
            plan = await conn.execute(
                text(
                    "EXPLAIN "
                    + str(statement.compile(conn.engine, compile_kwargs={"literal_binds": True}))
                )
            )
            return "\n".join(row[0] for row in plan)

    async def list_scope(self, scope: Scope, limit: int) -> list[ProviderMemory]:
        """Everything held in a scope, newest first, no query involved.

        Ordered by ``created_at`` rather than by relevance because the question here
        is "what do you have on this person", and an unqueryable store is one a caller
        cannot audit, export or hand to a subject who asks.
        """
        rows = await self._rows_in_scope(scope, order_by_recent=True, limit=limit)
        return [self._to_memory(row, None) for row in rows]

    async def get(self, native_id: str) -> ProviderMemory | None:
        async with self._gate, self._engine.connect() as conn:
            found = await conn.execute(
                select(self._table).where(self._table.c.native_id == native_id)
            )
            row = found.mappings().one_or_none()
        return None if row is None else self._to_memory(row, None)

    # -- delete ---------------------------------------------------------------

    async def delete(self, native_id: str) -> bool:
        async with self._gate, self._engine.begin() as conn:
            result = await conn.execute(
                delete(self._table).where(self._table.c.native_id == native_id)
            )
        return result.rowcount > 0

    async def delete_scope(self, scope: Scope) -> int:
        if not scope.labels:
            # One statement, no id list: an erasure must not be the query that
            # discovers a bind-parameter ceiling.
            async with self._gate, self._engine.begin() as conn:
                result = await conn.execute(delete(self._table).where(*self._conditions(scope)))
            return result.rowcount

        # Labels live in JSON and are matched in Python, so this path does need ids --
        # in chunks, for the same reason.
        doomed = [row["native_id"] for row in await self._rows_in_scope(scope)]
        if not doomed:
            return 0
        async with self._gate, self._engine.begin() as conn:
            for start in range(0, len(doomed), _CHUNK):
                chunk = doomed[start : start + _CHUNK]
                await conn.execute(delete(self._table).where(self._table.c.native_id.in_(chunk)))
        return len(doomed)

    # -- internals ------------------------------------------------------------

    def _conditions(self, scope: Scope) -> list[Any]:
        """Every dimension the caller set is applied. Dropping one would return
        another tenant's or another end-user's memories, so the filter follows
        ``scope.dims()`` rather than whichever columns happen to be indexed."""
        conditions = [self._table.c.subject == scope.subject]
        if scope.tenant:
            conditions.append(self._table.c.tenant == scope.tenant)
        if scope.agent:
            conditions.append(self._table.c.agent == scope.agent)
        if scope.session:
            conditions.append(self._table.c.session == scope.session)
        return conditions

    async def _rows_in_scope(
        self, scope: Scope, *, order_by_recent: bool = False, limit: int | None = None
    ) -> list[dict[str, Any]]:
        statement = select(self._table).where(*self._conditions(scope))
        if order_by_recent:
            statement = statement.order_by(self._table.c.created_at.desc(), self._table.c.native_id)
        # A label filter is applied after the fetch, so the limit cannot be pushed
        # into SQL without silently truncating before the filter runs.
        if limit is not None and not scope.labels:
            statement = statement.limit(limit)

        async with self._gate, self._engine.connect() as conn:
            rows = (await conn.execute(statement)).mappings().all()

        if not scope.labels:
            return [dict(row) for row in rows]
        matched = [
            dict(row)
            for row in rows
            if all((row["labels"] or {}).get(k) == v for k, v in scope.labels.items())
        ]
        return matched[:limit] if limit is not None else matched

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
