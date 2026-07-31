"""The gateway's own bookkeeping.

Three tables, and each earns its place:

``memory_index``   maps a gateway-issued id to ``(provider, native_id)``, so a
                   caller's identifiers survive a provider change and adapters
                   never have to know a gateway id exists.
``scope_binding``  records which provider an end-user lives on. Without it, a
                   request that names the wrong provider recalls nothing and
                   raises nothing -- the worst failure this system has.
``episode_journal`` keeps the raw material, opt-in per tenant. It is the only way
                   to move an end-user to a new provider when the old one has no
                   usable export, which is the common case.

SQLite by default; Postgres when ``DATABASE_URL`` says so.
"""

from __future__ import annotations

import asyncio
import hashlib
import secrets
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Index,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    delete,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import StaticPool

from memgw.types import Scope

# -- ids ----------------------------------------------------------------------

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_id_lock = threading.Lock()
_last_ms = 0
_last_rand = 0


def _b32(value: int, length: int) -> str:
    out = []
    for _ in range(length):
        out.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(out))


def new_id(prefix: str = "mg") -> str:
    """A monotonic ULID.

    Monotonic on purpose: two ids minted in the same millisecond still sort by
    creation order, so ``ORDER BY gateway_id`` is a stable chronology and pagination
    does not shuffle.
    """
    global _last_ms, _last_rand
    with _id_lock:
        ms = int(time.time() * 1000)
        if ms == _last_ms:
            _last_rand += 1
        else:
            _last_ms, _last_rand = ms, secrets.randbits(80)
        rand = _last_rand
    return f"{prefix}_{_b32(ms, 10)}{_b32(rand, 16)}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


#: Marks a native id the gateway has seen and deleted, as distinct from one it has
#: never seen. Reads drop it; a re-write revives it.
_GONE = object()

#: SQLite's default bind-parameter ceiling is 999 on builds still in the wild, and
#: an ``IN`` list is one parameter per element. Chunking keeps a large scope from
#: failing at exactly the moment it matters -- a bulk erasure.
_CHUNK = 500

#: Attempts at the map-or-insert transaction. One retry is enough: the second pass
#: finds whatever the winner wrote, so a third could only mean a different fault.
_RETRIES = 3


def _chunks(values: list[str], size: int = _CHUNK):
    for start in range(0, len(values), size):
        yield values[start : start + size]


# -- schema -------------------------------------------------------------------

metadata = MetaData()

memory_index = Table(
    "memory_index",
    metadata,
    Column("gateway_id", String(48), primary_key=True),
    Column("tenant_id", String(128), nullable=False),
    Column("provider", String(64), nullable=False),
    Column("native_id", String(512), nullable=False),
    Column("subject", String(256), nullable=False),
    Column("agent", String(256)),
    Column("session", String(256)),
    Column("content_hash", String(64), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("deleted_at", DateTime(timezone=True)),
    UniqueConstraint("tenant_id", "provider", "native_id", name="uq_memory_index_native"),
    Index("ix_memory_index_scope", "tenant_id", "subject", "provider"),
)

scope_binding = Table(
    "scope_binding",
    metadata,
    Column("tenant_id", String(128), primary_key=True),
    Column("subject", String(256), primary_key=True),
    Column("provider", String(64), nullable=False),
    Column("bound_at", DateTime(timezone=True), nullable=False),
    Column("migrated_from", String(64)),
)

episode_journal = Table(
    "episode_journal",
    metadata,
    Column("episode_id", String(48), primary_key=True),
    Column("tenant_id", String(128), nullable=False),
    Column("subject", String(256), nullable=False),
    Column("agent", String(256)),
    Column("session", String(256)),
    Column("payload", JSON, nullable=False),
    Column("ingested_to", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Index("ix_episode_journal_scope", "tenant_id", "subject"),
)


# -- rows ---------------------------------------------------------------------


@dataclass(frozen=True)
class CatalogRow:
    gateway_id: str
    tenant_id: str
    provider: str
    native_id: str
    subject: str
    agent: str | None
    session: str | None

    def scope(self) -> Scope:
        return Scope(subject=self.subject, agent=self.agent, session=self.session)


def _to_row(row: Any) -> CatalogRow:
    return CatalogRow(
        gateway_id=row["gateway_id"],
        tenant_id=row["tenant_id"],
        provider=row["provider"],
        native_id=row["native_id"],
        subject=row["subject"],
        agent=row["agent"],
        session=row["session"],
    )


@dataclass(frozen=True)
class JournalRow:
    episode_id: str
    subject: str
    agent: str | None
    session: str | None
    payload: dict[str, Any]
    ingested_to: dict[str, list[str]]


@dataclass(frozen=True)
class SchemaState:
    """Where the database is, versus where the code expects it to be."""

    current: str | None
    head: str | None

    @property
    def up_to_date(self) -> bool:
        return self.current == self.head

    def describe(self) -> str:
        if self.up_to_date:
            return f"at head ({self.head})"
        return (
            f"database is at {self.current or 'no revision'}, code expects {self.head}. "
            "Run `memgw migrate` -- until then any query may reference a column that "
            "is not there."
        )


@dataclass(frozen=True)
class RebindResult:
    provider: str
    orphaned_at: str | None
    orphaned_count: int


# -- catalog ------------------------------------------------------------------


class Catalog:
    def __init__(self, url: str = "sqlite+aiosqlite:///memgw.db") -> None:
        kwargs: dict[str, Any] = {}
        shared_connection = url.endswith(":memory:")
        if shared_connection:
            # Without a shared pool every connection gets its own empty database.
            kwargs = {"poolclass": StaticPool, "connect_args": {"check_same_thread": False}}
        self._engine: AsyncEngine = create_async_engine(url, **kwargs)

        #: StaticPool hands *the same* DBAPI connection to every caller, so two
        #: "concurrent transactions" are one transaction wearing two hats: a rollback
        #: in either undoes the other's committed work. Serialising is what makes that
        #: pool safe. A real pool gives out distinct connections and needs no gate --
        #: there, concurrency is settled by the unique constraint and a retry.
        self._gate: asyncio.Lock | None = asyncio.Lock() if shared_connection else None

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    async def init(self) -> None:
        """Bring an empty database up, and leave an existing one alone.

        A fresh database is created and stamped at head in one step -- replaying
        every migration to build tables that never existed is slower and no more
        correct. A database that already has tables is *not* migrated here: applying
        schema changes is a deploy decision with a rollback plan attached, not a side
        effect of a process starting. :meth:`schema_state` reports the mismatch and
        ``memgw migrate`` resolves it.
        """
        if await self._is_empty():
            async with self._write() as conn:
                await conn.run_sync(metadata.create_all)
            await self._stamp("head")
            return
        if await self._revision() is None:
            # Tables from before migrations existed. They match the first revision,
            # so record that rather than making the operator guess.
            await self._stamp("head")

    async def upgrade(self) -> None:
        """Run the migrations. What ``memgw migrate`` calls."""
        async with self._write() as conn:
            await conn.run_sync(self._alembic_upgrade)

    async def schema_state(self) -> SchemaState:
        head = self._head_revision()
        return SchemaState(current=await self._revision(), head=head)

    async def close(self) -> None:
        await self._engine.dispose()

    # -- migrations -----------------------------------------------------------

    @staticmethod
    def _alembic_config(connection: Any = None):
        from alembic.config import Config

        config = Config()
        config.set_main_option("script_location", str(Path(__file__).parent / "migrations"))
        if connection is not None:
            config.attributes["connection"] = connection
        return config

    def _head_revision(self) -> str | None:
        from alembic.script import ScriptDirectory

        return ScriptDirectory.from_config(self._alembic_config()).get_current_head()

    def _alembic_upgrade(self, connection) -> None:
        from alembic import command

        command.upgrade(self._alembic_config(connection), "head")

    def _alembic_stamp(self, connection, revision: str) -> None:
        from alembic import command

        command.stamp(self._alembic_config(connection), revision)

    async def _stamp(self, revision: str) -> None:
        async with self._write() as conn:
            await conn.run_sync(self._alembic_stamp, revision)

    async def _revision(self) -> str | None:
        from alembic.runtime.migration import MigrationContext

        async with self._read() as conn:
            return await conn.run_sync(
                lambda sync: MigrationContext.configure(sync).get_current_revision()
            )

    async def _is_empty(self) -> bool:
        from sqlalchemy import inspect

        async with self._read() as conn:
            names = await conn.run_sync(lambda sync: inspect(sync).get_table_names())
        return not [n for n in names if n != "alembic_version"]

    # -- connections ----------------------------------------------------------

    @asynccontextmanager
    async def _write(self) -> AsyncIterator[Any]:
        if self._gate is None:
            async with self._engine.begin() as conn:
                yield conn
            return
        async with self._gate, self._engine.begin() as conn:
            yield conn

    @asynccontextmanager
    async def _read(self) -> AsyncIterator[Any]:
        if self._gate is None:
            async with self._engine.connect() as conn:
                yield conn
            return
        async with self._gate, self._engine.connect() as conn:
            yield conn

    # -- memory_index ---------------------------------------------------------

    async def record(
        self, tenant: str, provider: str, native_id: str, scope: Scope, content: str
    ) -> str:
        """Issue (or reuse) the gateway id for a provider-native memory.

        Idempotent on ``(tenant, provider, native_id)`` so a search result can be
        mapped by calling this -- a hit the catalog never saw gets an id here rather
        than being dropped, which keeps a gateway pointed at a pre-existing provider
        store usable instead of half-blind.
        """
        [row] = await self.record_many(tenant, provider, [(native_id, content)], scope)
        return row.gateway_id

    async def record_many(
        self, tenant: str, provider: str, items: list[tuple[str, str]], scope: Scope
    ) -> list[CatalogRow]:
        """The write path: map a page of ids, reviving any the gateway had deleted.

        A provider that deduplicates can answer a fresh write with a native id this
        catalog buried earlier. That memory is genuinely back, so a write revives it --
        whereas a *read* of the same id stays dropped (:meth:`ensure_many`), because a
        read is just a provider that has not caught up with a delete yet.
        """
        return await self._map(tenant, provider, items, scope, revive=True)

    async def ensure(
        self, tenant: str, provider: str, native_id: str, scope: Scope, content: str
    ) -> CatalogRow | None:
        """Map one provider-native id to a gateway id. ``None`` when the gateway
        considers it deleted -- see :meth:`ensure_many`."""
        rows = await self.ensure_many(tenant, provider, [(native_id, content)], scope)
        return rows[0] if rows else None

    async def ensure_many(
        self,
        tenant: str,
        provider: str,
        items: list[tuple[str, str]],
        scope: Scope,
    ) -> list[CatalogRow]:
        """Map a page of ``(native_id, content)`` hits in one round trip.

        Three things this does that the obvious loop does not:

        *Batched.* A search returning fifty hits used to cost fifty sequential
        statements before the caller saw a single result.

        *Leaves stored scope alone.* The scope a caller searched with is usually
        broader than the scope each memory was written with -- recall across sessions
        is the whole point -- so the query's scope is used only for rows being seen
        for the first time, never written back over an existing one.

        *Drops what the gateway deleted.* A provider that has not yet propagated a
        delete keeps serving the memory; remapping it would hand back an id that
        ``GET`` immediately answers with a 404. Deleted means gone from reads.
        """
        return await self._map(tenant, provider, items, scope, revive=False)

    async def _map(
        self,
        tenant: str,
        provider: str,
        items: list[tuple[str, str]],
        scope: Scope,
        *,
        revive: bool,
    ) -> list[CatalogRow]:
        """Look up, insert what is missing, and return the mapping -- atomically.

        All of it happens in one transaction, because check-then-insert across two
        connections is a race by construction. When a genuinely concurrent writer wins
        the unique constraint, the transaction is rolled back and retried: on the
        second pass the lookup simply finds the row the winner wrote. Losing that race
        is ordinary, not exceptional, and must not reach the caller as a 500.
        """
        if not items:
            return []

        natives = [native for native, _ in items]
        for attempt in range(_RETRIES):
            try:
                async with self._write() as conn:
                    known = await self._find_by_native(conn, tenant, provider, natives)

                    missing = [
                        (native, content) for native, content in items if native not in known
                    ]
                    if missing:
                        known.update(await self._insert(conn, tenant, provider, missing, scope))

                    if revive:
                        doomed = [n for n in natives if known.get(n, _GONE) is _GONE]
                        if doomed:
                            await self._revive(conn, tenant, provider, doomed, scope, dict(items))
                            known.update(await self._find_by_native(conn, tenant, provider, doomed))

                    return [
                        row
                        for native in natives
                        if isinstance(row := known.get(native), CatalogRow)
                    ]
            except IntegrityError:
                if attempt == _RETRIES - 1:
                    raise
        raise AssertionError("unreachable")

    async def _insert(
        self,
        conn: Any,
        tenant: str,
        provider: str,
        missing: list[tuple[str, str]],
        scope: Scope,
    ) -> dict[str, CatalogRow]:
        now = _now()
        values = [
            {
                "gateway_id": new_id(),
                "tenant_id": tenant,
                "provider": provider,
                "native_id": native,
                "subject": scope.subject,
                "agent": scope.agent,
                "session": scope.session,
                "content_hash": _hash(content),
                "created_at": now,
                "updated_at": now,
                "deleted_at": None,
            }
            for native, content in missing
        ]
        await conn.execute(insert(memory_index), values)
        return {
            row["native_id"]: CatalogRow(
                gateway_id=row["gateway_id"],
                tenant_id=tenant,
                provider=provider,
                native_id=row["native_id"],
                subject=scope.subject,
                agent=scope.agent,
                session=scope.session,
            )
            for row in values
        }

    async def _revive(
        self,
        conn: Any,
        tenant: str,
        provider: str,
        natives: list[str],
        scope: Scope,
        contents: dict[str, str],
    ) -> None:
        """Re-writing the same native id is a write, and a write un-deletes."""
        now = _now()
        for native in natives:
            await conn.execute(
                update(memory_index)
                .where(
                    memory_index.c.tenant_id == tenant,
                    memory_index.c.provider == provider,
                    memory_index.c.native_id == native,
                )
                .values(
                    subject=scope.subject,
                    agent=scope.agent,
                    session=scope.session,
                    content_hash=_hash(contents.get(native, "")),
                    updated_at=now,
                    deleted_at=None,
                )
            )

    @staticmethod
    async def _find_by_native(
        conn: Any, tenant: str, provider: str, natives: list[str]
    ) -> dict[str, CatalogRow | object]:
        """Live rows as :class:`CatalogRow`; soft-deleted ones as the ``_GONE`` marker,
        so a caller can tell "never seen" from "deleted" without a second query."""
        if not natives:
            return {}
        found: dict[str, CatalogRow | object] = {}
        for chunk in _chunks(natives):
            rows = (
                (
                    await conn.execute(
                        select(memory_index).where(
                            memory_index.c.tenant_id == tenant,
                            memory_index.c.provider == provider,
                            memory_index.c.native_id.in_(chunk),
                        )
                    )
                )
                .mappings()
                .all()
            )
            for row in rows:
                found[row["native_id"]] = _GONE if row["deleted_at"] is not None else _to_row(row)
        return found

    async def resolve_gateway_id(self, tenant: str, gateway_id: str) -> CatalogRow | None:
        async with self._read() as conn:
            row = (
                (
                    await conn.execute(
                        select(memory_index).where(
                            memory_index.c.gateway_id == gateway_id,
                            memory_index.c.tenant_id == tenant,
                            memory_index.c.deleted_at.is_(None),
                        )
                    )
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            return None
        return CatalogRow(
            gateway_id=row["gateway_id"],
            tenant_id=row["tenant_id"],
            provider=row["provider"],
            native_id=row["native_id"],
            subject=row["subject"],
            agent=row["agent"],
            session=row["session"],
        )

    async def mark_deleted(self, tenant: str, gateway_id: str) -> bool:
        async with self._write() as conn:
            result = await conn.execute(
                update(memory_index)
                .where(
                    memory_index.c.gateway_id == gateway_id,
                    memory_index.c.tenant_id == tenant,
                    memory_index.c.deleted_at.is_(None),
                )
                .values(deleted_at=_now())
            )
        return result.rowcount > 0

    async def mark_scope_deleted(self, tenant: str, scope: Scope, provider: str) -> int:
        conditions = [
            memory_index.c.tenant_id == tenant,
            memory_index.c.provider == provider,
            memory_index.c.subject == scope.subject,
            memory_index.c.deleted_at.is_(None),
        ]
        if scope.agent:
            conditions.append(memory_index.c.agent == scope.agent)
        if scope.session:
            conditions.append(memory_index.c.session == scope.session)
        async with self._write() as conn:
            result = await conn.execute(
                update(memory_index).where(*conditions).values(deleted_at=_now())
            )
        return result.rowcount

    async def live_count(self, tenant: str, subject: str, provider: str) -> int:
        async with self._read() as conn:
            return (
                await conn.execute(
                    select(func.count())
                    .select_from(memory_index)
                    .where(
                        memory_index.c.tenant_id == tenant,
                        memory_index.c.subject == subject,
                        memory_index.c.provider == provider,
                        memory_index.c.deleted_at.is_(None),
                    )
                )
            ).scalar_one()

    # -- scope_binding --------------------------------------------------------

    async def get_binding(self, tenant: str, subject: str) -> str | None:
        async with self._read() as conn:
            return (
                await conn.execute(
                    select(scope_binding.c.provider).where(
                        scope_binding.c.tenant_id == tenant,
                        scope_binding.c.subject == subject,
                    )
                )
            ).scalar_one_or_none()

    async def bind(self, tenant: str, subject: str, provider: str) -> str:
        """Bind on first write. Never moves an existing binding -- that is ``rebind``,
        and it is a deliberate act with consequences the caller must be told about."""
        try:
            async with self._write() as conn:
                await conn.execute(
                    insert(scope_binding).values(
                        tenant_id=tenant, subject=subject, provider=provider, bound_at=_now()
                    )
                )
            return provider
        except IntegrityError:
            existing = await self.get_binding(tenant, subject)
            return existing or provider

    async def rebind(self, tenant: str, subject: str, provider: str) -> RebindResult:
        previous = await self.get_binding(tenant, subject)
        orphaned = 0
        if previous is not None and previous != provider:
            orphaned = await self.live_count(tenant, subject, previous)

        async with self._write() as conn:
            await conn.execute(
                delete(scope_binding).where(
                    scope_binding.c.tenant_id == tenant, scope_binding.c.subject == subject
                )
            )
            await conn.execute(
                insert(scope_binding).values(
                    tenant_id=tenant,
                    subject=subject,
                    provider=provider,
                    bound_at=_now(),
                    migrated_from=previous,
                )
            )
        return RebindResult(
            provider=provider,
            orphaned_at=previous if previous != provider else None,
            orphaned_count=orphaned,
        )

    # -- episode_journal ------------------------------------------------------

    async def journal(
        self,
        tenant: str,
        scope: Scope,
        payload: dict[str, Any],
        ingested_to: dict[str, list[str]],
    ) -> str:
        episode_id = new_id("ep")
        async with self._write() as conn:
            await conn.execute(
                insert(episode_journal).values(
                    episode_id=episode_id,
                    tenant_id=tenant,
                    subject=scope.subject,
                    agent=scope.agent,
                    session=scope.session,
                    payload=payload,
                    ingested_to=ingested_to,
                    created_at=_now(),
                )
            )
        return episode_id

    async def delete_journal(self, tenant: str, scope: Scope) -> int:
        """Erasure has to reach in here too.

        ``memory_index`` holds hashes and ids; the journal holds the raw transcript --
        the most sensitive thing the gateway ever stores, and the thing a subject
        deletion is usually *about*. Soft-deleting the index while leaving the episodes
        behind would make ``delete_scope`` a promise the gateway does not keep.
        """
        conditions = [
            episode_journal.c.tenant_id == tenant,
            episode_journal.c.subject == scope.subject,
        ]
        if scope.agent:
            conditions.append(episode_journal.c.agent == scope.agent)
        if scope.session:
            conditions.append(episode_journal.c.session == scope.session)
        async with self._write() as conn:
            result = await conn.execute(delete(episode_journal).where(*conditions))
        return result.rowcount

    async def journal_rows(self, tenant: str, subject: str) -> list[JournalRow]:
        async with self._read() as conn:
            rows = (
                (
                    await conn.execute(
                        select(episode_journal)
                        .where(
                            episode_journal.c.tenant_id == tenant,
                            episode_journal.c.subject == subject,
                        )
                        .order_by(episode_journal.c.episode_id)
                    )
                )
                .mappings()
                .all()
            )
        return [
            JournalRow(
                episode_id=row["episode_id"],
                subject=row["subject"],
                agent=row["agent"],
                session=row["session"],
                payload=row["payload"],
                ingested_to=row["ingested_to"],
            )
            for row in rows
        ]
