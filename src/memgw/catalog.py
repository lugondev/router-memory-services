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

import hashlib
import secrets
import threading
import time
from dataclasses import dataclass
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


@dataclass(frozen=True)
class JournalRow:
    episode_id: str
    subject: str
    agent: str | None
    session: str | None
    payload: dict[str, Any]
    ingested_to: dict[str, list[str]]


@dataclass(frozen=True)
class RebindResult:
    provider: str
    orphaned_at: str | None
    orphaned_count: int


# -- catalog ------------------------------------------------------------------


class Catalog:
    def __init__(self, url: str = "sqlite+aiosqlite:///memgw.db") -> None:
        kwargs: dict[str, Any] = {}
        if url.endswith(":memory:"):
            # Without a shared pool every connection gets its own empty database.
            kwargs = {"poolclass": StaticPool, "connect_args": {"check_same_thread": False}}
        self._engine: AsyncEngine = create_async_engine(url, **kwargs)

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    async def init(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(metadata.create_all)

    async def close(self) -> None:
        await self._engine.dispose()

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
        now = _now()
        async with self._engine.begin() as conn:
            found = (
                await conn.execute(
                    select(memory_index.c.gateway_id).where(
                        memory_index.c.tenant_id == tenant,
                        memory_index.c.provider == provider,
                        memory_index.c.native_id == native_id,
                    )
                )
            ).scalar_one_or_none()

            if found is not None:
                await conn.execute(
                    update(memory_index)
                    .where(memory_index.c.gateway_id == found)
                    .values(
                        subject=scope.subject,
                        agent=scope.agent,
                        session=scope.session,
                        content_hash=_hash(content),
                        updated_at=now,
                        deleted_at=None,
                    )
                )
                return found

            gateway_id = new_id()
            await conn.execute(
                insert(memory_index).values(
                    gateway_id=gateway_id,
                    tenant_id=tenant,
                    provider=provider,
                    native_id=native_id,
                    subject=scope.subject,
                    agent=scope.agent,
                    session=scope.session,
                    content_hash=_hash(content),
                    created_at=now,
                    updated_at=now,
                    deleted_at=None,
                )
            )
            return gateway_id

    async def ensure(
        self, tenant: str, provider: str, native_id: str, scope: Scope, content: str
    ) -> CatalogRow:
        """Map a provider-native id to a gateway id *without disturbing what is stored*.

        Search hits go through here rather than :meth:`record`. The scope a caller
        searched with is often broader than the scope the memory was written with --
        recall across sessions is the whole point -- so writing the query's scope back
        over the stored one would quietly widen it and break the next session filter.
        """
        found = await self._find_by_native(tenant, provider, native_id)
        if found is not None:
            return found
        gateway_id = await self.record(tenant, provider, native_id, scope, content)
        return CatalogRow(
            gateway_id=gateway_id,
            tenant_id=tenant,
            provider=provider,
            native_id=native_id,
            subject=scope.subject,
            agent=scope.agent,
            session=scope.session,
        )

    async def _find_by_native(self, tenant: str, provider: str, native_id: str) -> CatalogRow | None:
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    select(memory_index).where(
                        memory_index.c.tenant_id == tenant,
                        memory_index.c.provider == provider,
                        memory_index.c.native_id == native_id,
                    )
                )
            ).mappings().one_or_none()
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

    async def resolve_gateway_id(self, tenant: str, gateway_id: str) -> CatalogRow | None:
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    select(memory_index).where(
                        memory_index.c.gateway_id == gateway_id,
                        memory_index.c.tenant_id == tenant,
                        memory_index.c.deleted_at.is_(None),
                    )
                )
            ).mappings().one_or_none()
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
        async with self._engine.begin() as conn:
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
        async with self._engine.begin() as conn:
            result = await conn.execute(
                update(memory_index).where(*conditions).values(deleted_at=_now())
            )
        return result.rowcount

    async def live_count(self, tenant: str, subject: str, provider: str) -> int:
        async with self._engine.connect() as conn:
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
        async with self._engine.connect() as conn:
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
            async with self._engine.begin() as conn:
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

        async with self._engine.begin() as conn:
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
        async with self._engine.begin() as conn:
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

    async def journal_rows(self, tenant: str, subject: str) -> list[JournalRow]:
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    select(episode_journal)
                    .where(
                        episode_journal.c.tenant_id == tenant,
                        episode_journal.c.subject == subject,
                    )
                    .order_by(episode_journal.c.episode_id)
                )
            ).mappings().all()
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
