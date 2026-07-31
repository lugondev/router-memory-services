"""The verbs.

Everything provider-neutral lives here: resolve, gate on capability, call one
adapter, bridge native ids to gateway ids, and say plainly what was degraded or
unavailable. Both entry points -- the HTTP gateway and the embedded client --
are thin skins over this class.

One rule runs through every method: **the tenant comes from the credential and is
stamped onto the scope on the way down** (:meth:`Scope.under`). Adapters see it as
an ordinary scope dimension and must filter on it, because subject ids are chosen
by tenants and two tenants both naming an end-user ``u_1`` would otherwise share
a row on any provider instance they share.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from memgw.adapters.base import MemoryAdapter
from memgw.capabilities import Capabilities
from memgw.catalog import Catalog, CatalogRow, RebindResult
from memgw.degrade import (
    assert_as_of_supported,
    assert_delete_supported,
    assert_scope_supported,
    resolve_mode,
)
from memgw.errors import (
    GatewayError,
    InvalidRequest,
    MemoryNotFound,
    NotImplementedYet,
    ProviderError,
    ProviderUnhealthy,
    UnsupportedCapability,
)
from memgw.observability import observe
from memgw.router import resolve_provider
from memgw.types import (
    Episode,
    MemoryRecord,
    ProviderMemory,
    Scope,
    SearchMode,
    SearchQuery,
)

RebindStrategy = Literal["fresh_start", "migrate"]

#: How many memories a scope listing returns per call when the caller names no limit.
DEFAULT_LIST_LIMIT = 100


class WriteResult(BaseModel):
    """What a write returns.

    ``provider`` is here rather than inferred from ``results[0]`` because an
    extraction that kept nothing still went somewhere, and a caller reconciling its
    own records needs to know where.
    """

    results: list[MemoryRecord]
    provider: str


class SearchResult(BaseModel):
    results: list[MemoryRecord]
    provider: str
    degraded: bool = False
    requested: SearchMode | None = None
    served: SearchMode | None = None
    lost: list[str] = Field(default_factory=list)
    provider_unavailable: bool = False


@dataclass(frozen=True)
class ProviderStatus:
    name: str
    healthy: bool
    detail: str | None
    capabilities: Capabilities


class MemoryCore:
    def __init__(
        self,
        *,
        catalog: Catalog,
        providers: dict[str, MemoryAdapter],
        default_provider: str | None = None,
        journal_enabled: bool = False,
    ) -> None:
        self.catalog = catalog
        self.providers = providers
        self.default_provider = default_provider
        self.journal_enabled = journal_enabled

    # -- introspection --------------------------------------------------------

    def adapter(self, provider: str) -> MemoryAdapter:
        try:
            return self.providers[provider]
        except KeyError:
            raise InvalidRequest(
                f"provider {provider!r} is not configured",
                code="unknown_provider",
                details={"available": sorted(self.providers)},
            ) from None

    def capabilities(self, provider: str | None = None) -> Capabilities:
        name = provider or self.default_provider
        if name is None:
            raise InvalidRequest(
                "no provider named and no tenant default configured",
                code="no_provider_resolved",
            )
        return self.adapter(name).capabilities()

    async def providers_status(self) -> list[ProviderStatus]:
        out = []
        for name, adapter in sorted(self.providers.items()):
            try:
                health = await adapter.health()
                healthy, detail = health.ok, health.detail
            except Exception as exc:  # noqa: BLE001 -- a status page must never 500
                healthy, detail = False, str(exc)
            out.append(
                ProviderStatus(
                    name=name,
                    healthy=healthy,
                    detail=detail,
                    capabilities=adapter.capabilities(),
                )
            )
        return out

    # -- write ----------------------------------------------------------------

    async def ingest(
        self, tenant: str, episode: Episode, scope: Scope, *, provider: str | None = None
    ) -> WriteResult:
        with observe("ingest", tenant=tenant, subject=scope.subject) as event:
            name, adapter, bound = await self._prepare(tenant, scope, provider)
            event["provider"] = name

            if not adapter.capabilities().supports_ingest:
                raise UnsupportedCapability(
                    "provider cannot extract memories from raw episodes",
                    details={"provider": name, "hint": "send ready-made facts to :upsert"},
                )

            memories = await self._call(adapter, adapter.ingest(episode, bound), name)
            records = await self._records(tenant, name, memories, scope)
            await self.catalog.bind(tenant, scope.subject, name)

            if self.journal_enabled:
                await self.catalog.journal(
                    tenant,
                    scope,
                    episode.model_dump(mode="json", exclude_none=True),
                    {name: [memory.native_id for memory in memories]},
                )
            event["count"] = len(records)
            return WriteResult(results=records, provider=name)

    async def upsert(
        self, tenant: str, facts: list[str], scope: Scope, *, provider: str | None = None
    ) -> WriteResult:
        """Ready-made facts in, no extraction. The path a migration replays through,
        and the one a caller uses when it already knows what is worth remembering."""
        if not facts:
            raise InvalidRequest("upsert needs at least one fact", code="invalid_request")

        with observe("upsert", tenant=tenant, subject=scope.subject) as event:
            name, adapter, bound = await self._prepare(tenant, scope, provider)
            event["provider"] = name

            if not adapter.capabilities().supports_upsert:
                raise UnsupportedCapability(
                    "provider cannot store ready-made facts",
                    details={"provider": name},
                )

            memories = await self._call(adapter, adapter.upsert(facts, bound), name)
            records = await self._records(tenant, name, memories, scope)
            await self.catalog.bind(tenant, scope.subject, name)
            event["count"] = len(records)
            return WriteResult(results=records, provider=name)

    async def update(self, tenant: str, gateway_id: str, content: str) -> MemoryRecord:
        with observe("update", tenant=tenant) as event:
            row = await self._row(tenant, gateway_id)
            event["subject"] = row.subject
            event["provider"] = row.provider
            adapter = self.adapter(row.provider)
            if not adapter.capabilities().supports_update:
                raise UnsupportedCapability("provider cannot update a stored memory")

            memory = await self._call(adapter, adapter.update(row.native_id, content), row.provider)
            await self.catalog.record(tenant, row.provider, memory.native_id, row.scope(), content)
            return self._to_record(row.gateway_id, row.provider, memory, row.scope())

    # -- read -----------------------------------------------------------------

    async def search(
        self, tenant: str, query: SearchQuery, scope: Scope, *, provider: str | None = None
    ) -> SearchResult:
        with observe("search", tenant=tenant, subject=scope.subject) as event:
            name, adapter, bound = await self._prepare(tenant, scope, provider)
            caps = adapter.capabilities()
            event["provider"] = name
            event["mode"] = query.mode

            # Capability shortfalls are raised before any fail_open handling: not
            # being able to do graph search is a misconfiguration, not an outage, and
            # hiding it behind an empty result would make it permanent and invisible.
            if query.as_of is not None:
                assert_as_of_supported(caps)
            decision = resolve_mode(query.mode, caps, query.on_unsupported)

            effective = query.model_copy(
                update={"mode": decision.served, "limit": min(query.limit, caps.max_limit)}
            )

            try:
                memories = await adapter.search(effective, bound)
            except (UnsupportedCapability, InvalidRequest):
                raise
            except Exception as exc:  # noqa: BLE001 -- an outage, which fail_open covers
                if query.fail_open:
                    event["outcome_detail"] = "fail_open"
                    event["count"] = 0
                    return SearchResult(results=[], provider=name, provider_unavailable=True)
                raise await self._failure(adapter, name, exc) from exc

            records = await self._map_hits(tenant, name, memories, scope)
            event["count"] = len(records)
            return SearchResult(
                results=records,
                provider=name,
                degraded=decision.degraded,
                requested=query.mode,
                served=decision.served,
                lost=decision.lost,
            )

    async def list_scope(
        self,
        tenant: str,
        scope: Scope,
        *,
        limit: int = DEFAULT_LIST_LIMIT,
        provider: str | None = None,
    ) -> list[MemoryRecord]:
        """Read a scope back without a query.

        Search answers "what is relevant to this"; there was no way to ask "what do
        you hold on this person" -- which is what an export, a support ticket and a
        subject access request all actually need.
        """
        with observe("list", tenant=tenant, subject=scope.subject) as event:
            name, adapter, bound = await self._prepare(tenant, scope, provider)
            event["provider"] = name

            lister = getattr(adapter, "list_scope", None)
            if lister is None or not adapter.capabilities().supports_list:
                raise UnsupportedCapability(
                    "provider cannot enumerate a scope",
                    details={"provider": name},
                )

            capped = min(limit, adapter.capabilities().max_limit)
            memories = await self._call(adapter, lister(bound, capped), name)
            records = await self._map_hits(tenant, name, memories, scope)
            event["count"] = len(records)
            return records

    async def get(self, tenant: str, gateway_id: str) -> MemoryRecord:
        with observe("get", tenant=tenant) as event:
            row = await self._row(tenant, gateway_id)
            event["subject"] = row.subject
            event["provider"] = row.provider
            adapter = self.adapter(row.provider)
            memory = await self._call(adapter, adapter.get(row.native_id), row.provider)
            if memory is None:
                raise MemoryNotFound(f"{gateway_id} is no longer present at {row.provider}")
            return self._to_record(row.gateway_id, row.provider, memory, row.scope())

    # -- delete ---------------------------------------------------------------

    async def delete(self, tenant: str, gateway_id: str) -> None:
        with observe("delete", tenant=tenant) as event:
            row = await self._row(tenant, gateway_id)
            event["subject"] = row.subject
            event["provider"] = row.provider
            adapter = self.adapter(row.provider)
            assert_delete_supported(adapter.capabilities())

            await self._call(adapter, adapter.delete(row.native_id), row.provider)
            await self.catalog.mark_deleted(tenant, gateway_id)

    async def delete_scope(self, tenant: str, scope: Scope, *, provider: str | None = None) -> int:
        with observe("delete_scope", tenant=tenant, subject=scope.subject) as event:
            name, adapter, bound = await self._prepare(tenant, scope, provider)
            event["provider"] = name
            assert_delete_supported(adapter.capabilities(), by_scope=True)

            removed = await self._call(adapter, adapter.delete_scope(bound), name)
            await self.catalog.mark_scope_deleted(tenant, scope, name)
            # The raw episodes are the most sensitive thing here and the thing an
            # erasure is usually about. Index-only deletion would leave them behind.
            event["episodes_erased"] = await self.catalog.delete_journal(tenant, scope)
            event["count"] = removed
            return removed

    # -- binding --------------------------------------------------------------

    async def rebind(
        self, tenant: str, subject: str, provider: str, *, strategy: RebindStrategy = "fresh_start"
    ) -> RebindResult:
        with observe("rebind", tenant=tenant, subject=subject) as event:
            event["provider"] = provider
            if strategy == "migrate":
                raise NotImplementedYet(
                    "migrating an end-user's memories lands with the migration engine; "
                    "fresh_start strands them at the old provider and says so"
                )
            self.adapter(provider)  # refuse to bind to a provider that is not configured
            result = await self.catalog.rebind(tenant, subject, provider)
            event["orphaned"] = result.orphaned_count
            return result

    # -- internals ------------------------------------------------------------

    async def _prepare(
        self, tenant: str, scope: Scope, asserted: str | None
    ) -> tuple[str, MemoryAdapter, Scope]:
        """Resolve the provider, check the scope against it, and stamp the tenant.

        Every verb starts here, which is what makes "the tenant reaches the adapter"
        a property of the gateway rather than a thing each verb has to remember.
        """
        name = await resolve_provider(
            self.catalog,
            tenant,
            scope.subject,
            default_provider=self.default_provider,
            asserted=asserted,
        )
        adapter = self.adapter(name)
        bound = scope.under(tenant)
        assert_scope_supported(bound, adapter.capabilities())
        return name, adapter, bound

    async def _row(self, tenant: str, gateway_id: str) -> CatalogRow:
        row = await self.catalog.resolve_gateway_id(tenant, gateway_id)
        if row is None:
            # Also the answer for another tenant's id: a 403 would confirm it exists.
            raise MemoryNotFound(f"no memory {gateway_id!r}")
        return row

    async def _records(
        self, tenant: str, provider: str, memories: list[ProviderMemory], scope: Scope
    ) -> list[MemoryRecord]:
        """Mint gateway ids for a write, in one round trip rather than one per fact.

        Uses the reviving path: a provider that deduplicates can answer a fresh write
        with a native id this gateway deleted earlier, and that memory is genuinely
        back -- dropping it would lose a write the caller just made.
        """
        if not memories:
            return []
        rows = await self.catalog.record_many(
            tenant,
            provider,
            [(memory.native_id, memory.content) for memory in memories],
            scope,
        )
        by_native = {row.native_id: row for row in rows}
        return [
            self._to_record(by_native[memory.native_id].gateway_id, provider, memory, scope)
            for memory in memories
            if memory.native_id in by_native
        ]

    async def _map_hits(
        self, tenant: str, provider: str, memories: list[ProviderMemory], scope: Scope
    ) -> list[MemoryRecord]:
        """Map provider hits onto gateway ids, keeping the provider's ranking.

        Hits the catalog considers deleted are dropped rather than remapped: a provider
        that has not yet propagated a delete would otherwise hand back an id that
        ``GET`` answers with a 404 in the same breath.

        The query's scope is passed only so hits this catalog has never seen can be
        given one; ``ensure_many`` never writes it over a scope already stored, because
        a broad recall would otherwise widen every memory it touched.
        """
        if not memories:
            return []
        rows = await self.catalog.ensure_many(
            tenant,
            provider,
            [(memory.native_id, memory.content) for memory in memories],
            scope,
        )
        by_native = {row.native_id: row for row in rows}
        return [
            self._to_record(row.gateway_id, provider, memory, row.scope())
            for memory in memories
            if (row := by_native.get(memory.native_id)) is not None
        ]

    @staticmethod
    def _to_record(
        gateway_id: str, provider: str, memory: ProviderMemory, scope: Scope
    ) -> MemoryRecord:
        return MemoryRecord(
            id=gateway_id,
            provider=provider,
            native_id=memory.native_id,
            content=memory.content,
            scope=scope.without_tenant(),
            score=memory.score,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            valid_from=memory.valid_from,
            valid_to=memory.valid_to,
            provider_raw=memory.raw,
        )

    async def _call(self, adapter: MemoryAdapter, awaitable: Any, name: str) -> Any:
        """Run one adapter call, translating transport failures. Writes never fail
        open, so there is no leniency here -- only classification."""
        try:
            return await awaitable
        except GatewayError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise await self._failure(adapter, name, exc) from exc

    @staticmethod
    async def _failure(adapter: MemoryAdapter, name: str, exc: Exception) -> GatewayError:
        """503 when the provider is down, 424 when a healthy provider refused the call.

        Health is only consulted on the error path, so the happy path pays nothing for
        the distinction.
        """
        try:
            health = await adapter.health()
            down = not health.ok
        except Exception:  # noqa: BLE001
            down = True

        if down:
            return ProviderUnhealthy(
                f"provider {name!r} is unavailable", details={"provider": name, "cause": str(exc)}
            )
        return ProviderError(
            f"provider {name!r} failed the call",
            details={"provider": name, "cause": str(exc), "retryable": True},
        )
