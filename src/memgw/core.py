"""The verbs.

Everything provider-neutral lives here: resolve, gate on capability, call one
adapter, bridge native ids to gateway ids, and say plainly what was degraded or
unavailable. Both entry points -- the HTTP gateway and the embedded client --
are thin skins over this class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from memgw.adapters.base import MemoryAdapter
from memgw.capabilities import Capabilities
from memgw.catalog import Catalog, RebindResult
from memgw.degrade import assert_delete_supported, assert_scope_supported, resolve_mode
from memgw.errors import (
    GatewayError,
    InvalidRequest,
    MemoryNotFound,
    NotImplementedYet,
    ProviderError,
    ProviderUnhealthy,
    UnsupportedCapability,
)
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
    ) -> list[MemoryRecord]:
        name = await self._resolve(tenant, scope.subject, provider)
        adapter = self.adapter(name)
        caps = adapter.capabilities()

        assert_scope_supported(scope, caps)
        if not caps.supports_ingest:
            raise UnsupportedCapability(
                "provider cannot extract memories from raw episodes",
                details={"provider": name, "hint": "send ready-made facts once upsert ships"},
            )

        memories = await self._call(adapter, adapter.ingest(episode, scope))
        records = [await self._record(tenant, name, memory, scope) for memory in memories]

        # Bind on the first write only; bind() will not move an existing binding.
        await self.catalog.bind(tenant, scope.subject, name)

        if self.journal_enabled:
            await self.catalog.journal(
                tenant,
                scope,
                episode.model_dump(mode="json", exclude_none=True),
                {name: [memory.native_id for memory in memories]},
            )
        return records

    async def upsert(
        self, tenant: str, facts: list[str], scope: Scope, *, provider: str | None = None
    ) -> list[MemoryRecord]:
        """Reserved. The fact path is what migration replays, so it is specified now
        and returns 501 rather than being added to a published API later."""
        del tenant, facts, scope, provider
        raise NotImplementedYet("upsert lands with the migration engine")

    async def update(self, tenant: str, gateway_id: str, content: str) -> MemoryRecord:
        row = await self._row(tenant, gateway_id)
        adapter = self.adapter(row.provider)
        if not adapter.capabilities().supports_update:
            raise UnsupportedCapability("provider cannot update a stored memory")

        memory = await self._call(adapter, adapter.update(row.native_id, content))
        await self.catalog.record(tenant, row.provider, memory.native_id, row.scope(), content)
        return self._to_record(row.gateway_id, row.provider, memory, row.scope())

    # -- read -----------------------------------------------------------------

    async def search(
        self, tenant: str, query: SearchQuery, scope: Scope, *, provider: str | None = None
    ) -> SearchResult:
        name = await self._resolve(tenant, scope.subject, provider)
        adapter = self.adapter(name)
        caps = adapter.capabilities()

        assert_scope_supported(scope, caps)
        # A capability shortfall is raised before any fail_open handling: not being
        # able to do graph search is a misconfiguration, not an outage, and hiding it
        # behind an empty result would make it permanent and invisible.
        decision = resolve_mode(query.mode, caps, query.on_unsupported)

        effective = query.model_copy(
            update={"mode": decision.served, "limit": min(query.limit, caps.max_limit)}
        )

        try:
            memories = await adapter.search(effective, scope)
        except GatewayError:
            raise
        except Exception as exc:  # noqa: BLE001 -- provider transport failure
            if query.fail_open:
                return SearchResult(results=[], provider=name, provider_unavailable=True)
            raise await self._failure(adapter, name, exc) from exc

        results = []
        for memory in memories:
            row = await self.catalog.ensure(tenant, name, memory.native_id, scope, memory.content)
            results.append(self._to_record(row.gateway_id, name, memory, row.scope()))

        return SearchResult(
            results=results,
            provider=name,
            degraded=decision.degraded,
            requested=query.mode,
            served=decision.served,
            lost=decision.lost,
        )

    async def get(self, tenant: str, gateway_id: str) -> MemoryRecord:
        row = await self._row(tenant, gateway_id)
        adapter = self.adapter(row.provider)
        memory = await self._call(adapter, adapter.get(row.native_id))
        if memory is None:
            raise MemoryNotFound(f"{gateway_id} is no longer present at {row.provider}")
        return self._to_record(row.gateway_id, row.provider, memory, row.scope())

    # -- delete ---------------------------------------------------------------

    async def delete(self, tenant: str, gateway_id: str) -> None:
        row = await self._row(tenant, gateway_id)
        adapter = self.adapter(row.provider)
        assert_delete_supported(adapter.capabilities())

        await self._call(adapter, adapter.delete(row.native_id))
        await self.catalog.mark_deleted(tenant, gateway_id)

    async def delete_scope(self, tenant: str, scope: Scope, *, provider: str | None = None) -> int:
        name = await self._resolve(tenant, scope.subject, provider)
        adapter = self.adapter(name)
        caps = adapter.capabilities()

        assert_scope_supported(scope, caps)
        assert_delete_supported(caps, by_scope=True)

        removed = await self._call(adapter, adapter.delete_scope(scope))
        await self.catalog.mark_scope_deleted(tenant, scope, name)
        return removed

    # -- binding --------------------------------------------------------------

    async def rebind(
        self, tenant: str, subject: str, provider: str, *, strategy: RebindStrategy = "fresh_start"
    ) -> RebindResult:
        if strategy == "migrate":
            raise NotImplementedYet(
                "migrating an end-user's memories lands with the migration engine; "
                "fresh_start strands them at the old provider and says so"
            )
        self.adapter(provider)  # refuse to bind to a provider that is not configured
        return await self.catalog.rebind(tenant, subject, provider)

    # -- internals ------------------------------------------------------------

    async def _resolve(self, tenant: str, subject: str, asserted: str | None) -> str:
        return await resolve_provider(
            self.catalog,
            tenant,
            subject,
            default_provider=self.default_provider,
            asserted=asserted,
        )

    async def _row(self, tenant: str, gateway_id: str):
        row = await self.catalog.resolve_gateway_id(tenant, gateway_id)
        if row is None:
            # Also the answer for another tenant's id: a 403 would confirm it exists.
            raise MemoryNotFound(f"no memory {gateway_id!r}")
        return row

    async def _record(
        self, tenant: str, provider: str, memory: ProviderMemory, scope: Scope
    ) -> MemoryRecord:
        gateway_id = await self.catalog.record(
            tenant, provider, memory.native_id, scope, memory.content
        )
        return self._to_record(gateway_id, provider, memory, scope)

    @staticmethod
    def _to_record(
        gateway_id: str, provider: str, memory: ProviderMemory, scope: Scope
    ) -> MemoryRecord:
        return MemoryRecord(
            id=gateway_id,
            provider=provider,
            native_id=memory.native_id,
            content=memory.content,
            scope=scope,
            score=memory.score,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            valid_from=memory.valid_from,
            valid_to=memory.valid_to,
            provider_raw=memory.raw,
        )

    async def _call(self, adapter: MemoryAdapter, awaitable: Any) -> Any:
        """Run one adapter call, translating transport failures. Writes never fail
        open, so there is no leniency here -- only classification."""
        try:
            return await awaitable
        except GatewayError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise await self._failure(adapter, getattr(adapter, "name", "?"), exc) from exc

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
