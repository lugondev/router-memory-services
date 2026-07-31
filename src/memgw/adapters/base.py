"""The adapter contract.

An adapter never sees a gateway id. It speaks ``native_id`` and the catalog
bridges the two -- which is what keeps adapters thin, independently testable, and
unaffected when the catalog is added, moved or sharded.

Adapters are also responsible for refusing what they do not declare: an
undeclared search mode must raise :class:`~memgw.errors.UnsupportedCapability`
rather than quietly serving something else. The conformance suite checks both
directions, so a declaration that drifts from behaviour fails the build.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from memgw.capabilities import Capabilities
from memgw.types import Episode, HealthStatus, ProviderMemory, Scope, SearchQuery


@runtime_checkable
class MemoryAdapter(Protocol):
    name: str

    def capabilities(self) -> Capabilities:
        """What *this configured instance* can do. Not a class constant."""
        ...

    async def health(self) -> HealthStatus: ...

    async def ingest(self, episode: Episode, scope: Scope) -> list[ProviderMemory]:
        """Raw material in; the provider extracts. Raises UnsupportedCapability when
        ``supports_ingest`` is false."""
        ...

    async def upsert(self, facts: list[str], scope: Scope) -> list[ProviderMemory]:
        """Ready-made facts in; the provider only stores and indexes."""
        ...

    async def search(self, query: SearchQuery, scope: Scope) -> list[ProviderMemory]:
        """Reads always carry the full mapped scope. Forwarding a partial filter
        returns an empty result with no error on at least one provider."""
        ...

    async def get(self, native_id: str) -> ProviderMemory | None: ...

    async def update(self, native_id: str, content: str) -> ProviderMemory: ...

    async def delete(self, native_id: str) -> bool: ...

    async def delete_scope(self, scope: Scope) -> int: ...
