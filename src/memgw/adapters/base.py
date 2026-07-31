"""The adapter contract.

An adapter never sees a gateway id. It speaks ``native_id`` and the catalog
bridges the two -- which is what keeps adapters thin, independently testable, and
unaffected when the catalog is added, moved or sharded.

Adapters are also responsible for refusing what they do not declare: an
undeclared search mode must raise :class:`~memgw.errors.UnsupportedCapability`
rather than quietly serving something else. The conformance suite checks both
directions, so a declaration that drifts from behaviour fails the build.

Every ``scope`` handed to an adapter carries ``scope.tenant``, stamped by the
gateway from the credential. It is an ordinary filter dimension and must be
applied like any other: subject ids are chosen by tenants, so an adapter that
filters only by subject serves two tenants the same row the moment they both name
an end-user ``u_1``. An adapter that cannot express it omits ``"tenant"`` from
``scope_dims`` and the gateway refuses to route to it -- which is a loud failure
rather than a silent shared store.
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

    async def health(self) -> HealthStatus:
        """Is the provider reachable."""
        ...

    async def self_test(self) -> HealthStatus:
        """Does the provider *work*: write something, read it back, clean up.

        Optional, and a different question from :meth:`health`. Zep stayed reachable
        for hours while accepting every write and building nothing, so every read
        returned empty and nothing raised -- a state no reachability check can see.
        ``memgw doctor --probe`` calls this where it exists.
        """
        ...

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

    async def list_scope(self, scope: Scope, limit: int) -> list[ProviderMemory]:
        """Everything in a scope, no query. Optional: declare ``supports_list=False``
        and omit it. Anything that can be searched can usually be listed, and a store
        that cannot be listed cannot be audited or handed to the subject who asks."""
        ...

    async def get(self, native_id: str) -> ProviderMemory | None: ...

    async def update(self, native_id: str, content: str) -> ProviderMemory: ...

    async def delete(self, native_id: str) -> bool: ...

    async def delete_scope(self, scope: Scope) -> int: ...
