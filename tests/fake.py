"""A dict-backed adapter that exists only to prove the conformance suite bites.

Not shipped in ``src/``. It is the control specimen: if the suite passes against a
deliberately broken FakeAdapter, the suite is worthless.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from memgw.capabilities import Capabilities
from memgw.errors import UnsupportedCapability
from memgw.types import Episode, HealthStatus, ProviderMemory, Scope, SearchQuery


def default_caps(**over) -> Capabilities:
    base = dict(
        supports_ingest=True,
        supports_upsert=True,
        supports_update=True,
        supports_delete=True,
        supports_delete_by_scope=True,
        supports_list=True,
        search_modes=["semantic"],
        supports_score=True,
        max_limit=100,
        scope_dims=["tenant", "subject", "agent", "session"],
        supports_labels=True,
        memory_model="flat_facts",
        dedup="none",
        supports_export=True,
        supports_import=True,
        consistency="read_after_write",
        metered_externally=False,
    )
    base.update(over)
    return Capabilities(**base)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class FakeAdapter:
    name = "fake"

    def __init__(
        self,
        caps: Capabilities | None = None,
        *,
        broken_scope_filter: bool = False,
        write_delay: float = 0.0,
        healthy: bool = True,
        refuse_modes: set[str] | None = None,
        extract_nothing: bool = False,
        ignore_dims: set[str] | None = None,
    ) -> None:
        self._caps = caps or default_caps()
        self._broken_scope_filter = broken_scope_filter
        #: Dimensions this adapter declares and then quietly does not filter on --
        #: the drift the suite exists to catch, one dimension at a time.
        self._ignore_dims = ignore_dims or set()
        #: Most turns are not worth remembering; an extractor that keeps nothing is
        #: an ordinary outcome, not an error.
        self._extract_nothing = extract_nothing
        #: Modes the implementation refuses regardless of what capabilities() claims.
        #: Lets a test build an adapter whose declaration drifts from its behaviour.
        self._refuse_modes = refuse_modes or set()
        self._write_delay = write_delay
        self._healthy = healthy
        self._rows: dict[str, tuple[Scope, str, datetime]] = {}
        self._visible: set[str] = set()
        self._seq = 0

    # -- introspection --------------------------------------------------------

    def capabilities(self) -> Capabilities:
        return self._caps

    async def health(self) -> HealthStatus:
        return HealthStatus(ok=self._healthy, detail=None if self._healthy else "down")

    # -- write ----------------------------------------------------------------

    async def ingest(self, episode: Episode, scope: Scope) -> list[ProviderMemory]:
        if not self._caps.supports_ingest:
            raise UnsupportedCapability("fake adapter has no extractor configured")
        if self._extract_nothing:
            self._require_up()
            return []
        return await self.upsert([episode.as_text()], scope)

    async def upsert(self, facts: list[str], scope: Scope) -> list[ProviderMemory]:
        self._require_up()
        out = []
        for fact in facts:
            self._seq += 1
            native_id = f"fake-{self._seq}"
            self._rows[native_id] = (scope, fact, _now())
            if self._write_delay:
                # Model an eventually-consistent provider: accepted now, visible later.
                asyncio.get_running_loop().call_later(
                    self._write_delay, self._visible.add, native_id
                )
            else:
                self._visible.add(native_id)
            out.append(self._to_memory(native_id))
        return out

    async def update(self, native_id: str, content: str) -> ProviderMemory:
        scope, _, created = self._rows[native_id]
        self._rows[native_id] = (scope, content, created)
        return self._to_memory(native_id)

    # -- read -----------------------------------------------------------------

    async def search(self, query: SearchQuery, scope: Scope) -> list[ProviderMemory]:
        if query.mode not in self._caps.search_modes or query.mode in self._refuse_modes:
            raise UnsupportedCapability(f"fake adapter cannot serve {query.mode!r}")
        self._require_up()

        wanted = set(query.query.lower().split())
        hits = []
        for native_id, (row_scope, content, _) in self._rows.items():
            if native_id not in self._visible:
                continue
            if not self._matches(row_scope, scope):
                continue
            overlap = wanted & set(content.lower().split())
            if not overlap:
                continue
            hit = self._to_memory(native_id)
            hit.score = len(overlap) / max(len(wanted), 1)
            hits.append(hit)

        hits.sort(key=lambda h: h.score or 0.0, reverse=True)
        if query.min_score is not None:
            hits = [h for h in hits if (h.score or 0.0) >= query.min_score]
        return hits[: query.limit]

    async def list_scope(self, scope: Scope, limit: int) -> list[ProviderMemory]:
        return [
            self._to_memory(native_id)
            for native_id, (row_scope, _, _) in self._rows.items()
            if native_id in self._visible and self._matches(row_scope, scope)
        ][:limit]

    async def get(self, native_id: str) -> ProviderMemory | None:
        if native_id not in self._rows:
            return None
        return self._to_memory(native_id)

    def resurrect(self, native_id: str, content: str = "ghost memory") -> None:
        """Put a deleted row back, the way an eventually-consistent provider does when
        it has accepted a delete but not yet propagated it."""
        self._rows[native_id] = (Scope(subject="u_1", tenant="tenant-a"), content, _now())
        self._visible.add(native_id)

    # -- delete ---------------------------------------------------------------

    async def delete(self, native_id: str) -> bool:
        existed = self._rows.pop(native_id, None) is not None
        self._visible.discard(native_id)
        return existed

    async def delete_scope(self, scope: Scope) -> int:
        doomed = [
            nid for nid, (row_scope, _, _) in self._rows.items() if self._matches(row_scope, scope)
        ]
        for native_id in doomed:
            await self.delete(native_id)
        return len(doomed)

    # -- internals ------------------------------------------------------------

    def _require_up(self) -> None:
        if not self._healthy:
            raise ConnectionError("fake adapter is down")

    def _matches(self, row_scope: Scope, filter_scope: Scope) -> bool:
        if self._broken_scope_filter:
            # The bug this suite exists to catch: a filter that ignores the subject
            # and happily returns another end-user's memories.
            return True
        for dim, value in filter_scope.dims().items():
            if dim in self._ignore_dims:
                continue
            if getattr(row_scope, dim) != value:
                return False
        for key, value in filter_scope.labels.items():
            if row_scope.labels.get(key) != value:
                return False
        return True

    def _to_memory(self, native_id: str) -> ProviderMemory:
        scope, content, created = self._rows[native_id]
        del scope
        return ProviderMemory(
            native_id=native_id,
            content=content,
            created_at=created,
            updated_at=created,
            raw={"adapter": "fake"},
        )
