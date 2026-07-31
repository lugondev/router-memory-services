"""The Mem0 adapter.

Scope maps cleanly: ``subject → user_id``, ``agent → agent_id``,
``session → run_id``.

The trap this adapter exists to neutralise: a filter naming only ``user_id``
matches memories whose other entity ids are null. Write with an agent scope, read
with a plain user filter, and you get **nothing back and no error** -- memory that
looks like it works and recalls a blank. So reads always name every dimension,
using Mem0's documented ``"*"`` wildcard for the ones the caller left open. That
is what keeps "everything about this user, across all sessions" working here.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from memgw import adapters
from memgw.capabilities import Capabilities
from memgw.errors import ProviderError, UnsupportedCapability
from memgw.types import Episode, HealthStatus, ProviderMemory, Scope, SearchQuery

#: Mem0's "match any value" filter value. Without it an unset dimension means
#: "must be null", which is the silent-empty-recall bug.
ANY = "*"


def build_filters(scope: Scope) -> dict[str, Any]:
    """Every dimension, always. Unset ones become the wildcard rather than absent."""
    return {
        "user_id": scope.subject,
        "agent_id": scope.agent or ANY,
        "run_id": scope.session or ANY,
    }


def write_ids(scope: Scope) -> dict[str, Any]:
    """Writes name only what is actually set -- a wildcard is meaningless on a write."""
    ids: dict[str, Any] = {"user_id": scope.subject}
    if scope.agent:
        ids["agent_id"] = scope.agent
    if scope.session:
        ids["run_id"] = scope.session
    return ids


class Mem0Adapter:
    name = "mem0"

    def __init__(self, config: dict[str, Any] | None = None, *, client: Any = None) -> None:
        if client is not None:
            self._client = client
        else:
            from mem0 import AsyncMemory  # imported here so the package stays optional

            self._client = AsyncMemory.from_config(config) if config else AsyncMemory()

    # -- introspection --------------------------------------------------------

    def capabilities(self) -> Capabilities:
        return Capabilities(
            supports_ingest=True,
            supports_upsert=True,
            supports_update=True,
            supports_delete=True,
            supports_delete_by_scope=True,
            search_modes=["semantic"],
            supports_score=True,
            max_limit=100,
            scope_dims=["subject", "agent", "session"],
            supports_labels=True,
            memory_model="flat_facts",
            dedup="provider",
            # No usable bulk export, which is why an end-user on Mem0 can only be
            # moved by replaying the gateway's own journal.
            supports_export=False,
            supports_import=True,
            # Declared eventual on purpose. A local OSS deployment is usually
            # immediate, but the hosted path is not, and a provider that is
            # sometimes eventual is eventual. Over-declaring costs a poll;
            # under-declaring produces flaky reads nobody can reproduce.
            consistency="eventual",
            # add()/search() spend on their own LLM and embedder, outside any ledger
            # the caller controls. The gateway knows the call count, not the cost.
            metered_externally=True,
        )

    async def health(self) -> HealthStatus:
        try:
            await self._client.get_all(filters={"user_id": "__memgw_health__"}, top_k=1)
            return HealthStatus(ok=True)
        except Exception as exc:  # noqa: BLE001
            return HealthStatus(ok=False, detail=str(exc))

    # -- write ----------------------------------------------------------------

    async def ingest(self, episode: Episode, scope: Scope) -> list[ProviderMemory]:
        if episode.messages:
            messages = [{"role": m.role, "content": m.content} for m in episode.messages]
        else:
            messages = [{"role": "user", "content": episode.as_text()}]

        raw = await self._client.add(
            messages,
            metadata=dict(scope.labels) or None,
            infer=True,
            **write_ids(scope),
        )
        return self._to_memories(raw)

    async def upsert(self, facts: list[str], scope: Scope) -> list[ProviderMemory]:
        out: list[ProviderMemory] = []
        for fact in facts:
            raw = await self._client.add(
                [{"role": "user", "content": fact}],
                metadata=dict(scope.labels) or None,
                infer=False,
                **write_ids(scope),
            )
            out.extend(self._to_memories(raw))
        return out

    async def update(self, native_id: str, content: str) -> ProviderMemory:
        await self._client.update(native_id, text=content)
        got = await self.get(native_id)
        if got is None:
            raise ProviderError(f"mem0 lost {native_id!r} during update")
        return got

    # -- read -----------------------------------------------------------------

    async def search(self, query: SearchQuery, scope: Scope) -> list[ProviderMemory]:
        if query.mode != "semantic":
            raise UnsupportedCapability(
                f"mem0 adapter cannot serve {query.mode!r} search",
                details={"available": ["semantic"]},
            )
        raw = await self._client.search(
            query.query,
            top_k=query.limit,
            filters=build_filters(scope),
            threshold=query.min_score if query.min_score is not None else 0.0,
        )
        return self._to_memories(raw)

    async def get(self, native_id: str) -> ProviderMemory | None:
        raw = await self._client.get(native_id)
        if not raw:
            return None
        return self._to_memory(raw)

    # -- delete ---------------------------------------------------------------

    async def delete(self, native_id: str) -> bool:
        await self._client.delete(native_id)
        return True

    async def delete_scope(self, scope: Scope) -> int:
        """Enumerate then delete, rather than calling delete_all.

        delete_all takes the same entity ids but not the wildcard, so its behaviour
        for an unset dimension is exactly the ambiguity this adapter exists to avoid.
        Listing under the explicit filter and deleting what comes back is slower and
        unambiguous.
        """
        raw = await self._client.get_all(filters=build_filters(scope), top_k=1000)
        doomed = self._to_memories(raw)
        for memory in doomed:
            await self._client.delete(memory.native_id)
        return len(doomed)

    # -- internals ------------------------------------------------------------

    def _to_memories(self, raw: Any) -> list[ProviderMemory]:
        if raw is None:
            return []
        if isinstance(raw, dict):
            raw = raw.get("results", [])
        if isinstance(raw, dict):
            raw = [raw]
        return [self._to_memory(item) for item in raw if self._content(item)]

    @staticmethod
    def _content(item: Any) -> str | None:
        if not isinstance(item, dict):
            return None
        return item.get("memory") or item.get("text") or item.get("data")

    def _to_memory(self, item: dict[str, Any]) -> ProviderMemory:
        return ProviderMemory(
            native_id=str(item.get("id")),
            content=self._content(item) or "",
            created_at=item.get("created_at"),
            updated_at=item.get("updated_at"),
            score=item.get("score"),
            raw=item,
        )


#: Registered only when the optional dependency is present, so ``available()``
#: reports what can actually be built rather than what exists in the source tree.
if importlib.util.find_spec("mem0") is not None:  # pragma: no branch
    adapters.register("mem0", Mem0Adapter)
