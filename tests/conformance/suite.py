"""The shared conformance suite. Every adapter passes this or it does not ship.

Subclass ``ConformanceSuite`` in a module named ``test_<adapter>.py``, name the
subclass ``Test...``, and implement ``make_adapter``. The suite reads the
adapter's own ``capabilities()`` to decide which checks apply -- so the capability
schema is simultaneously the public API and the thing that drives the tests.

One check is exempt from that: **scope isolation is never skipped**, for any
adapter, under any declaration.

What this suite proves and what it does not: it catches a declaration that drifts
from behaviour -- a mode declared and then refused, a consistency promise the
adapter cannot keep, a scope filter that leaks. It cannot verify that a declared
``graph`` search really traverses a graph rather than quietly running semantic
search; no generic assertion distinguishes those. Semantic fidelity of an
exotic mode stays the adapter author's responsibility.
"""

from __future__ import annotations

import asyncio

import pytest

from memgw.adapters.base import MemoryAdapter
from memgw.errors import UnsupportedCapability
from memgw.types import Episode, ProviderMemory, Scope, SearchMode, SearchQuery

ALL_MODES: list[SearchMode] = ["semantic", "keyword", "hybrid", "graph", "temporal"]


class ConformanceSuite:
    #: How long to wait on an adapter that declares ``consistency="eventual"``.
    settle_timeout: float = 5.0
    settle_interval: float = 0.1

    async def make_adapter(self) -> MemoryAdapter:
        raise NotImplementedError

    # -- helpers --------------------------------------------------------------

    async def _write(self, adapter: MemoryAdapter, content: str, scope: Scope) -> ProviderMemory:
        caps = adapter.capabilities()
        if caps.supports_upsert:
            written = await adapter.upsert([content], scope)
        elif caps.supports_ingest:
            written = await adapter.ingest(Episode(text=content), scope)
        else:
            pytest.skip("adapter accepts neither facts nor episodes")
        assert written, "a write returned no memories"
        return written[0]

    async def _find(
        self, adapter: MemoryAdapter, needle: str, scope: Scope
    ) -> list[ProviderMemory]:
        return await adapter.search(SearchQuery(query=needle, limit=50), scope)

    async def _settle(self, adapter: MemoryAdapter, needle: str, scope: Scope) -> None:
        """Block until a write is visible, honouring the adapter's declared consistency.

        Absence is never polled for -- asserting "not there yet" against an
        eventually-consistent provider proves nothing. So every absence check first
        settles the write in its own scope, then asserts immediately elsewhere.
        """
        deadline = asyncio.get_running_loop().time() + self.settle_timeout
        while True:
            hits = await self._find(adapter, needle, scope)
            if any(needle in hit.content for hit in hits):
                return
            if adapter.capabilities().consistency == "read_after_write":
                raise AssertionError(
                    f"adapter declares read_after_write but {needle!r} was not readable "
                    "immediately after the write"
                )
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError(
                    f"{needle!r} never became visible within {self.settle_timeout}s"
                )
            await asyncio.sleep(self.settle_interval)

    async def _await_absent(self, adapter: MemoryAdapter, needle: str, scope: Scope) -> None:
        deadline = asyncio.get_running_loop().time() + self.settle_timeout
        while True:
            hits = await self._find(adapter, needle, scope)
            if not any(needle in hit.content for hit in hits):
                return
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError(f"{needle!r} was still readable after deletion")
            await asyncio.sleep(self.settle_interval)

    # -- 1. round trip --------------------------------------------------------

    async def test_write_then_read_back_with_a_full_scope(self):
        adapter = await self.make_adapter()
        scope = Scope(subject="u_1", agent="lugo", session="s_9")

        await self._write(adapter, "blackcoffee", scope)
        await self._settle(adapter, "blackcoffee", scope)

        hits = await self._find(adapter, "blackcoffee", scope)
        assert any("blackcoffee" in hit.content for hit in hits)

    # -- 2. isolation. never skipped ------------------------------------------

    async def test_one_subject_cannot_read_another_subjects_memories(self):
        adapter = await self.make_adapter()
        mine = Scope(subject="u_1", agent="lugo", session="s_9")
        theirs = Scope(subject="u_2", agent="lugo", session="s_9")

        await self._write(adapter, "blackcoffee", mine)
        await self._settle(adapter, "blackcoffee", mine)

        leaked = await self._find(adapter, "blackcoffee", theirs)
        assert not any("blackcoffee" in hit.content for hit in leaked), (
            "subject u_2 could read subject u_1's memory"
        )

    async def test_a_write_carrying_an_agent_scope_is_readable_through_the_recall_path(self):
        # The documented silent-filter trap: on at least one provider, writing with an
        # agent scope and reading with a plain subject filter returns nothing at all,
        # with no error. This asserts the adapter builds the full filter.
        adapter = await self.make_adapter()
        if "agent" not in adapter.capabilities().scope_dims:
            pytest.skip("adapter does not model an agent dimension")

        scoped = Scope(subject="u_1", agent="lugo")
        await self._write(adapter, "greentea", scoped)
        await self._settle(adapter, "greentea", scoped)

        hits = await self._find(adapter, "greentea", scoped)
        assert any("greentea" in hit.content for hit in hits)

    # -- 3. delete ------------------------------------------------------------

    async def test_a_deleted_memory_stops_coming_back(self):
        adapter = await self.make_adapter()
        if not adapter.capabilities().supports_delete:
            pytest.skip("adapter declares no delete")

        scope = Scope(subject="u_1")
        written = await self._write(adapter, "oolong", scope)
        await self._settle(adapter, "oolong", scope)

        assert await adapter.delete(written.native_id) is True
        await self._await_absent(adapter, "oolong", scope)

    # -- 4. delete by scope ---------------------------------------------------

    async def test_delete_scope_clears_its_scope_and_touches_no_other(self):
        adapter = await self.make_adapter()
        if not adapter.capabilities().supports_delete_by_scope:
            pytest.skip("adapter declares no delete-by-scope")

        mine = Scope(subject="u_1")
        theirs = Scope(subject="u_2")
        await self._write(adapter, "matcha", mine)
        await self._write(adapter, "matcha", theirs)
        await self._settle(adapter, "matcha", mine)
        await self._settle(adapter, "matcha", theirs)

        cleared = await adapter.delete_scope(mine)
        assert cleared >= 1

        await self._await_absent(adapter, "matcha", mine)
        survivors = await self._find(adapter, "matcha", theirs)
        assert any("matcha" in hit.content for hit in survivors), (
            "delete_scope removed another subject's memories"
        )

    # -- 5. the consistency declaration is true -------------------------------

    async def test_declared_consistency_is_the_truth(self):
        adapter = await self.make_adapter()
        scope = Scope(subject="u_1")
        await self._write(adapter, "rooibos", scope)

        if adapter.capabilities().consistency == "read_after_write":
            hits = await self._find(adapter, "rooibos", scope)
            assert any("rooibos" in hit.content for hit in hits), (
                "adapter declares read_after_write but the write was not immediately readable"
            )
        else:
            await self._settle(adapter, "rooibos", scope)

    # -- 6. the capability declaration is true --------------------------------

    async def test_declared_search_modes_work_and_undeclared_ones_refuse(self):
        adapter = await self.make_adapter()
        caps = adapter.capabilities()
        scope = Scope(subject="u_1")
        await self._write(adapter, "chamomile", scope)
        await self._settle(adapter, "chamomile", scope)

        for mode in caps.search_modes:
            await adapter.search(SearchQuery(query="chamomile", mode=mode), scope)

        for mode in ALL_MODES:
            if mode in caps.search_modes:
                continue
            with pytest.raises(UnsupportedCapability):
                await adapter.search(SearchQuery(query="chamomile", mode=mode), scope)

    async def test_declared_write_verbs_are_the_ones_that_work(self):
        adapter = await self.make_adapter()
        caps = adapter.capabilities()
        scope = Scope(subject="u_1")

        if caps.supports_ingest:
            assert await adapter.ingest(Episode(text="hibiscus"), scope)
        else:
            with pytest.raises(UnsupportedCapability):
                await adapter.ingest(Episode(text="hibiscus"), scope)

        if caps.supports_upsert:
            assert await adapter.upsert(["lavender"], scope)
        else:
            with pytest.raises(UnsupportedCapability):
                await adapter.upsert(["lavender"], scope)
