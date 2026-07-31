"""The suite, run against the control specimen -- including a broken one."""

from __future__ import annotations

import pytest

from memgw.errors import UnsupportedCapability
from memgw.types import Scope, SearchQuery
from tests.conformance.suite import ConformanceSuite
from tests.fake import FakeAdapter, default_caps


class TestFakeConformance(ConformanceSuite):
    async def make_adapter(self):
        return FakeAdapter()


class TestFakeEventualConformance(ConformanceSuite):
    """Same adapter, declaring (and behaving with) eventual consistency."""

    settle_timeout = 2.0

    async def make_adapter(self):
        return FakeAdapter(default_caps(consistency="eventual"), write_delay=0.2)


class TestTheSuiteActuallyBites:
    async def test_a_broken_scope_filter_fails_the_isolation_check(self):
        broken = FakeAdapter(broken_scope_filter=True)

        class _Broken(ConformanceSuite):
            async def make_adapter(self):
                return broken

        with pytest.raises(AssertionError, match="could read"):
            await _Broken().test_one_subject_cannot_read_another_subjects_memories()

    async def test_an_adapter_that_honours_subject_but_ignores_tenant_fails(self):
        """The precise shape of the bug this dimension was added for.

        Subject filtering looks correct in every single-tenant test, so a suite that
        only checked subjects would pass an adapter that serves two tenants the same
        row. This is the adapter every provider is before someone remembers tenants.
        """
        leaky = FakeAdapter(ignore_dims={"tenant"})

        class _Leaky(ConformanceSuite):
            async def make_adapter(self):
                return leaky

        await _Leaky().test_one_subject_cannot_read_another_subjects_memories()
        with pytest.raises(AssertionError, match="same subject id"):
            await _Leaky().test_one_tenant_cannot_read_another_tenants_memories()

    async def test_a_lying_consistency_declaration_fails(self):
        # Declares read_after_write, behaves eventually.
        liar = FakeAdapter(default_caps(consistency="read_after_write"), write_delay=5.0)

        class _Liar(ConformanceSuite):
            async def make_adapter(self):
                return liar

        with pytest.raises(AssertionError, match="read_after_write"):
            await _Liar().test_declared_consistency_is_the_truth()

    async def test_a_declared_mode_the_implementation_refuses_fails(self):
        # Declares graph, then refuses it. This is the drift the suite can catch.
        # What it cannot catch is an adapter that declares graph and answers with
        # plain semantic search -- no generic assertion distinguishes those.
        liar = FakeAdapter(default_caps(search_modes=["semantic", "graph"]), refuse_modes={"graph"})

        class _Liar(ConformanceSuite):
            async def make_adapter(self):
                return liar

        with pytest.raises(UnsupportedCapability, match="cannot serve"):
            await _Liar().test_declared_search_modes_work_and_undeclared_ones_refuse()

    async def test_delete_scope_that_over_deletes_fails(self):
        over = FakeAdapter(broken_scope_filter=True)

        class _Over(ConformanceSuite):
            async def make_adapter(self):
                return over

        with pytest.raises(AssertionError):
            await _Over().test_delete_scope_clears_its_scope_and_touches_no_other()


class TestFakeAdapterItself:
    async def test_labels_narrow_the_filter(self):
        adapter = FakeAdapter()
        await adapter.upsert(["pro tier note"], Scope(subject="u_1", labels={"tier": "pro"}))

        query = SearchQuery(query="note")
        assert await adapter.search(query, Scope(subject="u_1", labels={"tier": "pro"}))
        assert not await adapter.search(query, Scope(subject="u_1", labels={"tier": "free"}))
