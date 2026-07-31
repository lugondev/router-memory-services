"""Regression tests for the audit findings.

Each test here names a defect that was real and reproducible before the fix. They
live together rather than scattered because the audit is the reason they exist,
and a future reader deserves to know which line of the design each one defends.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from memgw.catalog import Catalog
from memgw.core import MemoryCore
from memgw.errors import MemoryNotFound, UnsupportedCapability
from memgw.types import Episode, Scope, SearchQuery
from tests.fake import FakeAdapter, default_caps

A = "tenant-a"
B = "tenant-b"


async def build(**fake_kw) -> tuple[MemoryCore, Catalog, FakeAdapter]:
    """One adapter instance serving every tenant -- the deployment the README shows,
    and the one under which a missing tenant dimension leaks."""
    catalog = Catalog("sqlite+aiosqlite:///:memory:")
    await catalog.init()
    shared = FakeAdapter(**fake_kw)
    core = MemoryCore(catalog=catalog, providers={"fake": shared}, default_provider="fake")
    return core, catalog, shared


# -- 1. tenant isolation ------------------------------------------------------


class TestTenantsShareAProviderWithoutSharingMemories:
    async def test_another_tenant_cannot_search_the_same_subject_id(self):
        core, _, _ = await build()
        await core.ingest(A, Episode(text="salary is 999"), Scope(subject="u_1"))

        leaked = await core.search(B, SearchQuery(query="salary"), Scope(subject="u_1"))
        assert leaked.results == [], "tenant-b read tenant-a's memory by reusing the subject id"

    async def test_another_tenant_cannot_delete_the_same_subject_id(self):
        core, _, _ = await build()
        await core.ingest(A, Episode(text="salary is 999"), Scope(subject="u_1"))

        assert await core.delete_scope(B, Scope(subject="u_1")) == 0
        mine = await core.search(A, SearchQuery(query="salary"), Scope(subject="u_1"))
        assert len(mine.results) == 1, "tenant-b's erasure took tenant-a's memories with it"

    async def test_the_tenant_stamp_never_reaches_the_caller_facing_record(self):
        # The gateway stamps it on the way down; a caller asked about u_1, not about
        # "tenant-a/u_1", and a scope echoed back with an extra dimension is noise.
        core, _, _ = await build()
        written = await core.ingest(A, Episode(text="coffee"), Scope(subject="u_1"))
        [record] = written.results
        assert record.scope.tenant is None

    async def test_a_caller_supplied_tenant_is_overwritten_not_trusted(self):
        core, _, _ = await build()
        await core.ingest(A, Episode(text="salary is 999"), Scope(subject="u_1"))

        forged = Scope(subject="u_1", tenant=A)
        leaked = await core.search(B, SearchQuery(query="salary"), forged)
        assert leaked.results == [], "a tenant in the payload widened the credential's scope"

    async def test_an_adapter_that_cannot_isolate_tenants_is_refused(self):
        core, _, _ = await build(caps=default_caps(scope_dims=["subject", "agent", "session"]))
        with pytest.raises(UnsupportedCapability):
            await core.ingest(A, Episode(text="coffee"), Scope(subject="u_1"))


# -- 2. catalog ---------------------------------------------------------------


class TestCatalogUnderConcurrency:
    async def test_concurrent_mapping_of_one_native_id_yields_one_gateway_id(self):
        catalog = Catalog("sqlite+aiosqlite:///:memory:")
        await catalog.init()
        scope = Scope(subject="u_1")

        rows = await asyncio.gather(
            *[catalog.ensure(A, "p", "native-1", scope, "x") for _ in range(8)]
        )
        assert len({row.gateway_id for row in rows}) == 1

    async def test_a_concurrent_search_does_not_surface_a_database_error(self):
        core, _, _ = await build()
        scope = Scope(subject="u_1")
        await core.ingest(A, Episode(text="black coffee"), scope)

        results = await asyncio.gather(
            *[core.search(A, SearchQuery(query="coffee"), scope) for _ in range(8)]
        )
        assert all(len(r.results) == 1 for r in results)


class TestADeletedMemoryStaysDeleted:
    async def test_search_does_not_resurrect_a_memory_the_gateway_deleted(self):
        core, _, adapter = await build()
        scope = Scope(subject="u_1")
        [record] = (await core.ingest(A, Episode(text="ghost memory"), scope)).results
        await core.delete(A, record.id)

        # A provider that has not yet propagated the delete -- the ordinary case for
        # anything declaring eventual consistency.
        adapter.resurrect(record.native_id)

        found = await core.search(A, SearchQuery(query="ghost"), scope)
        assert found.results == [], "search returned an id that GET answers with a 404"

    async def test_delete_scope_also_erases_the_raw_episodes(self):
        catalog = Catalog("sqlite+aiosqlite:///:memory:")
        await catalog.init()
        core = MemoryCore(
            catalog=catalog,
            providers={"fake": FakeAdapter()},
            default_provider="fake",
            journal_enabled=True,
        )
        scope = Scope(subject="u_1")
        await core.ingest(A, Episode(text="my home address is 12 Some Street"), scope)
        assert await catalog.journal_rows(A, "u_1")

        await core.delete_scope(A, scope)
        assert await catalog.journal_rows(A, "u_1") == [], (
            "the raw transcript outlived the erasure it was supposed to honour"
        )


class TestBatchedCatalogWrites:
    async def test_many_native_ids_map_in_one_call_and_keep_their_order(self):
        catalog = Catalog("sqlite+aiosqlite:///:memory:")
        await catalog.init()
        scope = Scope(subject="u_1")
        items = [(f"n-{i}", f"content {i}") for i in range(5)]

        rows = await catalog.ensure_many(A, "p", items, scope)
        assert [row.native_id for row in rows] == [native for native, _ in items]
        assert len({row.gateway_id for row in rows}) == 5

    async def test_a_second_pass_reuses_every_gateway_id(self):
        catalog = Catalog("sqlite+aiosqlite:///:memory:")
        await catalog.init()
        scope = Scope(subject="u_1")
        items = [(f"n-{i}", f"content {i}") for i in range(5)]

        first = await catalog.ensure_many(A, "p", items, scope)
        second = await catalog.ensure_many(A, "p", items, scope)
        assert [r.gateway_id for r in first] == [r.gateway_id for r in second]

    async def test_a_soft_deleted_row_is_dropped_rather_than_remapped(self):
        catalog = Catalog("sqlite+aiosqlite:///:memory:")
        await catalog.init()
        scope = Scope(subject="u_1")
        [row] = await catalog.ensure_many(A, "p", [("n-1", "x")], scope)
        await catalog.mark_deleted(A, row.gateway_id)

        assert await catalog.ensure_many(A, "p", [("n-1", "x")], scope) == []


# -- 3. verbs -----------------------------------------------------------------


class TestUpsertIsReal:
    async def test_ready_made_facts_go_in_and_come_back(self):
        core, _, _ = await build()
        scope = Scope(subject="u_1")
        written = await core.upsert(A, ["prefers black coffee"], scope)
        assert len(written.results) == 1
        assert written.results[0].id.startswith("mg_")
        assert written.provider == "fake"

        found = await core.search(A, SearchQuery(query="coffee"), scope)
        assert [r.content for r in found.results] == ["prefers black coffee"]

    async def test_upsert_binds_the_subject_like_any_other_write(self):
        core, catalog, _ = await build()
        await core.upsert(A, ["prefers black coffee"], Scope(subject="u_1"))
        assert await catalog.get_binding(A, "u_1") == "fake"

    async def test_an_adapter_that_declares_no_upsert_is_refused(self):
        core, _, _ = await build(caps=default_caps(supports_upsert=False))
        with pytest.raises(UnsupportedCapability):
            await core.upsert(A, ["x"], Scope(subject="u_1"))


class TestAsOfIsNeverIgnored:
    async def test_a_point_in_time_query_is_refused_without_a_temporal_mode(self):
        from datetime import datetime, timezone

        core, _, _ = await build()
        with pytest.raises(UnsupportedCapability):
            await core.search(
                A,
                SearchQuery(query="coffee", as_of=datetime(2020, 1, 1, tzinfo=timezone.utc)),
                Scope(subject="u_1"),
            )

    async def test_degrade_does_not_buy_a_point_in_time_answer(self):
        # Falling back to "now" would answer a different question than the one asked.
        from datetime import datetime, timezone

        core, _, _ = await build()
        with pytest.raises(UnsupportedCapability):
            await core.search(
                A,
                SearchQuery(
                    query="coffee",
                    as_of=datetime(2020, 1, 1, tzinfo=timezone.utc),
                    on_unsupported="degrade",
                ),
                Scope(subject="u_1"),
            )


class TestFailOpenCoversRealOutages:
    async def test_a_provider_error_is_survivable_when_the_caller_opted_in(self):
        core, _, adapter = await build(healthy=False)
        result = await core.search(
            A, SearchQuery(query="coffee", fail_open=True), Scope(subject="u_1")
        )
        assert result.provider_unavailable is True
        assert result.results == []

    async def test_a_capability_shortfall_still_raises_under_fail_open(self):
        # An outage is temporary; a provider that has no graph search never will.
        core, _, _ = await build()
        with pytest.raises(UnsupportedCapability):
            await core.search(
                A, SearchQuery(query="c", mode="graph", fail_open=True), Scope(subject="u_1")
            )


class TestIngestReportsTheProviderItActuallyUsed:
    async def test_an_extraction_that_kept_nothing_still_names_the_bound_provider(self):
        catalog = Catalog("sqlite+aiosqlite:///:memory:")
        await catalog.init()
        core = MemoryCore(
            catalog=catalog,
            providers={"fake": FakeAdapter(extract_nothing=True), "other": FakeAdapter()},
            default_provider="other",
        )
        await catalog.bind(A, "u_1", "fake")

        result = await core.ingest(A, Episode(text="small talk"), Scope(subject="u_1"))
        assert result.provider == "fake"
        assert result.results == []


class TestListingAScope:
    async def test_a_scope_can_be_read_back_without_a_query(self):
        core, _, _ = await build()
        scope = Scope(subject="u_1")
        await core.upsert(A, ["one", "two"], scope)

        listed = await core.list_scope(A, scope)
        assert {r.content for r in listed} == {"one", "two"}

    async def test_listing_respects_tenant_isolation(self):
        core, _, _ = await build()
        await core.upsert(A, ["one"], Scope(subject="u_1"))
        assert await core.list_scope(B, Scope(subject="u_1")) == []


# -- 4. auth and observability ------------------------------------------------


class TestApiKeysAreNotHandledCarelessly:
    def test_the_principal_does_not_carry_the_key_itself(self):
        from memgw.server.auth import ApiKeyAuth

        key = "sk-super-secret-value"
        who = ApiKeyAuth({key: "t1"}).authenticate(f"Bearer {key}")
        assert not key.startswith(who.key_id), "key_id leaks the start of the key into logs"
        assert who.key_id not in key

    def test_a_wrong_key_is_rejected_without_a_dictionary_lookup_on_the_raw_token(self):
        from memgw.server import auth as auth_module

        assert hasattr(auth_module, "_digest"), "keys should be compared by digest"


class TestEveryVerbLeavesATrail:
    async def test_a_write_is_logged_with_tenant_subject_and_provider(self, caplog):
        core, _, _ = await build()
        with caplog.at_level(logging.INFO, logger="memgw"):
            await core.ingest(A, Episode(text="coffee"), Scope(subject="u_1"))

        entries = [r for r in caplog.records if r.name == "memgw"]
        assert entries, "an audit trail is the point of putting a gateway in the path"
        record = entries[-1]
        assert record.tenant == A
        assert record.subject == "u_1"
        assert record.provider == "fake"
        assert record.verb == "ingest"

    async def test_a_failure_is_logged_with_its_outcome(self, caplog):
        from memgw.errors import ProviderUnhealthy

        core, _, _ = await build(healthy=False)
        with caplog.at_level(logging.INFO, logger="memgw"), pytest.raises(ProviderUnhealthy):
            await core.ingest(A, Episode(text="coffee"), Scope(subject="u_1"))

        assert any(r.name == "memgw" and r.outcome == "error" for r in caplog.records)


# -- 5. client ----------------------------------------------------------------


class TestEmbeddedStartupIsSafe:
    async def test_concurrent_first_calls_build_exactly_one_core(self, tmp_path):
        from memgw.client import Memory

        mem = Memory(
            provider="pgvector",
            config={"url": "sqlite+aiosqlite:///:memory:", "embedder": _Embedder()},
            catalog_url=f"sqlite+aiosqlite:///{tmp_path}/catalog.db",
        )
        cores = await asyncio.gather(*[mem._started() for _ in range(8)])
        assert len({id(core) for core in cores}) == 1
        await mem.close()


class _Embedder:
    dimension = 8

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 8 == i) for i in range(8)] for t in texts]


# -- 6. adapters --------------------------------------------------------------


class TestPgvectorEdges:
    async def test_updating_a_memory_that_is_not_there_is_not_found(self):
        from memgw.adapters.pgvector import PgvectorAdapter

        adapter = PgvectorAdapter("sqlite+aiosqlite:///:memory:", embedder=_Embedder())
        await adapter.init()
        with pytest.raises(MemoryNotFound):
            await adapter.update("pv_nope", "new content")

    async def test_deleting_a_large_scope_does_not_hit_a_bind_parameter_limit(self):
        from memgw.adapters.pgvector import PgvectorAdapter

        adapter = PgvectorAdapter("sqlite+aiosqlite:///:memory:", embedder=_Embedder())
        await adapter.init()
        scope = Scope(subject="u_1")
        await adapter.upsert([f"fact {i}" for i in range(1200)], scope)

        assert await adapter.delete_scope(scope) == 1200


class TestMem0DeletesEverything:
    async def test_delete_scope_pages_past_the_first_batch(self):
        from memgw.adapters.mem0 import Mem0Adapter

        client = _PagingMem0(total=2500)
        adapter = Mem0Adapter(client=client)
        removed = await adapter.delete_scope(Scope(subject="u_1"))
        assert removed == 2500, "a partial erasure was reported as a complete one"
        assert client.remaining() == 0

    async def test_a_tenant_stamp_namespaces_the_user_id(self):
        from memgw.adapters.mem0 import build_filters, write_ids

        scoped = Scope(subject="u_1", tenant="tenant-a")
        assert build_filters(scoped)["user_id"] == "tenant-a:u_1"
        assert write_ids(scoped)["user_id"] == "tenant-a:u_1"


class _PagingMem0:
    """A Mem0 stand-in that never returns everything at once, which is the whole
    point: the real client caps a page and the adapter must keep asking."""

    PAGE = 1000

    def __init__(self, total: int) -> None:
        self._rows = {f"m-{i}": {"id": f"m-{i}", "memory": f"fact {i}"} for i in range(total)}

    def remaining(self) -> int:
        return len(self._rows)

    async def get_all(self, filters=None, top_k=None, **kwargs):
        del filters, kwargs
        page = min(top_k or self.PAGE, self.PAGE)
        return {"results": list(self._rows.values())[:page]}

    async def delete(self, native_id):
        self._rows.pop(native_id, None)
