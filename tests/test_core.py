import pytest

from memgw.catalog import Catalog
from memgw.core import MemoryCore
from memgw.errors import (
    MemoryNotFound,
    NotImplementedYet,
    ProviderMismatch,
    ProviderUnhealthy,
    UnsupportedCapability,
)
from memgw.types import Episode, Scope, SearchQuery
from tests.fake import FakeAdapter, default_caps

TENANT = "t1"


async def build(*, journal=False, default_provider="fake", **fake_kw) -> tuple[MemoryCore, Catalog]:
    catalog = Catalog("sqlite+aiosqlite:///:memory:")
    await catalog.init()
    core = MemoryCore(
        catalog=catalog,
        providers={"fake": FakeAdapter(**fake_kw), "other": FakeAdapter()},
        default_provider=default_provider,
        journal_enabled=journal,
    )
    return core, catalog


class TestBinding:
    async def test_the_first_write_binds_and_the_second_reuses(self):
        core, catalog = await build()
        await core.ingest(TENANT, Episode(text="black coffee"), Scope(subject="u_1"))
        assert await catalog.get_binding(TENANT, "u_1") == "fake"

        await core.ingest(TENANT, Episode(text="green tea"), Scope(subject="u_1"))
        assert await catalog.get_binding(TENANT, "u_1") == "fake"

    async def test_search_never_binds(self):
        core, catalog = await build()
        await core.search(TENANT, SearchQuery(query="coffee"), Scope(subject="u_1"))
        assert await catalog.get_binding(TENANT, "u_1") is None

    async def test_a_disagreeing_provider_assertion_never_reaches_the_adapter(self):
        core, catalog = await build()
        await core.ingest(TENANT, Episode(text="coffee"), Scope(subject="u_1"))

        with pytest.raises(ProviderMismatch):
            await core.search(
                TENANT, SearchQuery(query="coffee"), Scope(subject="u_1"), provider="other"
            )


class TestIngest:
    async def test_ingest_returns_records_carrying_gateway_ids(self):
        core, _ = await build()
        written = await core.ingest(TENANT, Episode(text="black coffee"), Scope(subject="u_1"))
        assert len(written.results) == 1
        assert written.results[0].id.startswith("mg_")
        assert written.provider == "fake"
        assert written.results[0].content == "black coffee"

    async def test_ingest_is_refused_when_the_adapter_cannot_extract(self):
        core, _ = await build(caps=default_caps(supports_ingest=False))
        with pytest.raises(UnsupportedCapability):
            await core.ingest(TENANT, Episode(text="x"), Scope(subject="u_1"))

    async def test_a_write_never_fails_open(self):
        # An accepted ingest that stored nothing is a lie, so fail_open has no
        # meaning on the write path and a dead provider always raises.
        core, _ = await build(healthy=False)
        with pytest.raises(ProviderUnhealthy):
            await core.ingest(TENANT, Episode(text="x"), Scope(subject="u_1"))


class TestSearch:
    async def test_search_maps_provider_hits_to_gateway_ids(self):
        core, _ = await build()
        await core.ingest(TENANT, Episode(text="black coffee"), Scope(subject="u_1"))

        result = await core.search(TENANT, SearchQuery(query="coffee"), Scope(subject="u_1"))
        assert [r.content for r in result.results] == ["black coffee"]
        assert result.results[0].id.startswith("mg_")
        assert result.degraded is False

    async def test_a_dead_provider_is_a_503_by_default(self):
        core, _ = await build(healthy=False)
        with pytest.raises(ProviderUnhealthy):
            await core.search(TENANT, SearchQuery(query="coffee"), Scope(subject="u_1"))

    async def test_fail_open_returns_empty_but_says_so(self):
        core, _ = await build(healthy=False)
        result = await core.search(
            TENANT, SearchQuery(query="coffee", fail_open=True), Scope(subject="u_1")
        )
        assert result.results == []
        assert result.provider_unavailable is True
        assert result.provider == "fake"

    async def test_fail_open_does_not_swallow_a_capability_refusal(self):
        # Not being able to do graph search is not an outage, and pretending it is
        # would hide a permanent misconfiguration behind an empty result.
        core, _ = await build()
        with pytest.raises(UnsupportedCapability):
            await core.search(
                TENANT,
                SearchQuery(query="coffee", mode="graph", fail_open=True),
                Scope(subject="u_1"),
            )

    async def test_degrade_reports_what_it_gave_up(self):
        core, _ = await build()
        await core.ingest(TENANT, Episode(text="black coffee"), Scope(subject="u_1"))

        result = await core.search(
            TENANT,
            SearchQuery(query="coffee", mode="graph", on_unsupported="degrade"),
            Scope(subject="u_1"),
        )
        assert result.degraded is True
        assert (result.requested, result.served) == ("graph", "semantic")
        assert result.lost == ["graph_traversal"]

    async def test_limit_is_clamped_to_what_the_provider_allows(self):
        core, _ = await build(caps=default_caps(max_limit=2))
        for word in ("coffee one", "coffee two", "coffee three"):
            await core.ingest(TENANT, Episode(text=word), Scope(subject="u_1"))

        result = await core.search(
            TENANT, SearchQuery(query="coffee", limit=50), Scope(subject="u_1")
        )
        assert len(result.results) == 2


class TestGetUpdateDelete:
    async def test_get_returns_the_record(self):
        core, _ = await build()
        [written] = (
            await core.ingest(TENANT, Episode(text="black coffee"), Scope(subject="u_1"))
        ).results
        got = await core.get(TENANT, written.id)
        assert got.id == written.id
        assert got.content == "black coffee"

    async def test_another_tenants_id_is_a_404(self):
        core, _ = await build()
        [written] = (await core.ingest(TENANT, Episode(text="x"), Scope(subject="u_1"))).results
        with pytest.raises(MemoryNotFound):
            await core.get("t2", written.id)

    async def test_update_rewrites_the_content(self):
        core, _ = await build()
        [written] = (
            await core.ingest(TENANT, Episode(text="black coffee"), Scope(subject="u_1"))
        ).results
        updated = await core.update(TENANT, written.id, "green tea")
        assert updated.content == "green tea"
        assert updated.id == written.id

    async def test_delete_removes_it_from_search_and_from_get(self):
        core, _ = await build()
        [written] = (
            await core.ingest(TENANT, Episode(text="black coffee"), Scope(subject="u_1"))
        ).results
        await core.delete(TENANT, written.id)

        with pytest.raises(MemoryNotFound):
            await core.get(TENANT, written.id)
        result = await core.search(TENANT, SearchQuery(query="coffee"), Scope(subject="u_1"))
        assert result.results == []

    async def test_delete_scope_clears_the_subject(self):
        core, _ = await build()
        await core.ingest(TENANT, Episode(text="coffee one"), Scope(subject="u_1"))
        await core.ingest(TENANT, Episode(text="coffee two"), Scope(subject="u_1"))
        await core.ingest(TENANT, Episode(text="coffee three"), Scope(subject="u_2"))

        assert await core.delete_scope(TENANT, Scope(subject="u_1")) == 2
        survivors = await core.search(TENANT, SearchQuery(query="coffee"), Scope(subject="u_2"))
        assert len(survivors.results) == 1


class TestReservedVerbs:
    async def test_rebind_migrate_is_specified_but_not_built(self):
        core, _ = await build()
        with pytest.raises(NotImplementedYet):
            await core.rebind(TENANT, "u_1", "other", strategy="migrate")


class TestRebind:
    async def test_fresh_start_moves_the_binding_and_names_the_casualties(self):
        core, catalog = await build()
        await core.ingest(TENANT, Episode(text="coffee one"), Scope(subject="u_1"))
        await core.ingest(TENANT, Episode(text="coffee two"), Scope(subject="u_1"))

        result = await core.rebind(TENANT, "u_1", "other", strategy="fresh_start")

        assert result.orphaned_at == "fake"
        assert result.orphaned_count == 2
        assert await catalog.get_binding(TENANT, "u_1") == "other"
        # Stranded, not destroyed.
        assert await catalog.live_count(TENANT, "u_1", "fake") == 2


class TestJournal:
    async def test_journal_is_off_by_default(self):
        core, catalog = await build(journal=False)
        await core.ingest(TENANT, Episode(text="black coffee"), Scope(subject="u_1"))
        assert await catalog.journal_rows(TENANT, "u_1") == []

    async def test_journal_on_writes_one_row_with_every_native_id(self):
        core, catalog = await build(journal=True)
        await core.ingest(TENANT, Episode(text="black coffee"), Scope(subject="u_1"))

        rows = await catalog.journal_rows(TENANT, "u_1")
        assert len(rows) == 1
        assert list(rows[0].ingested_to) == ["fake"]
        assert len(rows[0].ingested_to["fake"]) == 1
        assert rows[0].payload["text"] == "black coffee"


class TestIntrospection:
    async def test_capabilities_reports_the_configured_instance(self):
        core, _ = await build(caps=default_caps(supports_ingest=False, max_limit=7))
        caps = core.capabilities("fake")
        assert caps.supports_ingest is False
        assert caps.max_limit == 7

    async def test_providers_status_reports_health_per_provider(self):
        core, _ = await build(healthy=False)
        status = await core.providers_status()
        by_name = {entry.name: entry for entry in status}
        assert by_name["fake"].healthy is False
        assert by_name["other"].healthy is True
