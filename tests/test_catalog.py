import pytest

from memgw.catalog import Catalog
from memgw.types import Scope


@pytest.fixture
async def catalog():
    cat = Catalog("sqlite+aiosqlite:///:memory:")
    await cat.init()
    yield cat
    await cat.close()


class TestRecord:
    async def test_a_gateway_id_is_issued_and_is_sortable(self, catalog):
        first = await catalog.record("t1", "mem0", "n-1", Scope(subject="u_1"), "black coffee")
        second = await catalog.record("t1", "mem0", "n-2", Scope(subject="u_1"), "green tea")
        assert first.startswith("mg_")
        assert first < second, "ids must sort by creation order"

    async def test_recording_the_same_native_id_twice_keeps_one_gateway_id(self, catalog):
        first = await catalog.record("t1", "mem0", "n-1", Scope(subject="u_1"), "black coffee")
        again = await catalog.record("t1", "mem0", "n-1", Scope(subject="u_1"), "black coffee!")
        assert again == first

    async def test_the_same_native_id_under_two_providers_is_two_rows(self, catalog):
        a = await catalog.record("t1", "mem0", "n-1", Scope(subject="u_1"), "x")
        b = await catalog.record("t1", "pgvector", "n-1", Scope(subject="u_1"), "x")
        assert a != b


class TestResolve:
    async def test_a_row_resolves_within_its_tenant(self, catalog):
        gid = await catalog.record("t1", "mem0", "n-1", Scope(subject="u_1", agent="lugo"), "x")
        row = await catalog.resolve_gateway_id("t1", gid)
        assert row is not None
        assert (row.provider, row.native_id, row.subject, row.agent) == ("mem0", "n-1", "u_1", "lugo")

    async def test_another_tenants_row_is_invisible(self, catalog):
        # The route turns this into a 404, not a 403: a 403 would confirm the id
        # exists, which is an existence oracle across tenants.
        gid = await catalog.record("t1", "mem0", "n-1", Scope(subject="u_1"), "x")
        assert await catalog.resolve_gateway_id("t2", gid) is None

    async def test_a_deleted_row_stops_resolving(self, catalog):
        gid = await catalog.record("t1", "mem0", "n-1", Scope(subject="u_1"), "x")
        assert await catalog.mark_deleted("t1", gid) is True
        assert await catalog.resolve_gateway_id("t1", gid) is None

    async def test_deleting_someone_elses_row_does_nothing(self, catalog):
        gid = await catalog.record("t1", "mem0", "n-1", Scope(subject="u_1"), "x")
        assert await catalog.mark_deleted("t2", gid) is False
        assert await catalog.resolve_gateway_id("t1", gid) is not None


class TestBinding:
    async def test_unbound_subject_reports_none(self, catalog):
        assert await catalog.get_binding("t1", "u_1") is None

    async def test_bind_then_read_back(self, catalog):
        await catalog.bind("t1", "u_1", "mem0")
        assert await catalog.get_binding("t1", "u_1") == "mem0"

    async def test_bind_is_idempotent_and_does_not_silently_switch(self, catalog):
        await catalog.bind("t1", "u_1", "mem0")
        await catalog.bind("t1", "u_1", "pgvector")
        assert await catalog.get_binding("t1", "u_1") == "mem0", (
            "bind must not move an end-user's memories; that is what rebind is for"
        )

    async def test_bindings_are_per_tenant(self, catalog):
        await catalog.bind("t1", "u_1", "mem0")
        assert await catalog.get_binding("t2", "u_1") is None


class TestRebind:
    async def test_rebind_reports_what_was_stranded_and_strands_it(self, catalog):
        await catalog.bind("t1", "u_1", "mem0")
        await catalog.record("t1", "mem0", "n-1", Scope(subject="u_1"), "a")
        await catalog.record("t1", "mem0", "n-2", Scope(subject="u_1"), "b")
        await catalog.record("t1", "mem0", "n-3", Scope(subject="u_2"), "other subject")

        result = await catalog.rebind("t1", "u_1", "pgvector")

        assert result.orphaned_at == "mem0"
        assert result.orphaned_count == 2
        assert await catalog.get_binding("t1", "u_1") == "pgvector"
        # fresh_start strands, it does not destroy.
        assert await catalog.live_count("t1", "u_1", "mem0") == 2

    async def test_rebinding_an_unbound_subject_just_binds_it(self, catalog):
        result = await catalog.rebind("t1", "u_1", "pgvector")
        assert result.orphaned_at is None
        assert result.orphaned_count == 0
        assert await catalog.get_binding("t1", "u_1") == "pgvector"


class TestJournal:
    async def test_one_ingest_writes_one_row_carrying_every_native_id(self, catalog):
        episode_id = await catalog.journal(
            "t1",
            Scope(subject="u_1", agent="lugo", session="s_9"),
            {"text": "I drink black coffee"},
            {"mem0": ["n-1", "n-2"]},
        )
        rows = await catalog.journal_rows("t1", "u_1")
        assert len(rows) == 1
        assert rows[0].episode_id == episode_id
        assert rows[0].ingested_to == {"mem0": ["n-1", "n-2"]}
        assert rows[0].payload == {"text": "I drink black coffee"}
        assert (rows[0].agent, rows[0].session) == ("lugo", "s_9")

    async def test_journal_rows_are_per_tenant(self, catalog):
        await catalog.journal("t1", Scope(subject="u_1"), {"text": "x"}, {"mem0": ["n-1"]})
        assert await catalog.journal_rows("t2", "u_1") == []
