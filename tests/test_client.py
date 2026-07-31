from __future__ import annotations

import httpx
import pytest

from memgw.catalog import Catalog
from memgw.client import Memory
from memgw.core import MemoryCore
from memgw.errors import GatewayError
from memgw.server import create_app
from memgw.types import Scope
from tests.conformance.test_pgvector import BagOfWordsEmbedder, IdentityExtractor
from tests.fake import FakeAdapter


@pytest.fixture
async def embedded():
    mem = Memory(
        provider="pgvector",
        config={
            "url": "sqlite+aiosqlite:///:memory:",
            "embedder": BagOfWordsEmbedder(),
            "extractor": IdentityExtractor(),
        },
        catalog_url="sqlite+aiosqlite:///:memory:",
    )
    yield mem
    await mem.close()


@pytest.fixture
async def proxy():
    catalog = Catalog("sqlite+aiosqlite:///:memory:")
    await catalog.init()
    core = MemoryCore(
        catalog=catalog,
        providers={"fake": FakeAdapter()},
        default_provider="fake",
    )
    app = create_app(core=core, api_keys={"k": "t1"})
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://memgw")
    mem = Memory(base_url="http://memgw", api_key="k", http=http)
    yield mem
    await http.aclose()
    await catalog.close()


class TestConstruction:
    def test_both_modes_at_once_is_refused(self):
        with pytest.raises(ValueError, match="pick one"):
            Memory(provider="pgvector", base_url="http://memgw")

    def test_neither_mode_is_refused(self):
        with pytest.raises(ValueError):
            Memory()

    def test_the_mode_is_visible(self):
        assert Memory(provider="pgvector", config={}).mode == "embedded"
        assert Memory(base_url="http://memgw").mode == "proxy"


class TestEmbedded:
    async def test_it_round_trips_through_a_real_adapter(self, embedded):
        await embedded.ingest(Scope(subject="u_1"), text="black coffee")
        result = await embedded.search(Scope(subject="u_1"), "coffee")
        assert [r.content for r in result.results] == ["black coffee"]

    async def test_capabilities_come_from_the_configured_instance(self, embedded):
        caps = await embedded.capabilities()
        assert caps.supports_ingest is True
        assert caps.memory_model == "flat_facts"


class TestProxy:
    async def test_it_round_trips_over_http(self, proxy):
        [written] = await proxy.ingest(Scope(subject="u_1"), text="black coffee")
        assert written.id.startswith("mg_")

        result = await proxy.search(Scope(subject="u_1"), "coffee")
        assert [r.content for r in result.results] == ["black coffee"]

    async def test_get_update_and_delete_work_over_http(self, proxy):
        [written] = await proxy.ingest(Scope(subject="u_1"), text="black coffee")

        assert (await proxy.get(written.id)).content == "black coffee"
        assert (await proxy.update(written.id, "green tea")).content == "green tea"

        await proxy.delete(written.id)
        with pytest.raises(GatewayError) as exc:
            await proxy.get(written.id)
        assert exc.value.status == 404

    async def test_a_gateway_error_arrives_as_the_same_exception_type(self, proxy):
        # Code written against embedded mode keeps working in proxy mode.
        await proxy.ingest(Scope(subject="u_1"), text="black coffee")
        with pytest.raises(GatewayError) as exc:
            await proxy.search(Scope(subject="u_1"), "coffee", mode="graph")
        assert exc.value.code == "unsupported_capability"
        assert exc.value.status == 422

    async def test_capabilities_come_over_the_wire(self, proxy):
        caps = await proxy.capabilities()
        assert caps.memory_model == "flat_facts"


class TestScopeHandle:
    async def test_search_without_a_session_recalls_across_sessions(self, embedded):
        person = embedded.scope("u_1", agent="lugo")
        await person.ingest(text="coffee one", session="s_1")
        await person.ingest(text="coffee two", session="s_2")

        # This is the query memory exists for. A composite key that baked the
        # session in could not express it.
        everything = await person.search("coffee")
        assert {r.content for r in everything.results} == {"coffee one", "coffee two"}

        just_one = await person.search("coffee", session="s_1")
        assert {r.content for r in just_one.results} == {"coffee one"}

    async def test_messages_may_be_plain_dicts(self, embedded):
        person = embedded.scope("u_1")
        await person.ingest([{"role": "user", "content": "I drink black coffee"}])
        result = await person.search("coffee")
        assert result.results

    async def test_delete_all_clears_the_subject(self, embedded):
        person = embedded.scope("u_1")
        await person.ingest(text="coffee one")
        await person.ingest(text="coffee two")

        assert await person.delete_all() == 2
        assert (await person.search("coffee")).results == []


class TestParseScope:
    def test_it_splits_a_composite_key(self):
        scope = Memory.parse_scope("t/u_1/s_9", "{tenant}/{subject}/{session}")
        assert (scope.subject, scope.session) == ("u_1", "s_9")

    def test_the_tenant_segment_is_parsed_but_not_carried(self):
        # Tenant comes from the credential. A tenant in a key is data, not authority,
        # so the segment is consumed by the pattern and then dropped: the gateway
        # stamps the real one on the way down.
        scope = Memory.parse_scope("t/u_1/s_9", "{tenant}/{subject}/{session}")
        assert scope.tenant is None
        assert scope.subject == "u_1"

    def test_a_key_that_does_not_match_is_refused(self):
        with pytest.raises(ValueError, match="does not match"):
            Memory.parse_scope("u_1", "{tenant}/{subject}/{session}")

    def test_a_format_without_a_subject_is_refused(self):
        with pytest.raises(ValueError, match="subject"):
            Memory.parse_scope("t/s_9", "{tenant}/{session}")
