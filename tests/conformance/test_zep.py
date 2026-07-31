"""The Zep adapter.

Zep is the first provider whose shape genuinely differs from the contract, so
most of what is asserted here is *refusal*: the dimensions it cannot honour, and
the fact that it says so instead of quietly returning the wrong rows.

The live conformance run needs a hosted Zep, so it is opt-in via
``MEMGW_ZEP_TEST=1``. Everything else here runs with no dependency at all.
"""

from __future__ import annotations

import os

import pytest

from memgw.errors import UnsupportedCapability
from memgw.types import Episode, Scope, SearchQuery
from tests.conformance.suite import ConformanceSuite


class FakeZep:
    """A Zep stand-in, shaped against the **installed SDK** rather than against what
    the adapter happens to call.

    The first version of this fake was written from the docs at the same time as the
    adapter, so it shared the adapter's misconceptions and cheerfully answered calls
    that do not exist -- ``graph.delete_edge`` and ``graph.get_edge``, where the real
    client has ``graph.edge.delete`` and ``graph.edge.get``. Fourteen green tests, and
    three methods that would have raised ``AttributeError`` on first contact.

    So this fake now exposes *only* the surface ``zep_cloud.client.AsyncZep`` exposes,
    and ``TestTheFakeMatchesTheRealClient`` compares the two whenever the SDK is
    installed. A fake written from the same assumption as the code under test proves
    the assumption is self-consistent, which is not the same as proving it is right.
    """

    def __init__(self, *, processes: bool = True) -> None:
        self.episodes: list[dict] = []
        self.deleted_users: list[str] = []
        self.added_users: list[str] = []
        #: False reproduces the state the real service was in: every write accepted
        #: with an id, and processed=false forever.
        self.processes = processes
        self.graph = self._Graph(self)
        self.user = self._User(self)

    class _Edge:
        """``client.graph.edge`` -- a sub-client, not methods on ``graph``."""

        def __init__(self, outer):
            self._outer = outer

        async def delete(self, uuid_, **kw):
            del kw
            before = len(self._outer.episodes)
            self._outer.episodes = [e for e in self._outer.episodes if e["uuid_"] != uuid_]
            removed = before - len(self._outer.episodes)
            return type_("SuccessResponse", message="ok", deleted=removed)

        async def get(self, uuid_, **kw):
            del kw
            for ep in self._outer.episodes:
                if ep["uuid_"] == uuid_:
                    return _edge_of(ep)
            return None

        async def get_by_user_id(self, user_id, *, limit=None, **kw):
            del kw
            if not self._outer.processes:
                # No graph built means no edges, which is why reads came back empty.
                return []
            found = [_edge_of(e) for e in self._outer.episodes if e["user_id"] == user_id]
            return found[:limit] if limit else found

    class _Episode:
        """``client.graph.episode`` -- how the adapter asks whether Zep did the work."""

        def __init__(self, outer):
            self._outer = outer

        async def get(self, uuid_, **kw):
            del kw
            for ep in self._outer.episodes:
                if ep["uuid_"] == uuid_:
                    return type_("Episode", uuid_=uuid_, processed=self._outer.processes)
            return None

    class _Graph:
        def __init__(self, outer):
            self._outer = outer
            self.edge = FakeZep._Edge(outer)
            self.episode = FakeZep._Episode(outer)

        async def add(self, *, user_id=None, graph_id=None, type=None, data=None, **kw):
            del kw
            uuid = f"ep-{len(self._outer.episodes)}"
            self._outer.episodes.append(
                {
                    "uuid_": uuid,
                    "user_id": user_id,
                    "graph_id": graph_id,
                    "type": type,
                    "data": data,
                }
            )
            return type_("Episode", uuid_=uuid, content=data)

        async def search(
            self, *, query, user_id=None, graph_id=None, scope="edges", limit=10, **kw
        ):
            del graph_id, kw
            hits = [
                _edge_of(ep)
                for ep in self._outer.episodes
                if ep["user_id"] == user_id
                and any(word in (ep["data"] or "").lower() for word in query.lower().split())
            ]
            return type_(
                "Results",
                edges=hits[:limit] if scope == "edges" else [],
                nodes=[] if scope == "edges" else hits[:limit],
            )

    class _User:
        def __init__(self, outer):
            self._outer = outer

        async def add(self, *, user_id, **kw):
            del kw
            self._outer.added_users.append(user_id)
            return type_("User", user_id=user_id)

        async def delete(self, user_id, **kw):
            del kw
            self._outer.deleted_users.append(user_id)
            self._outer.episodes = [e for e in self._outer.episodes if e["user_id"] != user_id]
            # The real client answers with a SuccessResponse. An adapter that reads a
            # count out of this gets None, and reports nothing deleted.
            return type_("SuccessResponse", message="user deleted")


def type_(name, **attrs):
    return type(name, (), attrs)()


def _edge_of(episode: dict):
    """One place that decides what an edge looks like, so search and listing cannot
    drift apart in the fake the way they could in the adapter."""
    return type_(
        "EntityEdge",
        uuid_=episode["uuid_"],
        fact=episode["data"],
        score=0.9,
        created_at=None,
        valid_at=None,
        invalid_at=None,
    )


def adapter_with(client):
    from memgw.adapters.zep import ZepAdapter

    return ZepAdapter(client=client)


def adapter():
    return adapter_with(FakeZep())


# -- what Zep is ---------------------------------------------------------------


class TestDeclaredNature:
    def test_it_is_a_temporal_graph_not_a_pile_of_facts(self):
        caps = adapter().capabilities()
        assert caps.memory_model == "temporal_graph"

    def test_it_offers_graph_and_temporal_search(self):
        # The first adapter that does, which is what finally exercises the whole
        # degradation matrix against something real.
        caps = adapter().capabilities()
        assert {"graph", "temporal"} <= set(caps.search_modes)

    def test_it_declares_eventual_consistency(self):
        # Zep builds the graph asynchronously after the write returns.
        assert adapter().capabilities().consistency == "eventual"

    def test_it_admits_its_spend_is_invisible(self):
        assert adapter().capabilities().metered_externally is True


# -- what Zep is not -----------------------------------------------------------


class TestTheDimensionsZepDoesNotHave:
    def test_it_does_not_claim_a_session_dimension(self):
        # Zep writes to a thread but its graph search filters by user, not thread.
        # Declaring session and then ignoring it is precisely the silent-wrong-answer
        # this whole contract exists to prevent.
        caps = adapter().capabilities()
        assert "session" not in caps.scope_dims
        assert "agent" not in caps.scope_dims

    def test_it_does_claim_tenant_and_subject(self):
        caps = adapter().capabilities()
        assert "tenant" in caps.scope_dims
        assert "subject" in caps.scope_dims

    async def test_a_session_scoped_search_is_refused_rather_than_widened(self):
        from memgw.catalog import Catalog
        from memgw.core import MemoryCore

        catalog = Catalog("sqlite+aiosqlite:///:memory:")
        await catalog.init()
        core = MemoryCore(catalog=catalog, providers={"zep": adapter()}, default_provider="zep")
        with pytest.raises(UnsupportedCapability) as caught:
            await core.search("t", SearchQuery(query="coffee"), Scope(subject="u_1", session="s_9"))
        assert "session" in caught.value.details["missing_scope_dims"]
        await catalog.close()


# -- the tenant trick ----------------------------------------------------------


class TestTenantNamespacing:
    def test_the_user_id_carries_the_tenant(self):
        from memgw.adapters.zep import zep_user_id

        assert zep_user_id(Scope(subject="u_1", tenant="tenant-a")) == "tenant-a:u_1"

    def test_without_a_tenant_the_subject_stands_alone(self):
        from memgw.adapters.zep import zep_user_id

        assert zep_user_id(Scope(subject="u_1")) == "u_1"

    async def test_one_tenant_cannot_read_another_tenants_memories(self):
        a = adapter()
        await a.upsert(["blackcoffee"], Scope(subject="u_1", tenant="tenant-a"))
        leaked = await a.search(
            SearchQuery(query="blackcoffee"), Scope(subject="u_1", tenant="tenant-b")
        )
        assert leaked == []


# -- calls ---------------------------------------------------------------------


class TestItCallsZepCorrectly:
    async def test_a_write_reaches_the_graph_under_the_namespaced_user(self):
        a = adapter()
        await a.upsert(["drinks black coffee"], Scope(subject="u_1", tenant="tenant-a"))
        [episode] = a._client.episodes
        assert episode["user_id"] == "tenant-a:u_1"
        assert episode["data"] == "drinks black coffee"

    async def test_ingest_sends_the_conversation_as_one_episode(self):
        from memgw.types import Message

        a = adapter()
        await a.ingest(
            Episode(messages=[Message(role="user", content="I drink black coffee")]),
            Scope(subject="u_1"),
        )
        assert len(a._client.episodes) == 1

    async def test_an_undeclared_mode_is_refused(self):
        a = adapter()
        with pytest.raises(UnsupportedCapability):
            await a.search(SearchQuery(query="x", mode="keyword"), Scope(subject="u_1"))

    async def test_delete_scope_removes_the_namespaced_user(self):
        a = adapter()
        await a.upsert(["blackcoffee"], Scope(subject="u_1", tenant="tenant-a"))
        removed = await a.delete_scope(Scope(subject="u_1", tenant="tenant-a"))
        assert removed == 1
        assert a._client.deleted_users == ["tenant-a:u_1"]


class TestTheFakeMatchesTheRealClient:
    """The guard that would have caught this the first time.

    Tier-1 tests are only worth what the fake is worth. When the fake and the code
    are written from the same reading of the same docs, green means "self-consistent"
    and nothing more -- three adapter methods called ``graph.delete_edge`` and
    ``graph.get_edge``, neither of which exists, and fourteen tests passed anyway.

    So whenever the SDK is installed, every attribute the adapter reaches for is
    checked against the real client. No credentials, no network.
    """

    #: Every client attribute path the adapter uses.
    USED = [
        ("graph", "add"),
        ("graph", "search"),
        ("graph", "edge", "delete"),
        ("graph", "edge", "get"),
        ("graph", "edge", "get_by_user_id"),
        ("graph", "episode", "get"),
        ("user", "add"),
        ("user", "delete"),
    ]

    def _walk(self, root, path):
        for step in path:
            root = getattr(root, step, None)
            if root is None:
                return None
        return root

    def test_every_call_the_adapter_makes_exists_on_the_real_client(self):
        zep_cloud = pytest.importorskip("zep_cloud", reason="zep-cloud not installed")
        real = zep_cloud.client.AsyncZep(api_key="not-used-no-request-is-made")

        missing = [".".join(p) for p in self.USED if self._walk(real, p) is None]
        assert not missing, f"the adapter calls methods the SDK does not have: {missing}"

    def test_the_fake_offers_the_same_calls(self):
        missing = [".".join(p) for p in self.USED if self._walk(FakeZep(), p) is None]
        assert not missing, f"the fake is missing: {missing}"

    def test_the_fake_does_not_invent_calls_the_real_client_lacks(self):
        # The direction that actually bit: a fake generous enough to answer anything
        # makes a wrong adapter look right.
        zep_cloud = pytest.importorskip("zep_cloud", reason="zep-cloud not installed")
        real = zep_cloud.client.AsyncZep(api_key="not-used-no-request-is-made")

        fake = FakeZep()
        for group in ("graph", "user"):
            invented = [
                name
                for name in dir(getattr(fake, group))
                if not name.startswith("_") and not hasattr(getattr(real, group), name)
            ]
            assert not invented, f"fake {group} invents {invented}"


@pytest.mark.skipif(
    os.environ.get("MEMGW_ZEP_TEST") != "1",
    reason="needs a hosted Zep (ZEP_API_KEY); set MEMGW_ZEP_TEST=1 to run",
)
class TestZepConformance(ConformanceSuite):
    settle_timeout = 60.0

    async def make_adapter(self):
        from memgw.adapters.zep import ZepAdapter

        return ZepAdapter(api_key=os.environ["ZEP_API_KEY"])
