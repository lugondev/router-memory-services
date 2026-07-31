"""The conformance suite against the self-hosted adapter, on SQLite."""

from __future__ import annotations

import pytest

from memgw.adapters.pgvector import PgvectorAdapter
from memgw.errors import UnsupportedCapability
from memgw.types import Episode, Scope
from tests.conformance.suite import ConformanceSuite

DIM = 64


class BagOfWordsEmbedder:
    """Deterministic and dependency-free: a token lands in a fixed bucket, so cosine
    similarity is token overlap. Good enough to exercise ranking without an API key."""

    dimension = DIM

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vector = [0.0] * DIM
            for token in text.lower().split():
                vector[hash(token) % DIM] += 1.0
            out.append(vector)
        return out


class IdentityExtractor:
    async def extract(self, episode: Episode) -> list[str]:
        return [episode.as_text()]


async def make(*, extractor: bool = True) -> PgvectorAdapter:
    adapter = PgvectorAdapter(
        "sqlite+aiosqlite:///:memory:",
        embedder=BagOfWordsEmbedder(),
        extractor=IdentityExtractor() if extractor else None,
    )
    await adapter.init()
    return adapter


class TestPgvectorConformance(ConformanceSuite):
    async def make_adapter(self):
        return await make()


class TestCapabilitiesFollowConfiguration:
    async def test_without_an_extractor_it_says_it_cannot_ingest(self):
        adapter = await make(extractor=False)
        assert adapter.capabilities().supports_ingest is False

    async def test_without_an_extractor_ingest_refuses_rather_than_storing_transcript(self):
        adapter = await make(extractor=False)
        with pytest.raises(UnsupportedCapability):
            await adapter.ingest(Episode(text="I drink black coffee"), Scope(subject="u_1"))

    async def test_without_an_extractor_facts_still_go_in(self):
        adapter = await make(extractor=False)
        written = await adapter.upsert(["prefers black coffee"], Scope(subject="u_1"))
        assert len(written) == 1

    async def test_with_an_extractor_it_says_it_can_ingest(self):
        adapter = await make()
        assert adapter.capabilities().supports_ingest is True


class TestScopeFiltering:
    async def test_a_session_filter_narrows_and_no_session_recalls_across_sessions(self):
        adapter = await make()
        scope_a = Scope(subject="u_1", agent="lugo", session="s_1")
        scope_b = Scope(subject="u_1", agent="lugo", session="s_2")
        await adapter.upsert(["coffee one"], scope_a)
        await adapter.upsert(["coffee two"], scope_b)

        from memgw.types import SearchQuery

        only_s1 = await adapter.search(SearchQuery(query="coffee"), scope_a)
        assert {hit.content for hit in only_s1} == {"coffee one"}

        # The query memory exists for: everything about this user, no session.
        both = await adapter.search(SearchQuery(query="coffee"), Scope(subject="u_1", agent="lugo"))
        assert {hit.content for hit in both} == {"coffee one", "coffee two"}

    async def test_labels_narrow_the_result(self):
        from memgw.types import SearchQuery

        adapter = await make()
        await adapter.upsert(["pro note"], Scope(subject="u_1", labels={"tier": "pro"}))
        await adapter.upsert(["free note"], Scope(subject="u_1", labels={"tier": "free"}))

        hits = await adapter.search(
            SearchQuery(query="note"), Scope(subject="u_1", labels={"tier": "pro"})
        )
        assert {hit.content for hit in hits} == {"pro note"}


class TestRanking:
    async def test_the_closer_memory_ranks_first(self):
        from memgw.types import SearchQuery

        adapter = await make()
        scope = Scope(subject="u_1")
        await adapter.upsert(["black coffee no sugar"], scope)
        await adapter.upsert(["prefers loud music"], scope)

        hits = await adapter.search(SearchQuery(query="black coffee"), scope)
        assert hits[0].content == "black coffee no sugar"
        assert hits[0].score is not None and hits[0].score > 0
