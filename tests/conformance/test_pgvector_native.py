"""The self-hosted adapter on real Postgres, with a real ``vector`` column.

Until now the adapter called itself *pgvector* and used none of it: embeddings
were stored as JSON, every row in scope was pulled into Python, and cosine was
computed a float at a time. Correct, and wrong at any size -- a subject with ten
thousand memories shipped ten thousand vectors over the wire to answer one query.

These tests need a Postgres with the extension, so they are opt-in:

    docker compose -f docker-compose.test.yml up -d
    MEMGW_PG_TEST=1 pytest tests/conformance/test_pgvector_native.py
"""

from __future__ import annotations

import os
import uuid

import pytest

from memgw.types import Scope, SearchQuery
from tests.conformance.suite import ConformanceSuite
from tests.conformance.test_pgvector import BagOfWordsEmbedder, IdentityExtractor

PG_URL = os.environ.get("MEMGW_PG_URL", "postgresql+asyncpg://memgw:memgw@localhost:55432/memgw")

live = pytest.mark.skipif(
    os.environ.get("MEMGW_PG_TEST") != "1",
    reason="needs Postgres with the vector extension; set MEMGW_PG_TEST=1",
)


async def make(**kw):
    from memgw.adapters.pgvector import PgvectorAdapter

    adapter = PgvectorAdapter(
        PG_URL,
        embedder=BagOfWordsEmbedder(),
        extractor=IdentityExtractor(),
        table=f"memgw_t_{uuid.uuid4().hex[:10]}",
        **kw,
    )
    await adapter.init()
    return adapter


@live
class TestPgvectorNativeConformance(ConformanceSuite):
    async def make_adapter(self):
        return await make()


@live
class TestItActuallyUsesTheExtension:
    async def test_the_embedding_column_is_a_vector_not_json(self):
        from sqlalchemy import text

        adapter = await make()
        async with adapter.engine.connect() as conn:
            found = await conn.execute(
                text(
                    "select udt_name from information_schema.columns "
                    "where table_name = :t and column_name = 'embedding'"
                ),
                {"t": adapter.table_name},
            )
            assert found.scalar_one() == "vector"
        await adapter.close()

    async def test_ranking_happens_in_the_database(self):
        """The point of the change: the limit is applied by Postgres.

        With scoring in Python the adapter had to read every row in scope before it
        could drop any, so `limit` saved nothing at all.
        """
        adapter = await make()
        scope = Scope(subject="u_1", tenant="t")
        await adapter.upsert([f"coffee number {i}" for i in range(50)], scope)

        plan = await adapter.explain_search(SearchQuery(query="coffee", limit=3), scope)
        assert "Limit" in plan, plan
        assert "<=>" in plan or "vector" in plan.lower(), plan
        await adapter.close()

    async def test_the_closest_memory_still_ranks_first(self):
        adapter = await make()
        scope = Scope(subject="u_1", tenant="t")
        await adapter.upsert(["black coffee no sugar", "prefers loud music"], scope)

        hits = await adapter.search(SearchQuery(query="black coffee"), scope)
        assert hits[0].content == "black coffee no sugar"
        assert hits[0].score is not None and hits[0].score > 0
        await adapter.close()

    async def test_a_score_is_still_a_similarity_and_not_a_distance(self):
        # pgvector's <=> returns cosine *distance*: 0 is identical. Handing that back
        # as a score would invert every ranking a caller applies min_score to.
        adapter = await make()
        scope = Scope(subject="u_1", tenant="t")
        await adapter.upsert(["black coffee"], scope)

        [hit] = await adapter.search(SearchQuery(query="black coffee"), scope)
        assert hit.score is not None
        assert 0.9 < hit.score <= 1.0, f"identical text scored {hit.score}"
        await adapter.close()

    async def test_min_score_filters_on_similarity(self):
        adapter = await make()
        scope = Scope(subject="u_1", tenant="t")
        await adapter.upsert(["black coffee no sugar", "prefers loud music"], scope)

        strict = await adapter.search(SearchQuery(query="black coffee", min_score=0.5), scope)
        assert [h.content for h in strict] == ["black coffee no sugar"]
        await adapter.close()

    async def test_tenants_stay_isolated_on_the_native_path(self):
        adapter = await make()
        await adapter.upsert(["salary is 999"], Scope(subject="u_1", tenant="tenant-a"))
        leaked = await adapter.search(
            SearchQuery(query="salary"), Scope(subject="u_1", tenant="tenant-b")
        )
        assert leaked == []
        await adapter.close()


@live
class TestTheDimensionIsPartOfTheSchema:
    async def test_an_embedder_whose_dimension_changed_is_refused_not_silently_wrong(self):
        """A ``vector(n)`` column has a fixed width.

        Swapping the embedding model changes n, and a mismatch has to be an error at
        startup. The alternative is a store holding two incompatible geometries, whose
        only symptom is recall quietly getting worse.
        """
        from memgw.adapters.pgvector import PgvectorAdapter

        table = f"memgw_t_{uuid.uuid4().hex[:10]}"
        first = PgvectorAdapter(
            PG_URL, embedder=BagOfWordsEmbedder(), table=table, extractor=IdentityExtractor()
        )
        await first.init()
        await first.close()

        class Wider(BagOfWordsEmbedder):
            dimension = 128

        second = PgvectorAdapter(PG_URL, embedder=Wider(), table=table)
        with pytest.raises(ValueError, match="dimension"):
            await second.init()
        await second.close()


class TestSqliteStillWorksWithoutPostgres:
    async def test_the_sqlite_path_is_unchanged_and_needs_no_extension(self):
        from memgw.adapters.pgvector import PgvectorAdapter

        adapter = PgvectorAdapter(
            "sqlite+aiosqlite:///:memory:",
            embedder=BagOfWordsEmbedder(),
            extractor=IdentityExtractor(),
        )
        await adapter.init()
        scope = Scope(subject="u_1", tenant="t")
        await adapter.upsert(["black coffee no sugar", "prefers loud music"], scope)

        hits = await adapter.search(SearchQuery(query="black coffee"), scope)
        assert hits[0].content == "black coffee no sugar"
        await adapter.close()

    async def test_it_says_which_engine_it_is_running(self):
        from memgw.adapters.pgvector import PgvectorAdapter

        adapter = PgvectorAdapter("sqlite+aiosqlite:///:memory:", embedder=BagOfWordsEmbedder())
        assert adapter.native_vectors is False
        await adapter.close()
