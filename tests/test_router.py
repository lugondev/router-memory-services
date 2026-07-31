import pytest

from memgw.catalog import Catalog
from memgw.errors import InvalidRequest, ProviderMismatch
from memgw.router import resolve_provider


@pytest.fixture
async def catalog():
    cat = Catalog("sqlite+aiosqlite:///:memory:")
    await cat.init()
    yield cat
    await cat.close()


class TestResolution:
    async def test_a_binding_wins_over_the_tenant_default(self, catalog):
        await catalog.bind("t1", "u_1", "mem0")
        assert await resolve_provider(catalog, "t1", "u_1", default_provider="pgvector") == "mem0"

    async def test_an_unbound_subject_falls_back_to_the_default(self, catalog):
        assert (
            await resolve_provider(catalog, "t1", "u_1", default_provider="pgvector") == "pgvector"
        )

    async def test_no_binding_and_no_default_is_a_refusal_not_a_guess(self, catalog):
        # Picking one here would silently strand this end-user's memories in a
        # backend nobody chose.
        with pytest.raises(InvalidRequest) as exc:
            await resolve_provider(catalog, "t1", "u_1", default_provider=None)
        assert exc.value.code == "no_provider_resolved"
        assert exc.value.status == 400

    async def test_resolving_never_creates_a_binding(self, catalog):
        await resolve_provider(catalog, "t1", "u_1", default_provider="pgvector")
        assert await catalog.get_binding("t1", "u_1") is None, (
            "a read must not pin an end-user to a provider"
        )


class TestAssertion:
    async def test_an_agreeing_assertion_passes_through(self, catalog):
        await catalog.bind("t1", "u_1", "mem0")
        got = await resolve_provider(
            catalog, "t1", "u_1", default_provider="pgvector", asserted="mem0"
        )
        assert got == "mem0"

    async def test_a_disagreeing_assertion_is_loud(self, catalog):
        await catalog.bind("t1", "u_1", "mem0")
        with pytest.raises(ProviderMismatch) as exc:
            await resolve_provider(
                catalog, "t1", "u_1", default_provider="pgvector", asserted="pgvector"
            )
        assert exc.value.status == 409
        assert exc.value.details == {"asserted": "pgvector", "bound": "mem0"}

    async def test_an_assertion_against_the_default_agrees(self, catalog):
        got = await resolve_provider(
            catalog, "t1", "u_1", default_provider="pgvector", asserted="pgvector"
        )
        assert got == "pgvector"

    async def test_an_assertion_against_an_unbound_subject_still_checks_the_default(self, catalog):
        with pytest.raises(ProviderMismatch):
            await resolve_provider(
                catalog, "t1", "u_1", default_provider="pgvector", asserted="mem0"
            )
