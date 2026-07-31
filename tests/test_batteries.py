"""The batteries: what has to exist before "plug in an API key and go" is true.

Before these, running the self-hosted adapter meant writing an ``Embedder`` class,
and running the gateway meant writing a module that constructs one. Both are
reasonable asks of a library and unreasonable asks of a product.
"""

from __future__ import annotations

import pytest

from memgw.types import Episode, Message

# -- built-in embedder / extractor --------------------------------------------


class _FakeOpenAI:
    """Stands in for the OpenAI client, recording what it was asked for.

    Faithful to the two response shapes the real SDK returns, which is the whole
    point: what these tests can prove is that we call it correctly and read the
    answer correctly. That the real service answers this shape is what the live
    conformance run is for.
    """

    def __init__(self, dimension: int = 8, reply: str | None = None) -> None:
        self.dimension = dimension
        self.reply = reply
        self.embed_calls: list[list[str]] = []
        self.chat_calls: list[dict] = []
        self.embeddings = self._Embeddings(self)
        self.chat = self._Chat(self)

    class _Embeddings:
        def __init__(self, outer):
            self._outer = outer

        async def create(self, *, model, input, **kw):
            del model, kw
            self._outer.embed_calls.append(list(input))
            data = [
                type("E", (), {"embedding": [float(i)] * self._outer.dimension})()
                for i in range(len(input))
            ]
            return type("R", (), {"data": data})()

    class _Chat:
        def __init__(self, outer):
            self.completions = self
            self._outer = outer

        async def create(self, **kw):
            self._outer.chat_calls.append(kw)
            content = self._outer.reply
            message = type("M", (), {"content": content})()
            return type("R", (), {"choices": [type("C", (), {"message": message})()]})()


class TestOpenAIEmbedder:
    async def test_it_embeds_a_batch_in_one_call(self):
        from memgw.embedding import OpenAIEmbedder

        client = _FakeOpenAI()
        embedder = OpenAIEmbedder(client=client, dimension=8)

        vectors = await embedder.embed(["one", "two", "three"])
        assert len(vectors) == 3
        assert len(client.embed_calls) == 1, "a batch must not become three round trips"
        assert client.embed_calls[0] == ["one", "two", "three"]

    async def test_it_declares_the_dimension_of_the_model_it_uses(self):
        from memgw.embedding import OpenAIEmbedder

        def dimension_of(model: str) -> int:
            return OpenAIEmbedder(client=_FakeOpenAI(), model=model).dimension

        assert dimension_of("text-embedding-3-small") == 1536
        assert dimension_of("text-embedding-3-large") == 3072

    async def test_an_unknown_model_must_be_told_its_dimension(self):
        from memgw.embedding import OpenAIEmbedder

        with pytest.raises(ValueError, match="dimension"):
            OpenAIEmbedder(client=_FakeOpenAI(), model="some-future-model")

    async def test_embedding_nothing_costs_no_call(self):
        from memgw.embedding import OpenAIEmbedder

        client = _FakeOpenAI()
        assert await OpenAIEmbedder(client=client, dimension=8).embed([]) == []
        assert client.embed_calls == []


class TestOpenAIExtractor:
    async def test_it_returns_the_facts_the_model_chose(self):
        from memgw.embedding import OpenAIExtractor

        client = _FakeOpenAI(reply='{"facts": ["drinks black coffee", "lives in Hanoi"]}')
        facts = await OpenAIExtractor(client=client).extract(
            Episode(messages=[Message(role="user", content="I drink black coffee in Hanoi")])
        )
        assert facts == ["drinks black coffee", "lives in Hanoi"]

    async def test_keeping_nothing_is_a_valid_answer(self):
        # Most turns are not worth remembering. This is the behaviour that a real
        # Mem0 showed us and that the conformance suite had wrong.
        from memgw.embedding import OpenAIExtractor

        client = _FakeOpenAI(reply='{"facts": []}')
        assert await OpenAIExtractor(client=client).extract(Episode(text="ok thanks")) == []

    async def test_a_model_that_answers_with_nonsense_keeps_nothing_rather_than_crashing(self):
        # An extractor that raises turns one bad LLM reply into a failed ingest and a
        # 424 the caller can do nothing about. Keeping nothing is the safe reading.
        from memgw.embedding import OpenAIExtractor

        client = _FakeOpenAI(reply="I'm sorry, I can't help with that.")
        assert await OpenAIExtractor(client=client).extract(Episode(text="hello")) == []

    async def test_it_never_stores_more_than_the_model_returned(self):
        from memgw.embedding import OpenAIExtractor

        client = _FakeOpenAI(reply='{"facts": ["a", "", "  ", "b"]}')
        assert await OpenAIExtractor(client=client).extract(Episode(text="x")) == ["a", "b"]


# -- configuration from the environment ---------------------------------------


class TestSettingsFromEnvironment:
    def test_the_smallest_working_configuration_is_one_api_key(self, monkeypatch):
        from memgw.settings import Settings

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        settings = Settings.from_env({"OPENAI_API_KEY": "sk-test"})
        assert settings.providers == ["pgvector"]
        assert settings.default_provider == "pgvector"
        assert settings.openai_api_key == "sk-test"

    def test_api_keys_parse_into_a_key_to_tenant_map(self):
        from memgw.settings import Settings

        settings = Settings.from_env(
            {"OPENAI_API_KEY": "sk-test", "MEMGW_API_KEYS": "k1:tenant-a,k2:tenant-b"}
        )
        assert settings.api_keys == {"k1": "tenant-a", "k2": "tenant-b"}

    def test_a_single_key_with_no_tenant_gets_a_default_tenant(self):
        from memgw.settings import Settings

        settings = Settings.from_env({"OPENAI_API_KEY": "sk-test", "MEMGW_API_KEYS": "k1"})
        assert settings.api_keys == {"k1": "default"}

    def test_a_malformed_provider_list_is_refused_at_startup(self):
        from memgw.settings import Settings

        with pytest.raises(ValueError, match="zep"):
            Settings.from_env({"OPENAI_API_KEY": "sk-test", "MEMGW_PROVIDERS": "pgvector,zepp"})

    def test_a_default_provider_outside_the_list_is_refused_at_startup(self):
        # Better a startup failure than a 400 on the first request of the day.
        from memgw.settings import Settings

        with pytest.raises(ValueError, match="DEFAULT_PROVIDER"):
            Settings.from_env(
                {
                    "OPENAI_API_KEY": "sk-test",
                    "MEMGW_PROVIDERS": "pgvector",
                    "MEMGW_DEFAULT_PROVIDER": "mem0",
                }
            )

    def test_pgvector_without_an_openai_key_is_refused_with_a_useful_message(self):
        from memgw.settings import Settings

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            Settings.from_env({"MEMGW_PROVIDERS": "pgvector"})


class TestBuildingFromSettings:
    async def test_one_key_gets_a_working_pgvector_gateway(self):
        from memgw.settings import Settings, build_core

        core = await build_core(
            Settings.from_env(
                {
                    "OPENAI_API_KEY": "sk-test",
                    "MEMGW_CATALOG_URL": "sqlite+aiosqlite:///:memory:",
                    "MEMGW_PGVECTOR_URL": "sqlite+aiosqlite:///:memory:",
                }
            )
        )
        caps = core.capabilities("pgvector")
        assert caps.supports_ingest is True, "an extractor should be wired from the same key"
        assert "tenant" in caps.scope_dims
        await core.catalog.close()

    async def test_the_app_is_built_without_writing_a_module(self):
        from memgw.settings import Settings, create_app_from_settings

        app = create_app_from_settings(
            Settings.from_env(
                {
                    "OPENAI_API_KEY": "sk-test",
                    "MEMGW_API_KEYS": "k1:tenant-a",
                    "MEMGW_CATALOG_URL": "sqlite+aiosqlite:///:memory:",
                    "MEMGW_PGVECTOR_URL": "sqlite+aiosqlite:///:memory:",
                }
            )
        )
        # The OpenAPI document, not app.routes: it is the surface callers actually
        # get, and it stays true however FastAPI nests routers internally.
        paths = app.openapi()["paths"]
        assert "/v1/memories:search" in paths
        assert "/v1/memories:ingest" in paths

    async def test_the_interactive_docs_can_be_closed_for_production(self):
        # /docs and /redoc need no credential. They leak only the shape of the API,
        # but a gateway in front of other people's memories should be able to say no.
        from memgw.settings import Settings, create_app_from_settings

        base = {
            "OPENAI_API_KEY": "sk-test",
            "MEMGW_CATALOG_URL": "sqlite+aiosqlite:///:memory:",
            "MEMGW_PGVECTOR_URL": "sqlite+aiosqlite:///:memory:",
        }
        open_app = create_app_from_settings(Settings.from_env(base))
        closed = create_app_from_settings(Settings.from_env({**base, "MEMGW_DOCS": "false"}))

        assert open_app.docs_url == "/docs"
        assert closed.docs_url is None
        assert closed.redoc_url is None


class TestMem0DefaultsAvoidTheKnownTraps:
    def test_the_llm_is_pinned_rather_than_left_to_mem0(self):
        # Mem0 sends temperature=0.1 with every extraction and a reasoning model
        # answers 400. Leaving the model unset is a 400 waiting for a release.
        from memgw.settings import Settings, mem0_config

        config = mem0_config(Settings.from_env({"OPENAI_API_KEY": "sk-test"}))
        assert config["llm"]["config"]["model"]

    def test_the_vector_store_is_a_server_not_a_local_file(self):
        # Local Qdrant persists through SQLite while AsyncMemory dispatches across
        # threads, and it flocks its directory. Both are silent hangs in production.
        from memgw.settings import Settings, mem0_config

        config = mem0_config(Settings.from_env({"OPENAI_API_KEY": "sk-test"}))
        assert "host" in config["vector_store"]["config"]
        assert "path" not in config["vector_store"]["config"]
