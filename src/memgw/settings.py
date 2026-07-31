"""Configuration, and the defaults that make an API key enough.

Two rules shape this module.

**Fail at startup, not at 3am.** Everything checkable is checked here: an unknown
provider name, a default that is not in the provider list, a provider whose
credentials are missing. A gateway that boots happily and then answers the first
real request with a 400 has moved the failure from a deploy you were watching to
a page you were not.

**Defaults encode what we learned the hard way.** Mem0's own defaults produce a
400 on the first extraction and a lock-up on the first concurrent write. The
config built here pins around both. Every default in this file that looks
arbitrary is a bug someone already had.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from memgw.embedding import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EXTRACTION_MODEL,
    OpenAIEmbedder,
    OpenAIExtractor,
)

#: Providers this build knows how to construct from environment variables alone.
#: A provider absent here can still be wired in Python; it just cannot be summoned
#: by setting a variable, which is the difference between "supported" and "shipped".
KNOWN_PROVIDERS = ("pgvector", "mem0", "zep")

#: Providers that ship but have never passed the live conformance suite. Kept as a
#: plain name list so the warning works before any adapter is constructed -- which is
#: the point, since constructing one needs credentials.
EXPERIMENTAL = ("zep",)

DEFAULT_CATALOG_URL = "sqlite+aiosqlite:///memgw.db"
DEFAULT_PGVECTOR_URL = "sqlite+aiosqlite:///memgw_pgvector.db"

#: Which environment variable each provider cannot start without.
REQUIRED_KEYS = {
    "pgvector": ("OPENAI_API_KEY", "the built-in embedder needs it"),
    "mem0": ("OPENAI_API_KEY", "Mem0 runs its own LLM and embedder through OpenAI"),
    "zep": ("ZEP_API_KEY", "Zep is a hosted service"),
}


class Settings(BaseModel):
    providers: list[str] = Field(default_factory=lambda: ["pgvector"])
    default_provider: str = "pgvector"
    api_keys: dict[str, str] = Field(default_factory=dict)

    catalog_url: str = DEFAULT_CATALOG_URL
    journal: bool = False

    openai_api_key: str | None = None
    openai_base_url: str | None = None
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dimension: int | None = None
    extraction_model: str = DEFAULT_EXTRACTION_MODEL

    pgvector_url: str = DEFAULT_PGVECTOR_URL

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "memgw"

    zep_api_key: str | None = None

    docs: bool = True
    """Serve /docs and /redoc. Off is the right answer for a public deployment."""

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        e = dict(os.environ if env is None else env)

        providers = _csv(e.get("MEMGW_PROVIDERS", "pgvector"))
        unknown = [p for p in providers if p not in KNOWN_PROVIDERS]
        if unknown:
            raise ValueError(
                f"unknown provider(s) {unknown}; this build can configure "
                f"{list(KNOWN_PROVIDERS)} from the environment"
            )
        if not providers:
            raise ValueError("MEMGW_PROVIDERS is empty; name at least one provider")

        default_provider = e.get("MEMGW_DEFAULT_PROVIDER", providers[0])
        if default_provider not in providers:
            raise ValueError(
                f"MEMGW_DEFAULT_PROVIDER={default_provider!r} is not in MEMGW_PROVIDERS "
                f"{providers}; nothing would ever route to it"
            )

        settings = cls(
            providers=providers,
            default_provider=default_provider,
            api_keys=_api_keys(e.get("MEMGW_API_KEYS", "")),
            catalog_url=e.get("MEMGW_CATALOG_URL", DEFAULT_CATALOG_URL),
            journal=_flag(e.get("MEMGW_JOURNAL")),
            openai_api_key=e.get("OPENAI_API_KEY"),
            openai_base_url=e.get("OPENAI_BASE_URL"),
            embedding_model=e.get("MEMGW_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            embedding_dimension=_int(e.get("MEMGW_EMBEDDING_DIMENSION")),
            extraction_model=e.get("MEMGW_EXTRACTION_MODEL", DEFAULT_EXTRACTION_MODEL),
            pgvector_url=e.get("MEMGW_PGVECTOR_URL", DEFAULT_PGVECTOR_URL),
            qdrant_host=e.get("MEMGW_QDRANT_HOST", "localhost"),
            qdrant_port=int(e.get("MEMGW_QDRANT_PORT", "6333")),
            qdrant_collection=e.get("MEMGW_QDRANT_COLLECTION", "memgw"),
            zep_api_key=e.get("ZEP_API_KEY"),
            docs=_flag(e.get("MEMGW_DOCS", "true")),
        )
        settings._check_credentials()
        return settings

    def _check_credentials(self) -> None:
        for provider in self.providers:
            var, why = REQUIRED_KEYS[provider]
            if not getattr(self, var.lower(), None):
                raise ValueError(f"provider {provider!r} needs {var}: {why}")

    # -- derived ---------------------------------------------------------------

    def embedder(self) -> OpenAIEmbedder:
        return OpenAIEmbedder(
            api_key=self.openai_api_key,
            model=self.embedding_model,
            dimension=self.embedding_dimension,
        )

    def extractor(self) -> OpenAIExtractor:
        return OpenAIExtractor(api_key=self.openai_api_key, model=self.extraction_model)


def mem0_config(settings: Settings) -> dict[str, Any]:
    """Mem0, configured around its own defaults.

    *The model is pinned.* Mem0 sends ``temperature=0.1`` and ``top_p=0.1`` with
    every extraction; a reasoning model rejects both with a 400. Leaving the model
    unset means the gateway breaks the week Mem0 changes its default.

    *The vector store is a server.* Mem0 defaults to a local file-backed Qdrant,
    which persists through SQLite while ``AsyncMemory`` dispatches every call on a
    different thread-pool thread, and which takes an exclusive lock on its
    directory. The first is an error under concurrency; the second is a hang.

    Also set ``MEM0_TELEMETRY=False`` in the environment -- telemetry opens a second,
    undeclared local Qdrant under ``~/.mem0`` and walks into the same lock.
    """
    return {
        "llm": {
            "provider": "openai",
            "config": {"model": settings.extraction_model, "api_key": settings.openai_api_key},
        },
        "embedder": {
            "provider": "openai",
            "config": {"model": settings.embedding_model, "api_key": settings.openai_api_key},
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "host": settings.qdrant_host,
                "port": settings.qdrant_port,
                "collection_name": settings.qdrant_collection,
            },
        },
    }


async def build_adapter(name: str, settings: Settings) -> Any:
    if name == "pgvector":
        from memgw.adapters.pgvector import PgvectorAdapter

        adapter = PgvectorAdapter(
            settings.pgvector_url,
            embedder=settings.embedder(),
            extractor=settings.extractor(),
        )
        await adapter.init()
        return adapter

    if name == "mem0":
        from memgw.adapters.mem0 import Mem0Adapter

        return Mem0Adapter(mem0_config(settings))

    if name == "zep":
        from memgw.adapters.zep import ZepAdapter

        return ZepAdapter(api_key=settings.zep_api_key)

    raise ValueError(f"no environment recipe for provider {name!r}")


async def build_core(settings: Settings):
    from memgw.catalog import Catalog
    from memgw.core import MemoryCore

    catalog = Catalog(settings.catalog_url)
    await catalog.init()

    providers = {name: await build_adapter(name, settings) for name in settings.providers}
    return MemoryCore(
        catalog=catalog,
        providers=providers,
        default_provider=settings.default_provider,
        journal_enabled=settings.journal,
    )


def create_app_from_settings(settings: Settings):
    """Synchronous on purpose: an ASGI app has to be constructible without a running
    event loop, because that is how ``uvicorn --factory`` constructs it. The
    databases are opened during the app's lifespan instead."""
    from memgw.server import create_app

    return create_app(
        core_factory=lambda: build_core(settings),
        api_keys=settings.api_keys,
        docs=settings.docs,
    )


# -- parsing -------------------------------------------------------------------


def _csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _api_keys(raw: str) -> dict[str, str]:
    """``key:tenant,key:tenant`` -- or bare keys, which land on one default tenant.

    The bare form exists because most first deployments have exactly one tenant, and
    making them invent a name for it is friction with nothing on the other side.
    """
    out: dict[str, str] = {}
    for entry in _csv(raw):
        key, _, tenant = entry.partition(":")
        out[key.strip()] = tenant.strip() or "default"
    return out


def _flag(raw: str | None) -> bool:
    return (raw or "").strip().lower() in ("1", "true", "yes", "on")


def _int(raw: str | None) -> int | None:
    return int(raw) if raw and raw.strip() else None
