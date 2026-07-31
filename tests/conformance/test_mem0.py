"""Mem0 adapter.

The conformance run needs a real Mem0 (an LLM and an embedder, i.e. credentials),
so it is opt-in via ``MEMGW_MEM0_TEST=1``. The filter tests below are **not**
skippable and need no dependency at all -- they guard the one bug that makes Mem0
memory look like it works and recall a blank.
"""

from __future__ import annotations

import os
import uuid

import pytest

from memgw.adapters.mem0 import ANY, Mem0Adapter, build_filters, write_ids
from memgw.types import Scope
from tests.conformance.suite import ConformanceSuite

mem0 = pytest.importorskip("mem0", reason="mem0ai not installed")


class TestFiltersAreAlwaysComplete:
    """No credentials, no skipping. These are the trap."""

    def test_a_read_names_every_dimension(self):
        scope = Scope(subject="u_1", agent="lugo", session="s_9")
        assert build_filters(scope) == {"user_id": "u_1", "agent_id": "lugo", "run_id": "s_9"}

    def test_an_unset_dimension_becomes_a_wildcard_not_an_omission(self):
        # Omitting agent_id would mean "agent_id must be null", so a memory written
        # with an agent scope would be invisible to a subject-level recall -- with no
        # error at all. The wildcard is what keeps cross-session recall working.
        assert build_filters(Scope(subject="u_1")) == {
            "user_id": "u_1",
            "agent_id": ANY,
            "run_id": ANY,
        }

    def test_a_partial_scope_wildcards_only_what_is_open(self):
        assert build_filters(Scope(subject="u_1", agent="lugo")) == {
            "user_id": "u_1",
            "agent_id": "lugo",
            "run_id": ANY,
        }

    def test_the_subject_is_never_wildcarded(self):
        # A wildcard here would return every user's memories.
        for scope in (Scope(subject="u_1"), Scope(subject="u_1", session="s_9")):
            assert build_filters(scope)["user_id"] == "u_1"
            assert build_filters(scope)["user_id"] != ANY

    def test_writes_name_only_what_is_set(self):
        # A wildcard is meaningless on a write: you cannot store a memory under
        # "any agent".
        assert write_ids(Scope(subject="u_1")) == {"user_id": "u_1"}
        assert write_ids(Scope(subject="u_1", agent="lugo")) == {
            "user_id": "u_1",
            "agent_id": "lugo",
        }


class TestDeclaredNature:
    def test_it_admits_its_spend_is_invisible(self):
        caps = Mem0Adapter(client=object()).capabilities()
        assert caps.metered_externally is True

    def test_it_admits_it_cannot_be_exported(self):
        # Which is exactly why moving an end-user off Mem0 needs the journal.
        caps = Mem0Adapter(client=object()).capabilities()
        assert caps.supports_export is False

    def test_it_declares_eventual_consistency(self):
        caps = Mem0Adapter(client=object()).capabilities()
        assert caps.consistency == "eventual"


def live_config(collection: str) -> dict:
    """A Mem0 that actually starts. Three things pinned, each learned from a failure.

    *The model.* Mem0 sends ``temperature=0.1`` and ``top_p=0.1`` with every
    extraction, and a reasoning model rejects both with a 400. The default model is
    whatever Mem0 ships this week, so this pins one that takes them.

    *A Qdrant server, not local mode.* Local Qdrant persists through SQLite, and
    ``AsyncMemory`` dispatches every call with ``asyncio.to_thread`` -- so each call
    arrives on a different pool thread and SQLite refuses it. Local mode also flocks
    its directory, so a second adapter in the same process cannot open one at all.
    Neither is memgw's bug and neither is fixable from the adapter; the answer is a
    server. ``docker run -p 6333:6333 qdrant/qdrant``.

    *A collection per adapter.* Each test starts empty, which is what makes the
    isolation checks mean anything.
    """
    return {
        "llm": {
            "provider": "openai",
            "config": {"model": os.environ.get("MEMGW_MEM0_MODEL", "gpt-4o-mini")},
        },
        "embedder": {
            "provider": "openai",
            "config": {"model": "text-embedding-3-small"},
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "host": os.environ.get("MEMGW_QDRANT_HOST", "localhost"),
                "port": int(os.environ.get("MEMGW_QDRANT_PORT", "6333")),
                "collection_name": collection,
            },
        },
    }


@pytest.mark.skipif(
    os.environ.get("MEMGW_MEM0_TEST") != "1",
    reason="needs a real Mem0 (OPENAI_API_KEY + a Qdrant server); set MEMGW_MEM0_TEST=1",
)
class TestMem0Conformance(ConformanceSuite):
    #: Mem0 declares eventual consistency and means it -- extraction is an LLM call.
    settle_timeout = 30.0

    async def make_adapter(self):
        return Mem0Adapter(live_config(f"memgw_{uuid.uuid4().hex[:12]}"))
