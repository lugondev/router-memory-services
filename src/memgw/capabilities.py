"""What a *configured* provider can actually do.

Capabilities are instance-level, not class-level: configuration changes the
answer. Mem0 OSS without a graph store has no ``graph`` mode; the pgvector
adapter without an extractor reports ``supports_ingest=False`` and can only take
ready-made facts. So ``capabilities()`` is a method on a built adapter, never a
constant on its class.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from memgw.types import SearchMode

MemoryModel = Literal["flat_facts", "temporal_graph", "memory_blocks", "documents"]
Dedup = Literal["none", "provider", "gateway"]
Consistency = Literal["read_after_write", "eventual"]


class Capabilities(BaseModel):
    # write
    supports_ingest: bool
    supports_upsert: bool
    supports_update: bool
    supports_delete: bool
    supports_delete_by_scope: bool

    # read
    supports_list: bool = False
    """Can enumerate a scope without a query. Separate from ``supports_export``: Mem0
    lists fine but has no bulk export, and "show me everything you hold on this
    person" is the question a support ticket and a subject access request both ask."""

    search_modes: list[SearchMode]
    supports_score: bool
    max_limit: int = Field(gt=0)

    # scope
    scope_dims: list[Literal["tenant", "subject", "agent", "session"]]
    """``tenant`` belongs here and not in the gateway's own bookkeeping: subject ids
    are chosen by tenants, so an adapter that filters only by subject serves two
    tenants the same row the moment they both name an end-user ``u_1``. An adapter
    that cannot express the dimension must omit it and be refused, loudly."""

    supports_labels: bool

    # nature -- states that providers differ in *kind*, not in feature checklist
    memory_model: MemoryModel
    dedup: Dedup

    # portability -- with export, migration is a copy; without it, a journal replay
    supports_export: bool
    supports_import: bool

    # operations
    consistency: Consistency
    """``eventual`` means ingest-then-search returns nothing for a while. Mem0 and Zep
    both behave this way and neither says so plainly; it is the root cause of flaky
    integration tests across the whole category."""

    experimental: bool = False
    """True when this adapter has never passed the live conformance suite.

    Every provider is selectable by one name in one list, which quietly implies they
    are peers. They are not: some have been run against the real service and some
    have only been checked against its SDK. A caller choosing a backend for other
    people's memories is entitled to know which kind they are picking.
    """

    metered_externally: bool
    """True when the provider makes its own LLM/embedding calls inside add()/search().
    The spend is then invisible to any ledger the caller controls -- the gateway knows
    the call count, not the cost. Disclosure is the entire remedy."""
