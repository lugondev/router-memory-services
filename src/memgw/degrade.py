"""The degradation matrix.

One rule governs everything here:

    Degradation may reduce **quality**. It may never reduce **isolation** or
    **deletability**.

So a graph query may fall back to semantic search when the caller opts in, but a
scope dimension the provider cannot honour is always a 422 -- dropping one would
spill one end-user's memories into another's results -- and so is a missing
delete, because that is somebody's right to erasure.
"""

from __future__ import annotations

from pydantic import BaseModel

from memgw.capabilities import Capabilities
from memgw.errors import UnsupportedCapability
from memgw.types import OnUnsupported, Scope, SearchMode

#: The *only* permitted substitutions. A mode absent from this table is rejected
#: even under ``on_unsupported="degrade"``: serving semantic search in answer to a
#: keyword query is a quiet lie about what ran.
SUBSTITUTIONS: dict[SearchMode, tuple[SearchMode, list[str]]] = {
    "graph": ("semantic", ["graph_traversal"]),
    "temporal": ("semantic", ["fact_invalidation"]),
    "hybrid": ("semantic", ["keyword_match"]),
}


class DegradeResult(BaseModel):
    served: SearchMode
    degraded: bool
    lost: list[str]


def assert_as_of_supported(caps: Capabilities) -> None:
    """Never degrades, and not because point-in-time recall is precious.

    Silently dropping ``as_of`` answers a *different question* than the one asked --
    "what is true now" in place of "what was true then" -- and returns a confident,
    well-formed, wrong answer. A quality knob may be turned down; the question may
    not be swapped.
    """
    if "temporal" not in caps.search_modes:
        raise UnsupportedCapability(
            "provider cannot answer as-of a past time; serving the present instead "
            "would answer a different question",
            details={"available": list(caps.search_modes)},
        )


def resolve_mode(
    requested: SearchMode, caps: Capabilities, on_unsupported: OnUnsupported
) -> DegradeResult:
    if requested in caps.search_modes:
        return DegradeResult(served=requested, degraded=False, lost=[])

    if on_unsupported == "reject":
        raise UnsupportedCapability(
            f"provider does not support {requested!r} search",
            details={"requested": requested, "available": list(caps.search_modes)},
        )

    substitute = SUBSTITUTIONS.get(requested)
    if substitute is None or substitute[0] not in caps.search_modes:
        raise UnsupportedCapability(
            f"no permitted fallback for {requested!r} search",
            details={"requested": requested, "available": list(caps.search_modes)},
        )

    served, lost = substitute
    return DegradeResult(served=served, degraded=True, lost=list(lost))


def assert_scope_supported(
    scope: Scope, caps: Capabilities, on_unsupported: OnUnsupported = "reject"
) -> None:
    """Never degrades. ``on_unsupported`` is accepted only to make that explicit at
    every call site -- a caller asking for leniency does not get to lose isolation."""
    del on_unsupported

    missing = [dim for dim in scope.dims() if dim not in caps.scope_dims]
    if missing:
        raise UnsupportedCapability(
            "provider cannot honour every scope dimension; dropping one would leak "
            "memories across subjects",
            details={"missing_scope_dims": missing, "scope_dims": list(caps.scope_dims)},
        )

    if scope.labels and not caps.supports_labels:
        raise UnsupportedCapability(
            "provider does not support labels",
            details={"labels": sorted(scope.labels)},
        )


def assert_delete_supported(
    caps: Capabilities, on_unsupported: OnUnsupported = "reject", *, by_scope: bool = False
) -> None:
    """Never degrades. A silently skipped delete is a broken erasure promise."""
    del on_unsupported

    supported = caps.supports_delete_by_scope if by_scope else caps.supports_delete
    if not supported:
        raise UnsupportedCapability(
            "provider cannot delete by scope" if by_scope else "provider cannot delete",
            details={"by_scope": by_scope},
        )
