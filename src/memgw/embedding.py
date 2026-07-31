"""What the self-hosted adapter needs brought to it.

The commercial providers earn their keep in extraction: you hand them a
conversation and they decide what is worth remembering. A self-hosted store has
no such step, so it has to be given one -- and if it is not, it says so
(``supports_ingest=False``) instead of silently storing raw transcript and
calling it memory.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from memgw.types import Episode


@runtime_checkable
class Embedder(Protocol):
    dimension: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class Extractor(Protocol):
    async def extract(self, episode: Episode) -> list[str]:
        """Decide what in this episode is worth remembering. Returning [] is a valid
        answer -- most turns are not worth keeping."""
        ...


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
