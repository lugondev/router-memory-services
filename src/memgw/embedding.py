"""What the self-hosted adapter needs brought to it.

The commercial providers earn their keep in extraction: you hand them a
conversation and they decide what is worth remembering. A self-hosted store has
no such step, so it has to be given one -- and if it is not, it says so
(``supports_ingest=False``) instead of silently storing raw transcript and
calling it memory.

:class:`Embedder` and :class:`Extractor` stay protocols, because the whole point
is that you can bring your own. But shipping *only* protocols made the honest
baseline adapter unusable without writing a class first, which is a fair ask of a
library and an unfair one of a product. So an OpenAI implementation of each ships
here: one API key, and the self-hosted path works.
"""

from __future__ import annotations

import json
import math
from typing import Any, Protocol, runtime_checkable

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


#: Dimensions of the models we know. An unknown model is not guessed at: a wrong
#: dimension is not an error anywhere, it is silently terrible recall.
OPENAI_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EXTRACTION_MODEL = "gpt-4o-mini"

_EXTRACTION_PROMPT = """\
You maintain a long-term memory about a person. From the conversation below, \
extract only durable facts worth remembering later: preferences, traits, \
relationships, commitments, stable circumstances.

Rules:
- Write each fact as a short standalone sentence about the person.
- Do NOT extract questions, small talk, pleasantries, or one-off task requests.
- Most conversations contain nothing worth keeping. Returning an empty list is \
the correct answer far more often than not.

Reply with JSON only: {"facts": ["...", "..."]}

Conversation:
"""


class OpenAIEmbedder:
    """Embeddings from OpenAI, one call per batch.

    ``client`` is injectable so the calling code can share a configured client,
    hand in an Azure or compatible endpoint, or -- in tests -- hand in something
    that never touches the network.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimension: int | None = None,
        client: Any = None,
    ) -> None:
        if dimension is None:
            dimension = OPENAI_DIMENSIONS.get(model)
        if dimension is None:
            raise ValueError(
                f"unknown embedding model {model!r}: pass dimension= explicitly. "
                "Guessing it would not fail, it would just retrieve badly."
            )
        self.model = model
        self.dimension = dimension
        self._client = client or _openai_client(api_key)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


class OpenAIExtractor:
    """Decides what in an episode is worth remembering.

    This is the step the commercial providers charge for, and the reason a bare
    store reports ``supports_ingest=False`` without one.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_EXTRACTION_MODEL,
        client: Any = None,
        prompt: str = _EXTRACTION_PROMPT,
    ) -> None:
        self.model = model
        self._prompt = prompt
        self._client = client or _openai_client(api_key)

    async def extract(self, episode: Episode) -> list[str]:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": self._prompt + episode.as_text()}],
            response_format={"type": "json_object"},
        )
        return _facts(response.choices[0].message.content)


def _facts(reply: str | None) -> list[str]:
    """Read the model's answer, and keep nothing when it is not an answer.

    A model that replies with prose instead of JSON is having a bad day, not
    signalling an outage. Raising here would turn one stray reply into a failed
    ingest and a 424 the caller can do nothing about; keeping nothing loses at most
    one episode's facts and stays true to "most turns are not worth keeping".
    """
    if not reply:
        return []
    try:
        payload = json.loads(reply)
    except (ValueError, TypeError):
        return []
    facts = payload.get("facts") if isinstance(payload, dict) else None
    if not isinstance(facts, list):
        return []
    return [fact.strip() for fact in facts if isinstance(fact, str) and fact.strip()]


def _openai_client(api_key: str | None) -> Any:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover -- depends on the install extra
        raise ImportError(
            "the built-in embedder needs the openai package: pip install 'memgw[openai]'"
        ) from exc
    return AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
