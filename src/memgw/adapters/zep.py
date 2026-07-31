"""The Zep adapter.

Zep is the first provider whose shape genuinely differs from the contract, and
that difference is the interesting part of this file.

**What it has that nobody else did.** Zep builds a *temporal knowledge graph*:
facts are edges with ``valid_at`` / ``invalid_at``, so "what did we believe last
March" is a question it can actually answer. It is therefore the first adapter to
declare ``graph`` and ``temporal`` search, and the first thing that makes the
degradation matrix more than a design document.

**What it does not have, and will not pretend to.** Zep scopes its graph by
``user_id`` (or ``graph_id``). Threads exist -- you write messages into one -- but
graph search filters by user, not by thread. There is no agent dimension at all.

So this adapter declares ``scope_dims = ["tenant", "subject"]`` and nothing more.
The consequence is deliberate and worth stating plainly: **a search carrying a
session is a 422, not a search**. The alternative -- accepting the session and
quietly searching the whole user -- returns a confident answer drawn from the
wrong conversation, which is the failure mode this entire contract was built to
make impossible. A refusal you can read beats a result you cannot trust.

**Tenant.** Zep has none, so the tenant rides inside ``user_id`` as
``tenant:subject``, exactly as it does for Mem0.
"""

from __future__ import annotations

import asyncio
import importlib.util
import time
import uuid
from typing import Any

from memgw import adapters
from memgw.capabilities import Capabilities
from memgw.errors import ProviderError, UnsupportedCapability
from memgw.types import Episode, HealthStatus, ProviderMemory, Scope, SearchQuery

#: Separates tenant from subject inside ``user_id``. Zep indexes that field, so it
#: is the one place isolation can be carried.
TENANT_SEP = ":"

#: Modes this adapter serves. ``temporal`` is here because Zep edges carry validity
#: intervals, not because a flag exists somewhere.
MODES = ["semantic", "graph", "temporal"]


def zep_user_id(scope: Scope) -> str:
    if scope.tenant:
        return f"{scope.tenant}{TENANT_SEP}{scope.subject}"
    return scope.subject


class ZepAdapter:
    name = "zep"

    def __init__(self, *, api_key: str | None = None, client: Any = None) -> None:
        if client is not None:
            self._client = client
        else:
            from zep_cloud.client import AsyncZep  # imported here so the package stays optional

            self._client = AsyncZep(api_key=api_key)
        #: Zep needs a user to exist before data can hang off it. Creating one is
        #: idempotent in effect but not free, so each is created at most once here.
        self._known_users: set[str] = set()

    # -- introspection --------------------------------------------------------

    def capabilities(self) -> Capabilities:
        return Capabilities(
            supports_ingest=True,
            supports_upsert=True,
            # Zep has no "edit this fact" verb: you add a correcting episode and the
            # graph invalidates the old edge. That is a different operation with a
            # different meaning, so declaring update would be a lie.
            supports_update=False,
            supports_delete=True,
            supports_delete_by_scope=True,
            supports_list=True,
            search_modes=MODES,
            supports_score=True,
            max_limit=50,
            # No agent, no session. See the module docstring: a session-scoped search
            # would have to be silently widened to the whole user, and that answer is
            # worse than no answer.
            scope_dims=["tenant", "subject"],
            supports_labels=False,
            memory_model="temporal_graph",
            dedup="provider",
            supports_export=False,
            supports_import=True,
            # The graph is built after the write returns. Zep does not hide this, but
            # it does not put it in the response either.
            consistency="eventual",
            # Never passed the live conformance suite. Against the account it was
            # tried on, writes returned 200 with an episode id, `processed` stayed
            # false indefinitely, and every read came back empty with no error. See
            # `self_test` below, which is how that surfaces before a deploy.
            experimental=True,
            # Zep runs its own extraction and embedding inside add().
            metered_externally=True,
        )

    async def health(self) -> HealthStatus:
        try:
            await self._client.graph.search(query="__memgw_health__", user_id="__memgw__", limit=1)
            return HealthStatus(ok=True)
        except Exception as exc:  # noqa: BLE001
            # A missing user is a healthy Zep answering honestly, not an outage.
            if _is_not_found(exc):
                return HealthStatus(ok=True)
            return HealthStatus(ok=False, detail=str(exc))

    async def self_test(self, *, timeout: float = 60.0, interval: float = 2.0) -> HealthStatus:
        """Does it *work*, not just answer.

        ``health()`` asks whether Zep is reachable, and Zep was reachable the entire
        time it was broken: every write returned ``200`` with an episode id, and the
        graph was never built, so every read came back empty and nothing raised. That
        is the failure this gateway exists to make visible, arriving from the provider
        itself -- and no reachability check will ever catch it.

        So this writes one episode to a throwaway subject, waits for Zep to admit it
        processed it, and deletes the subject either way.
        """
        probe = Scope(subject=f"__memgw_probe_{uuid.uuid4().hex[:12]}__")
        user_id = zep_user_id(probe)
        deadline = time.monotonic() + timeout
        try:
            written = await self._add("memgw probe: the sky is green on Tuesdays.", probe)
            episode_id = written[0].native_id

            while time.monotonic() < deadline:
                if await self._processed(episode_id):
                    return HealthStatus(ok=True, detail="wrote an episode and Zep processed it")
                await asyncio.sleep(interval)

            return HealthStatus(
                ok=False,
                detail=(
                    f"episode {episode_id[:8]} was accepted but stayed processed=false for "
                    f"{timeout:.0f}s. Zep is reachable and is not building the graph, so "
                    "every read will return empty with no error at all. Check the project's "
                    "plan and ingestion quota."
                ),
            )
        except Exception as exc:  # noqa: BLE001 -- a probe reports, it never raises
            return HealthStatus(ok=False, detail=f"probe failed: {exc}")
        finally:
            try:
                await self._client.user.delete(user_id)
            except Exception:  # noqa: BLE001 -- best effort; the probe owns nothing else
                pass
            self._known_users.discard(user_id)

    async def _processed(self, episode_id: str) -> bool:
        try:
            episode = await self._client.graph.episode.get(episode_id)
        except Exception:  # noqa: BLE001 -- still settling, or briefly 503
            return False
        return bool(_attr(episode, "processed"))

    # -- write ----------------------------------------------------------------

    async def ingest(self, episode: Episode, scope: Scope) -> list[ProviderMemory]:
        """One episode in; Zep decides what becomes an edge.

        The whole conversation goes as a single episode rather than message by
        message, because Zep resolves references across an episode -- splitting it
        would cost exactly the context that makes the graph worth having.
        """
        return await self._add(episode.as_text(), scope)

    async def upsert(self, facts: list[str], scope: Scope) -> list[ProviderMemory]:
        out: list[ProviderMemory] = []
        for fact in facts:
            out.extend(await self._add(fact, scope))
        return out

    async def update(self, native_id: str, content: str) -> ProviderMemory:
        raise UnsupportedCapability(
            "Zep has no update verb: correct a fact by adding an episode that "
            "contradicts it, and the graph invalidates the old edge",
            details={"provider": "zep"},
        )

    async def _add(self, data: str, scope: Scope) -> list[ProviderMemory]:
        user_id = await self._ensure_user(scope)
        raw = await self._client.graph.add(user_id=user_id, type="text", data=data)
        native_id = _attr(raw, "uuid_") or _attr(raw, "uuid")
        if not native_id:
            raise ProviderError("zep accepted the episode but returned no id")
        return [ProviderMemory(native_id=str(native_id), content=data, raw=_as_dict(raw))]

    async def _ensure_user(self, scope: Scope) -> str:
        user_id = zep_user_id(scope)
        if user_id in self._known_users:
            return user_id
        try:
            await self._client.user.add(user_id=user_id)
        except Exception as exc:  # noqa: BLE001
            # Already existing is the common case and not an error.
            if not _is_conflict(exc):
                raise
        self._known_users.add(user_id)
        return user_id

    # -- read -----------------------------------------------------------------

    async def search(self, query: SearchQuery, scope: Scope) -> list[ProviderMemory]:
        if query.mode not in MODES:
            raise UnsupportedCapability(
                f"zep adapter cannot serve {query.mode!r} search",
                details={"available": MODES},
            )
        # 'nodes' returns entities, 'edges' returns facts. A memory is a fact, so a
        # graph query still reads edges -- the difference between graph and semantic
        # here is how Zep ranks them, not what a memory is.
        raw = await self._client.graph.search(
            query=query.query,
            user_id=zep_user_id(scope),
            scope="edges",
            limit=min(query.limit, 50),
        )
        return self._to_memories(raw, query.min_score)

    async def list_scope(self, scope: Scope, limit: int) -> list[ProviderMemory]:
        """A real listing, not a search with a wildcard query.

        Zep has ``graph.edge.get_by_user_id``. Passing ``query="*"`` to a semantic
        search would rank facts by their similarity to an asterisk, which is not an
        ordering so much as a coin toss.
        """
        edges = await self._client.graph.edge.get_by_user_id(
            zep_user_id(scope), limit=min(limit, 50)
        )
        return [self._to_memory(edge) for edge in edges or []]

    async def get(self, native_id: str) -> ProviderMemory | None:
        raw = await self._client.graph.edge.get(native_id)
        return None if raw is None else self._to_memory(raw)

    # -- delete ---------------------------------------------------------------

    async def delete(self, native_id: str) -> bool:
        await self._client.graph.edge.delete(native_id)
        return True

    async def delete_scope(self, scope: Scope) -> int:
        """Deleting the user cascades through the graph.

        Zep offers no partial-scope delete, and this adapter declares no dimension
        finer than the subject -- so "delete this scope" and "delete this user" are
        the same operation, which is the only reason this is honest.

        The count is taken *before* the delete, because ``user.delete`` answers with a
        ``SuccessResponse`` and no number. Reading a count out of that yields zero,
        and an erasure that reports "0 deleted" is indistinguishable from one that
        did nothing.
        """
        user_id = zep_user_id(scope)
        try:
            doomed = await self._client.graph.edge.get_by_user_id(user_id, limit=1000)
            removed = len(doomed or [])
        except Exception:  # noqa: BLE001 -- the delete matters, the tally does not
            removed = 0
        await self._client.user.delete(user_id)
        self._known_users.discard(user_id)
        return removed

    # -- internals ------------------------------------------------------------

    def _to_memories(self, raw: Any, min_score: float | None) -> list[ProviderMemory]:
        edges = _attr(raw, "edges") or []
        memories = [self._to_memory(edge) for edge in edges]
        if min_score is None:
            return memories
        return [m for m in memories if (m.score or 0.0) >= min_score]

    @staticmethod
    def _to_memory(edge: Any) -> ProviderMemory:
        return ProviderMemory(
            native_id=str(_attr(edge, "uuid_") or _attr(edge, "uuid") or ""),
            content=_attr(edge, "fact") or _attr(edge, "content") or "",
            created_at=_attr(edge, "created_at"),
            # Zep's whole point: a fact is true over an interval, not forever.
            valid_from=_attr(edge, "valid_at"),
            valid_to=_attr(edge, "invalid_at"),
            score=_attr(edge, "score"),
            raw=_as_dict(edge),
        )


def _attr(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _as_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    dump = getattr(obj, "model_dump", None)
    if dump is not None:
        try:
            return dump(mode="json")
        except Exception:  # noqa: BLE001 -- raw is a convenience, never a failure
            pass
    return {}


def _is_not_found(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 404 or "not found" in str(exc).lower()


def _is_conflict(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) in (400, 409) or "exist" in str(exc).lower()


#: Registered only when the optional dependency is present, so ``available()``
#: reports what can actually be built.
if importlib.util.find_spec("zep_cloud") is not None:
    adapters.register("zep", ZepAdapter)
