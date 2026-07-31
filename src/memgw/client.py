"""One client, two modes.

    Memory(provider="pgvector", config={...})     # embedded
    Memory(base_url="https://...", api_key=...)   # proxy

Embedded is a single fixed provider with no routing and nothing to deploy -- the
answer to "one library, swap the backend by configuration". Proxy adds bindings,
tenancy and observability -- the answer to "give each end-user a different memory
backend". Routing is deliberately *not* retrofitted into embedded mode: it needs
state that outlives a process, and state that outlives a process is a server.
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from memgw import adapters
from memgw.capabilities import Capabilities
from memgw.catalog import Catalog
from memgw.core import MemoryCore, SearchResult
from memgw.errors import GatewayError
from memgw.types import Episode, MemoryRecord, Message, Scope, SearchQuery

EMBEDDED_TENANT = "local"


class ScopeHandle:
    """A subject (and optionally an agent) held for reuse.

    Purely client-side sugar. The wire still carries three separate dimensions,
    which is what makes ``search`` with no session mean "everything about this
    person" instead of "this conversation only".
    """

    def __init__(self, client: Memory, subject: str, agent: str | None = None) -> None:
        self._client = client
        self.subject = subject
        self.agent = agent

    def _scope(self, session: str | None = None, labels: dict[str, str] | None = None) -> Scope:
        return Scope(subject=self.subject, agent=self.agent, session=session, labels=labels or {})

    async def ingest(
        self,
        messages: list[Message] | list[dict[str, Any]] | None = None,
        *,
        text: str | None = None,
        session: str | None = None,
        labels: dict[str, str] | None = None,
    ) -> list[MemoryRecord]:
        return await self._client.ingest(self._scope(session, labels), messages=messages, text=text)

    async def search(
        self,
        query: str,
        *,
        session: str | None = None,
        labels: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> SearchResult:
        return await self._client.search(self._scope(session, labels), query, **kwargs)

    async def delete_all(self, *, session: str | None = None) -> int:
        return await self._client.delete_scope(self._scope(session))


class Memory:
    def __init__(
        self,
        *,
        provider: str | None = None,
        config: dict[str, Any] | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        catalog_url: str = "sqlite+aiosqlite:///memgw.db",
        journal: bool = False,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        if provider and (base_url or http is not None):
            raise ValueError(
                "provider= is embedded mode and base_url= is proxy mode; pick one. "
                "Per-end-user provider routing lives in the gateway, so it needs proxy mode."
            )
        if not provider and not (base_url or http is not None):
            raise ValueError("give provider= for embedded mode or base_url= for proxy mode")

        self._provider = provider
        self._config = config or {}
        self._catalog_url = catalog_url
        self._journal = journal
        self._core: MemoryCore | None = None
        self._adapter: Any = None

        self._http = http
        self._base_url = base_url
        self._api_key = api_key

    @property
    def mode(self) -> str:
        return "embedded" if self._provider else "proxy"

    def scope(self, subject: str, agent: str | None = None) -> ScopeHandle:
        return ScopeHandle(self, subject, agent)

    @staticmethod
    def parse_scope(key: str, fmt: str) -> Scope:
        """Split a composite key a caller already holds, e.g.
        ``parse_scope("t/u_1/s_9", "{tenant}/{subject}/{session}")``.

        Splitting happens here, in the client. What travels on the wire is always the
        structured triple -- an opaque composite cannot express cross-session recall,
        cannot be checked against a credential, and cannot be erased by subject.
        """
        pattern = re.escape(fmt)
        for field in ("tenant", "subject", "agent", "session"):
            pattern = pattern.replace(re.escape("{" + field + "}"), f"(?P<{field}>[^/]+)")
        match = re.fullmatch(pattern, key)
        if match is None:
            raise ValueError(f"{key!r} does not match {fmt!r}")
        found = match.groupdict()
        if not found.get("subject"):
            raise ValueError(f"{fmt!r} has no {{subject}}, which is the one required dimension")
        return Scope(
            subject=found["subject"], agent=found.get("agent"), session=found.get("session")
        )

    # -- verbs ----------------------------------------------------------------

    async def ingest(
        self,
        scope: Scope,
        *,
        messages: list[Message] | list[dict[str, Any]] | None = None,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[MemoryRecord]:
        normalised = [
            m if isinstance(m, Message) else Message(**m) for m in (messages or [])
        ] or None

        if self._provider:
            core = await self._started()
            episode = Episode(messages=normalised, text=text, metadata=metadata or {})
            return await core.ingest(EMBEDDED_TENANT, episode, scope)

        body = {
            "scope": scope.model_dump(),
            "messages": [m.model_dump(mode="json") for m in normalised] if normalised else None,
            "text": text,
            "metadata": metadata or {},
        }
        payload = await self._post("/v1/memories:ingest", body)
        return [MemoryRecord(**record) for record in payload["results"]]

    async def search(self, scope: Scope, query: str, **kwargs: Any) -> SearchResult:
        if self._provider:
            core = await self._started()
            return await core.search(EMBEDDED_TENANT, SearchQuery(query=query, **kwargs), scope)

        payload = await self._post(
            "/v1/memories:search", {"scope": scope.model_dump(), "query": query, **kwargs}
        )
        return SearchResult(**payload)

    async def get(self, gateway_id: str) -> MemoryRecord:
        if self._provider:
            core = await self._started()
            return await core.get(EMBEDDED_TENANT, gateway_id)
        return MemoryRecord(**await self._request("GET", f"/v1/memories/{gateway_id}"))

    async def update(self, gateway_id: str, content: str) -> MemoryRecord:
        if self._provider:
            core = await self._started()
            return await core.update(EMBEDDED_TENANT, gateway_id, content)
        payload = await self._request(
            "PATCH", f"/v1/memories/{gateway_id}", json={"content": content}
        )
        return MemoryRecord(**payload)

    async def delete(self, gateway_id: str) -> None:
        if self._provider:
            core = await self._started()
            await core.delete(EMBEDDED_TENANT, gateway_id)
            return
        await self._request("DELETE", f"/v1/memories/{gateway_id}")

    async def delete_scope(self, scope: Scope) -> int:
        if self._provider:
            core = await self._started()
            return await core.delete_scope(EMBEDDED_TENANT, scope)
        payload = await self._post("/v1/memories:delete", {"scope": scope.model_dump()})
        return payload["deleted"]

    async def capabilities(self) -> Capabilities:
        """Async in both modes. In proxy mode this is a round trip, and pretending
        otherwise would mean two different call signatures for the same question."""
        if self._provider:
            core = await self._started()
            return core.capabilities()
        return Capabilities(**await self._request("GET", "/v1/capabilities"))

    async def close(self) -> None:
        if self._core is not None:
            await self._core.catalog.close()
            self._core = None
        if self._adapter is not None and hasattr(self._adapter, "close"):
            await self._adapter.close()
            self._adapter = None
        if self._http is not None and self._base_url:
            await self._http.aclose()

    # -- internals ------------------------------------------------------------

    async def _started(self) -> MemoryCore:
        if self._core is not None:
            return self._core

        assert self._provider is not None
        self._adapter = adapters.build(self._provider, **self._config)
        if hasattr(self._adapter, "init"):
            await self._adapter.init()

        catalog = Catalog(self._catalog_url)
        await catalog.init()
        self._core = MemoryCore(
            catalog=catalog,
            providers={self._provider: self._adapter},
            default_provider=self._provider,
            journal_enabled=self._journal,
        )
        return self._core

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
            self._http = httpx.AsyncClient(base_url=self._base_url or "", headers=headers)
        return self._http

    async def _post(self, path: str, body: dict[str, Any]) -> Any:
        return await self._request("POST", path, json=body)

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None
        response = await self._client().request(method, path, headers=headers, **kwargs)
        if response.status_code >= 400:
            raise _to_error(response)
        if response.status_code == 204 or not response.content:
            return None
        return response.json()


def _to_error(response: httpx.Response) -> GatewayError:
    """Rebuild the gateway's error on the client side.

    A proxy-mode caller gets the same exception type it would get embedded, so code
    written against one mode keeps working in the other.
    """
    try:
        payload = response.json()["error"]
    except Exception:  # noqa: BLE001
        return GatewayError(response.text or "gateway request failed", status=response.status_code)
    return GatewayError(
        payload.get("message", ""),
        code=payload.get("code"),
        status=response.status_code,
        details=payload.get("details") or {},
    )
