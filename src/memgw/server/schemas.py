"""Request and response bodies.

``provider`` appears on every scoped request and is read as an assertion, never an
instruction: the gateway resolves the provider from the subject's binding and a
disagreement is a 409. Callers that track provider get a free integrity check;
callers that do not omit the field.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from memgw.capabilities import Capabilities
from memgw.types import MemoryRecord, Message, OnUnsupported, Scope, SearchMode


class _Scoped(BaseModel):
    scope: Scope
    provider: str | None = None
    tenant: str | None = None
    """Optional and only ever checked. The real tenant comes from the credential --
    including ``scope.tenant``, which the gateway overwrites rather than reads."""


class IngestIn(_Scoped):
    messages: list[Message] | None = None
    text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpsertIn(_Scoped):
    facts: list[str] = Field(min_length=1)


class ListIn(_Scoped):
    limit: int = Field(default=100, ge=1)


class SearchIn(_Scoped):
    query: str
    mode: SearchMode = "semantic"
    limit: int = Field(default=10, ge=1)
    min_score: float | None = None
    as_of: datetime | None = None
    on_unsupported: OnUnsupported = "reject"
    fail_open: bool = False


class DeleteScopeIn(_Scoped):
    """POST rather than DELETE: the scope travels in a body, and DELETE with a body
    is unevenly supported across proxies and clients."""


class UpdateIn(BaseModel):
    content: str
    tenant: str | None = None


class RebindIn(BaseModel):
    provider: str
    strategy: Literal["fresh_start", "migrate"] = "fresh_start"
    tenant: str | None = None


class SearchOut(BaseModel):
    results: list[MemoryRecord]
    provider: str
    degraded: bool
    requested: SearchMode | None
    served: SearchMode | None
    lost: list[str]
    provider_unavailable: bool


class WriteOut(BaseModel):
    results: list[MemoryRecord]
    provider: str
    """Named even when ``results`` is empty: an extraction that kept nothing still
    went to a provider, and a caller reconciling its own records needs to know which."""


#: The two write verbs answer with the same shape, so a caller can handle them the
#: same way. Kept as an alias rather than collapsed: the names appear in the OpenAPI
#: document, and ``IngestOut`` is what the published contract already called it.
IngestOut = WriteOut
UpsertOut = WriteOut


class ListOut(BaseModel):
    results: list[MemoryRecord]
    provider: str


class DeleteScopeOut(BaseModel):
    deleted: int
    provider: str


class BindingOut(BaseModel):
    subject: str
    provider: str | None


class RebindOut(BaseModel):
    subject: str
    provider: str
    orphaned_at: str | None
    orphaned_count: int
    note: str | None = None


class ProviderOut(BaseModel):
    name: str
    healthy: bool
    detail: str | None
    capabilities: Capabilities
