"""The wire types.

The scope stays three separate dimensions all the way to the adapter. Collapsing
them into one composite key is tempting -- callers usually already hold one --
but it costs cross-session recall (``search(subject=u_1)`` with no session is the
single most important memory query, and an adapter can only map a blob onto one
native dimension), tenant isolation, and erasure by subject.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Role = Literal["user", "assistant", "system"]
SearchMode = Literal["semantic", "keyword", "hybrid", "graph", "temporal"]
OnUnsupported = Literal["reject", "degrade"]


class Scope(BaseModel):
    subject: str
    agent: str | None = None
    session: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)

    tenant: str | None = None
    """Stamped by the gateway from the credential on the way down to the adapter, and
    **never** read from a caller's payload -- :meth:`under` overwrites whatever was
    there. It is a scope dimension rather than gateway bookkeeping because subject ids
    are chosen by tenants: two tenants naming an end-user ``u_1`` share one row in any
    provider that is only filtered by subject, and neither of them ever finds out.
    """

    @field_validator("subject")
    @classmethod
    def _subject_must_be_real(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("subject is required and must be non-empty")
        return v

    @field_validator("agent", "session")
    @classmethod
    def _blank_is_none(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return v.strip() or None

    def under(self, tenant: str) -> Scope:
        """The same scope, stamped with the credential's tenant.

        Overwrites rather than merges: a tenant that arrived in the request body is a
        claim, and a claim may never widen what the credential already decided.
        """
        return self.model_copy(update={"tenant": tenant})

    def without_tenant(self) -> Scope:
        """What the caller gets back. They asked about ``u_1``, not about
        ``tenant-a/u_1``; echoing the stamp back is noise in every response."""
        return self.model_copy(update={"tenant": None})

    def dims(self) -> dict[str, str]:
        """The dimensions actually set -- exactly what a provider filter must cover.

        A read that forwards fewer dimensions than the write used comes back empty
        with no error on at least one provider, so filters are built from this and
        never from a subset.
        """
        out = {}
        if self.tenant:
            out["tenant"] = self.tenant
        out["subject"] = self.subject
        if self.agent:
            out["agent"] = self.agent
        if self.session:
            out["session"] = self.session
        return out


class Message(BaseModel):
    role: Role
    content: str
    at: datetime | None = None


class Episode(BaseModel):
    """Raw material for ``ingest``. Exactly one of ``messages`` / ``text``."""

    messages: list[Message] | None = None
    text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _exactly_one_source(self) -> Episode:
        if (self.messages is None) == (self.text is None):
            raise ValueError("provide exactly one of messages / text")
        if self.messages is not None and not self.messages:
            raise ValueError("messages must not be empty")
        if self.text is not None and not self.text.strip():
            raise ValueError("text must not be blank")
        return self

    def as_text(self) -> str:
        if self.text is not None:
            return self.text
        return "\n".join(f"{m.role}: {m.content}" for m in self.messages or [])


class SearchQuery(BaseModel):
    query: str
    mode: SearchMode = "semantic"
    limit: int = Field(default=10, ge=1)
    min_score: float | None = None
    as_of: datetime | None = None
    on_unsupported: OnUnsupported = "reject"
    fail_open: bool = False


class ProviderMemory(BaseModel):
    """What an adapter returns. Speaks ``native_id`` only -- adapters never see a
    gateway id, the catalog bridges the two."""

    native_id: str
    content: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    score: float | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class MemoryRecord(BaseModel):
    """What the caller receives.

    ``provider_raw`` is deliberate: without an escape hatch every provider-specific
    feature is swallowed by the abstraction and power users have to abandon the
    gateway. With it, the gateway need not model every feature to stay useful.
    """

    id: str
    provider: str
    native_id: str
    content: str
    scope: Scope
    score: float | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    provider_raw: dict[str, Any] = Field(default_factory=dict)


class HealthStatus(BaseModel):
    ok: bool
    detail: str | None = None
