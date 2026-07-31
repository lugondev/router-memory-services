"""The gateway's error vocabulary.

Every failure the caller can act on is one of these. Each class carries its own
HTTP status so the server needs exactly one exception handler, and each can
narrow its ``code`` without changing that status -- ``no_provider_resolved`` is
an ``InvalidRequest``, not a category of its own.
"""

from __future__ import annotations

from typing import Any


class GatewayError(Exception):
    code = "internal_error"
    status = 500

    def __init__(
        self,
        message: str = "",
        *,
        details: dict[str, Any] | None = None,
        code: str | None = None,
        status: int | None = None,
    ) -> None:
        self.message = message or self.code
        self.details = details or {}
        if code is not None:
            self.code = code
        if status is not None:
            self.status = status
        super().__init__(self.message)

    def to_body(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


class InvalidRequest(GatewayError):
    code = "invalid_request"
    status = 400


class Unauthenticated(GatewayError):
    code = "unauthenticated"
    status = 401


class TenantMismatch(GatewayError):
    """A payload asserted a tenant other than the credential's."""

    code = "tenant_mismatch"
    status = 403


class MemoryNotFound(GatewayError):
    """Unknown id *for this tenant*. Another tenant's id is also a 404: a 403 here
    would confirm the id exists, which is an existence oracle across tenants."""

    code = "memory_not_found"
    status = 404


class ProviderMismatch(GatewayError):
    """The asserted provider disagrees with the subject's binding. Loud on purpose:
    ingesting to one backend and recalling from another otherwise returns an empty
    result with no error at all."""

    code = "provider_mismatch"
    status = 409


class UnsupportedCapability(GatewayError):
    code = "unsupported_capability"
    status = 422


class ProviderError(GatewayError):
    """The upstream provider failed. ``details['retryable']`` says whether to try again."""

    code = "provider_error"
    status = 424


class NotImplementedYet(GatewayError):
    code = "not_implemented"
    status = 501


class ProviderUnhealthy(GatewayError):
    code = "provider_unhealthy"
    status = 503
