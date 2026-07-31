"""Who is calling.

One rule, and everything else follows from it: **``tenant_id`` comes from the
credential and never from the payload**. Anything in a request body may narrow the
scope; nothing in it may widen the scope. Without that, a caller who composes its
own identifiers can read another tenant's memory by crafting one.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

from memgw.errors import TenantMismatch, Unauthenticated


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Principal:
    tenant_id: str

    key_id: str
    """A digest prefix, never the key. Principals end up in logs and error contexts,
    and a "harmless" first-eight-characters convention puts a third of a short key
    into every access log line."""


class ApiKeyAuth:
    """MVP: one key per tenant, backend to backend.

    Reserved, not built: when the library runs inside an end-user's app a tenant key
    cannot live there, and short-lived subject-scoped tokens minted by the
    integrator's backend take over. The contract already says a subject in the token
    wins over a subject in the body.
    """

    def __init__(self, keys: dict[str, str]) -> None:
        #: Keyed by digest, so the raw keys are not held in memory and a lookup cannot
        #: leak how far a guess got. Dict lookup on a raw token compares byte by byte
        #: and stops at the first difference, which is a measurable oracle.
        self._by_digest = {_digest(key): tenant for key, tenant in keys.items()}

    def authenticate(self, header: str | None) -> Principal:
        if not header:
            raise Unauthenticated("missing Authorization header")

        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise Unauthenticated("expected 'Authorization: Bearer <key>'")

        digest = _digest(token)
        tenant = None
        for known, owner in self._by_digest.items():
            # Every candidate is compared, and each comparison is constant time, so
            # neither the answer nor the time taken depends on how close a guess was.
            if hmac.compare_digest(known, digest):
                tenant = owner
        if tenant is None:
            raise Unauthenticated("unknown api key")
        return Principal(tenant_id=tenant, key_id=digest[:12])


def assert_no_wider_tenant(principal: Principal, asserted: str | None) -> None:
    if asserted is not None and asserted != principal.tenant_id:
        raise TenantMismatch(
            "payload asserts a tenant other than the credential's",
            details={"asserted": asserted},
        )
