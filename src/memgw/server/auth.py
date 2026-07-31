"""Who is calling.

One rule, and everything else follows from it: **``tenant_id`` comes from the
credential and never from the payload**. Anything in a request body may narrow the
scope; nothing in it may widen the scope. Without that, a caller who composes its
own identifiers can read another tenant's memory by crafting one.
"""

from __future__ import annotations

from dataclasses import dataclass

from memgw.errors import TenantMismatch, Unauthenticated


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    key_id: str


class ApiKeyAuth:
    """MVP: one key per tenant, backend to backend.

    Reserved, not built: when the library runs inside an end-user's app a tenant key
    cannot live there, and short-lived subject-scoped tokens minted by the
    integrator's backend take over. The contract already says a subject in the token
    wins over a subject in the body.
    """

    def __init__(self, keys: dict[str, str]) -> None:
        self._keys = dict(keys)

    def authenticate(self, header: str | None) -> Principal:
        if not header:
            raise Unauthenticated("missing Authorization header")

        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise Unauthenticated("expected 'Authorization: Bearer <key>'")

        tenant = self._keys.get(token)
        if tenant is None:
            raise Unauthenticated("unknown api key")
        return Principal(tenant_id=tenant, key_id=token[:8])


def assert_no_wider_tenant(principal: Principal, asserted: str | None) -> None:
    if asserted is not None and asserted != principal.tenant_id:
        raise TenantMismatch(
            "payload asserts a tenant other than the credential's",
            details={"asserted": asserted},
        )
