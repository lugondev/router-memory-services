"""Which provider serves this end-user.

The gateway resolves this, not the caller. A request that names the wrong
provider does not fail loudly on its own -- it recalls nothing and raises
nothing, because an empty result is indistinguishable from a user with no
memories yet. Per-end-user provider diversity turns that from a possibility into
a certainty, so the binding is authoritative and a caller's ``provider`` field is
only ever checked against it.
"""

from __future__ import annotations

from memgw.catalog import Catalog
from memgw.errors import InvalidRequest, ProviderMismatch


async def resolve_provider(
    catalog: Catalog,
    tenant: str,
    subject: str,
    *,
    default_provider: str | None,
    asserted: str | None = None,
) -> str:
    """Binding, else tenant default, else refuse.

    ``asserted`` is a claim to verify, never an instruction to obey: disagreeing
    with the resolved provider is a 409 and no provider call is made. A caller that
    tracks provider itself gets a free integrity check; one that does not omits the
    field and loses nothing.

    Resolving never binds. Binding happens on the first *write* only, so a stray
    search cannot pin an end-user to whatever the default happened to be that day.
    """
    bound = await catalog.get_binding(tenant, subject)
    resolved = bound or default_provider

    if resolved is None:
        raise InvalidRequest(
            f"no provider bound for subject {subject!r} and no tenant default configured",
            code="no_provider_resolved",
            details={"subject": subject},
        )

    if asserted is not None and asserted != resolved:
        raise ProviderMismatch(
            f"asserted provider {asserted!r} disagrees with the resolved provider {resolved!r}",
            details={"asserted": asserted, "bound": resolved},
        )

    return resolved
