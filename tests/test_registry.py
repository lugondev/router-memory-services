import importlib.util

import pytest

from memgw import adapters
from memgw.errors import InvalidRequest

#: Every adapter that ships, and the package each one needs. ``None`` means it has
#: no optional dependency and must always be on offer.
SHIPPED = [("pgvector", None), ("mem0", "mem0"), ("zep", "zep_cloud")]


def test_the_self_hosted_adapter_is_always_on_offer():
    assert "pgvector" in adapters.available()


@pytest.mark.parametrize(("provider", "package"), SHIPPED)
def test_an_adapter_is_offered_exactly_when_it_can_be_built(provider, package):
    """``available()`` reports what can actually be constructed, not what exists in
    the source tree -- otherwise a caller picks a provider that cannot start.

    Parametrised over every shipped adapter on purpose: the Zep adapter existed,
    was tested, and was reachable through ``MEMGW_PROVIDERS`` for a whole afternoon
    while ``available()`` never mentioned it, because the registry's list of modules
    to import had not been updated. A per-adapter test names the one that is missing.
    """
    buildable = package is None or importlib.util.find_spec(package) is not None
    assert (provider in adapters.available()) is buildable


def test_an_unknown_provider_names_the_ones_that_exist():
    with pytest.raises(InvalidRequest) as exc:
        adapters.get("supermemory")
    assert exc.value.code == "unknown_provider"
    assert "pgvector" in exc.value.details["available"]


def test_the_environment_and_the_registry_agree_on_what_ships():
    """``settings.KNOWN_PROVIDERS`` and the registry are two lists of the same thing,
    and two lists of the same thing drift."""
    from memgw.settings import KNOWN_PROVIDERS

    assert set(KNOWN_PROVIDERS) == {provider for provider, _ in SHIPPED}
