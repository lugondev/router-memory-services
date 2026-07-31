import importlib.util

import pytest

from memgw import adapters
from memgw.errors import InvalidRequest


def test_the_self_hosted_adapter_is_always_on_offer():
    assert "pgvector" in adapters.available()


def test_an_optional_adapter_is_offered_only_when_it_can_be_built():
    # available() reports what can actually be constructed, not what exists in the
    # source tree -- otherwise a caller picks a provider that cannot start.
    installed = importlib.util.find_spec("mem0") is not None
    assert ("mem0" in adapters.available()) is installed


def test_an_unknown_provider_names_the_ones_that_exist():
    with pytest.raises(InvalidRequest) as exc:
        adapters.get("zep")
    assert exc.value.code == "unknown_provider"
    assert "pgvector" in exc.value.details["available"]
