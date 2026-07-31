import pytest

pytest_plugins: list[str] = []


@pytest.fixture
def anyio_backend():
    return "asyncio"
