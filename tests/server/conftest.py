from __future__ import annotations

import httpx
import pytest

from memgw.catalog import Catalog
from memgw.core import MemoryCore
from memgw.server import create_app
from tests.fake import FakeAdapter, default_caps

KEY = "key-tenant-one"
OTHER_KEY = "key-tenant-two"


@pytest.fixture
async def gateway():
    """An app whose 'down' provider is genuinely down.

    That is what lets a test prove the gateway never called the adapter: if a
    request that should short-circuit had reached it, the answer would be a 503
    rather than the expected 409.
    """
    catalog = Catalog("sqlite+aiosqlite:///:memory:")
    await catalog.init()
    core = MemoryCore(
        catalog=catalog,
        providers={
            "fake": FakeAdapter(),
            "down": FakeAdapter(healthy=False),
            "flat": FakeAdapter(default_caps(scope_dims=["subject"])),
        },
        default_provider="fake",
    )
    app = create_app(core=core, api_keys={KEY: "t1", OTHER_KEY: "t2"})

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://memgw") as client:
        yield client, core, catalog
    await catalog.close()


def auth(key: str = KEY) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}
