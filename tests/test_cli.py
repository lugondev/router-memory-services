"""The command line, and the check that makes a misconfiguration readable.

``doctor`` exists because every failure this project hit against a real provider
was a configuration problem wearing a stack trace: a 400 about ``temperature``, a
SQLite threading error from a vector store, a lock held by a telemetry client
nobody knew was running. Each one cost an afternoon and each one is a single
yes/no question. Asking them up front is the difference between a product and a
library with good docs.
"""

from __future__ import annotations

import pytest


class TestDoctorReadsTheConfiguration:
    async def test_a_missing_key_is_reported_as_a_failed_check_not_a_traceback(self):
        from memgw.cli import doctor

        report = await doctor({"MEMGW_PROVIDERS": "pgvector"})
        assert report.ok is False
        [failure] = [c for c in report.checks if not c.ok]
        assert "OPENAI_API_KEY" in failure.detail

    async def test_a_workable_configuration_passes_its_settings_check(self):
        from memgw.cli import doctor

        report = await doctor(
            {
                "OPENAI_API_KEY": "sk-test",
                "MEMGW_API_KEYS": "k1:tenant-a",
                "MEMGW_CATALOG_URL": "sqlite+aiosqlite:///:memory:",
                "MEMGW_PGVECTOR_URL": "sqlite+aiosqlite:///:memory:",
            }
        )
        names = {c.name: c for c in report.checks}
        assert names["settings"].ok is True
        assert names["catalog"].ok is True

    async def test_it_warns_when_the_gateway_has_no_api_keys_at_all(self):
        # A gateway with no keys authenticates nobody. It starts, it answers 401 to
        # every request, and it looks healthy the whole time.
        from memgw.cli import doctor

        report = await doctor(
            {
                "OPENAI_API_KEY": "sk-test",
                "MEMGW_CATALOG_URL": "sqlite+aiosqlite:///:memory:",
                "MEMGW_PGVECTOR_URL": "sqlite+aiosqlite:///:memory:",
            }
        )
        keys = next(c for c in report.checks if c.name == "api_keys")
        assert keys.ok is False
        assert "MEMGW_API_KEYS" in keys.detail

    async def test_mem0_without_a_reachable_qdrant_is_named_before_it_hangs(self):
        # Mem0's own default is a local Qdrant that deadlocks under async. If the
        # configured server is unreachable, say so here rather than at first write.
        from memgw.cli import doctor

        report = await doctor(
            {
                "OPENAI_API_KEY": "sk-test",
                "MEMGW_PROVIDERS": "mem0",
                "MEMGW_API_KEYS": "k1",
                "MEMGW_CATALOG_URL": "sqlite+aiosqlite:///:memory:",
                "MEMGW_QDRANT_PORT": "1",  # nothing listens here
            }
        )
        qdrant = next(c for c in report.checks if c.name == "qdrant")
        assert qdrant.ok is False
        assert "6333" in qdrant.detail or "1" in qdrant.detail

    async def test_a_report_renders_as_lines_a_human_can_act_on(self):
        from memgw.cli import doctor, render

        report = await doctor({"MEMGW_PROVIDERS": "pgvector"})
        text = render(report)
        assert "settings" in text
        assert "OPENAI_API_KEY" in text


class TestTheAuditLogIsActuallyOn:
    """The library attaches a NullHandler and sets no level, which is correct for a
    library and useless for a server: ``memgw serve`` produced not one audit line.
    Turning it on is the application's job, and ``serve`` is the application."""

    def test_configuring_logging_gives_the_memgw_logger_a_level_and_a_handler(self):
        import logging

        from memgw.cli import configure_logging

        configure_logging("info")
        log = logging.getLogger("memgw")
        assert log.level == logging.INFO
        assert any(not isinstance(h, logging.NullHandler) for h in log.handlers)

    def test_configuring_twice_does_not_double_every_line(self):
        import logging

        from memgw.cli import configure_logging

        configure_logging("info")
        before = len(logging.getLogger("memgw").handlers)
        configure_logging("info")
        assert len(logging.getLogger("memgw").handlers) == before

    async def test_a_verb_actually_emits_through_the_configured_handler(self):
        import io
        import logging

        from memgw.catalog import Catalog
        from memgw.cli import configure_logging
        from memgw.core import MemoryCore
        from memgw.types import Episode, Scope
        from tests.fake import FakeAdapter

        configure_logging("info")
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        log = logging.getLogger("memgw")
        log.addHandler(handler)
        try:
            catalog = Catalog("sqlite+aiosqlite:///:memory:")
            await catalog.init()
            core = MemoryCore(
                catalog=catalog, providers={"fake": FakeAdapter()}, default_provider="fake"
            )
            await core.ingest("t", Episode(text="black coffee"), Scope(subject="u_1"))
            await catalog.close()
        finally:
            log.removeHandler(handler)
        assert "ingest" in stream.getvalue()


class TestServeIsWiredUp:
    def test_the_console_script_exposes_serve_and_doctor(self):
        from memgw.cli import main

        with pytest.raises(SystemExit) as exit_info:
            main(["--help"])
        assert exit_info.value.code == 0

    def test_an_unknown_command_exits_nonzero(self):
        from memgw.cli import main

        with pytest.raises(SystemExit) as exit_info:
            main(["frobnicate"])
        assert exit_info.value.code != 0

    def test_the_factory_returns_an_app_rather_than_a_coroutine(self, monkeypatch):
        """``uvicorn --factory`` calls the factory *synchronously*.

        An ``async def`` factory therefore hands uvicorn a coroutine object, which it
        takes for an ASGI2 app and calls -- and every request becomes
        ``TypeError: 'coroutine' object is not callable``, a 500 on ``/healthz``
        included. A test that awaited the factory proved the coroutine builds an app;
        it did not prove uvicorn could use it. This one does.
        """
        import inspect

        from memgw.cli import asgi_app

        _env(monkeypatch)
        app = asgi_app()
        assert not inspect.iscoroutine(app)
        assert "/v1/memories:search" in app.openapi()["paths"]

    async def test_the_app_answers_healthz_once_its_lifespan_has_started(self, monkeypatch):
        # The end-to-end shape: build it the way uvicorn does, start it the way
        # uvicorn does, then make a real request through the ASGI stack.
        import httpx

        from memgw.cli import asgi_app

        _env(monkeypatch)
        app = asgi_app()
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://memgw") as client:
                assert (await client.get("/healthz")).status_code == 200
                # And the core really was constructed during startup, not at import.
                unauthorised = await client.post(
                    "/v1/memories:search",
                    json={"scope": {"subject": "u_1"}, "query": "coffee"},
                )
                assert unauthorised.status_code == 401


def _env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MEMGW_API_KEYS", "k1:tenant-a")
    monkeypatch.setenv("MEMGW_CATALOG_URL", "sqlite+aiosqlite:///:memory:")
    monkeypatch.setenv("MEMGW_PGVECTOR_URL", "sqlite+aiosqlite:///:memory:")
