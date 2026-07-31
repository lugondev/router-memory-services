"""``memgw serve`` and ``memgw doctor``.

``doctor`` is the more important of the two. Every failure this project hit
against a real provider was a configuration problem wearing a stack trace: a 400
about ``temperature`` from a model that does not take one, a SQLite threading
error thrown by a vector store three layers down, a lock held by a telemetry
client nobody knew was running. Each cost an afternoon; each is a yes/no question
that can be asked in a second. Asking them before the first request is most of
the difference between a product and a library with good documentation.

Every failure names the thing to change. Every check reads and nothing more --
except ``--probe``, which writes one record to a throwaway subject and deletes it,
and is opt-in for exactly that reason.

``--probe`` exists because one failure was not a configuration problem at all. Zep
accepted every write with a ``200`` and an episode id, never built the graph, and
answered every read with an empty list and no error. It was *reachable* the whole
time. No amount of checking configuration or pinging an endpoint sees that; the
only question that does is "write something and read it back".
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import socket
import sys
from dataclasses import dataclass, field
from typing import Any

from memgw.types import HealthStatus


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.checks.append(Check(name=name, ok=ok, detail=detail))


async def preflight(env: dict[str, str] | None = None) -> Report:
    """What ``serve`` checks before booting: configuration only.

    Deliberately not the provider probe. A provider outage must not stop the gateway
    from starting -- the other providers still work, ``fail_open`` exists for exactly
    this, and a gateway that refuses to boot because a third party is down has turned
    their outage into yours.
    """
    return await doctor(env, probe=False)


async def doctor(
    env: dict[str, str] | None = None,
    *,
    probe: bool = False,
    adapters: dict[str, Any] | None = None,
) -> Report:
    """Answer, in order, every question that has actually gone wrong.

    ``probe`` additionally asks each provider to prove it *works* rather than merely
    answers -- see :meth:`memgw.adapters.base.MemoryAdapter.self_test`. It writes to
    a throwaway subject and deletes it, so it is off unless asked for: a boot that
    writes to a customer's provider every time is not a health check, it is a side
    effect.
    """
    from memgw.settings import Settings

    report = Report()
    e = dict(os.environ if env is None else env)

    try:
        settings = Settings.from_env(e)
    except ValueError as exc:
        report.add("settings", False, str(exc))
        return report
    report.add(
        "settings",
        True,
        f"providers={settings.providers} default={settings.default_provider}",
    )

    # A gateway with no keys starts cleanly and answers 401 to everything, forever.
    report.add(
        "api_keys",
        bool(settings.api_keys),
        f"{len(settings.api_keys)} key(s) configured"
        if settings.api_keys
        else "no MEMGW_API_KEYS: every request will be a 401",
    )

    await _check_catalog(report, settings)

    if "mem0" in settings.providers:
        _check_qdrant(report, settings)
        report.add(
            "mem0_telemetry",
            _flagged_off(e.get("MEM0_TELEMETRY")),
            "MEM0_TELEMETRY is not False: Mem0 opens a second, undeclared local "
            "Qdrant under ~/.mem0 which locks its own directory"
            if not _flagged_off(e.get("MEM0_TELEMETRY"))
            else "disabled",
        )

    if "zep" in settings.providers:
        report.add(
            "zep_key",
            bool(settings.zep_api_key),
            "ZEP_API_KEY present" if settings.zep_api_key else "ZEP_API_KEY is missing",
        )

    _warn_about_experimental(report, settings)

    if probe:
        await _probe_providers(report, settings, adapters)

    return report


def _warn_about_experimental(report: Report, settings) -> None:
    """Name the providers that have only been checked against an SDK.

    Not a failure -- you may well want one. But every provider is chosen from one
    list by one name, which quietly implies they are peers, and they are not.
    """
    from memgw.settings import EXPERIMENTAL

    risky = [p for p in settings.providers if p in EXPERIMENTAL]
    if not risky:
        return
    report.add(
        "experimental",
        True,
        f"{risky} have never passed the live conformance suite; "
        "run `memgw doctor --probe` before trusting one with real memories",
    )


async def _probe_providers(report: Report, settings, adapters: dict[str, Any] | None) -> None:
    """Ask each provider to prove it works, where it knows how."""
    from memgw.settings import build_adapter

    for name in settings.providers:
        try:
            if adapters is not None:
                adapter = adapters[name]
            else:
                adapter = await build_adapter(name, settings)
        except Exception as exc:  # noqa: BLE001 -- doctor reports, never raises
            report.add(f"{name}_pipeline", False, f"could not be built: {exc}")
            continue

        self_test = getattr(adapter, "self_test", None)
        if self_test is None:
            # Most adapters will not implement one. Absence is not failure.
            continue
        try:
            status = await self_test()
        except Exception as exc:  # noqa: BLE001
            status = HealthStatus(ok=False, detail=f"probe raised: {exc}")
        report.add(f"{name}_pipeline", status.ok, status.detail or "")


async def _check_catalog(report: Report, settings) -> None:
    from memgw.catalog import Catalog

    # Construction is inside the try as well: create_async_engine imports the DBAPI
    # eagerly, so a Postgres URL with no driver installed raises here rather than on
    # first use -- and a stack trace ending in ModuleNotFoundError is precisely the
    # answer doctor exists to replace.
    catalog = None
    try:
        catalog = Catalog(settings.catalog_url)
        await catalog.init()
        report.add("catalog", True, f"reachable at {_redact(settings.catalog_url)}")
    except Exception as exc:  # noqa: BLE001 -- doctor reports, never raises
        report.add("catalog", False, f"{_redact(settings.catalog_url)}: {_driver_hint(exc)}")
        if catalog is not None:
            await catalog.close()
        return

    # An old database under a new binary fails at the first query that mentions a
    # column the migration never added -- which is to say, in production, on a
    # request, rather than here.
    try:
        state = await catalog.schema_state()
        report.add("schema", state.up_to_date, state.describe())
    except Exception as exc:  # noqa: BLE001
        report.add("schema", False, f"could not be read: {exc}")
    finally:
        await catalog.close()


def _check_qdrant(report: Report, settings) -> None:
    """A TCP connect, nothing more. Mem0 will not tell you its store is unreachable
    until the first write, and then it tells you with a timeout."""
    host, port = settings.qdrant_host, settings.qdrant_port
    try:
        with socket.create_connection((host, port), timeout=2):
            report.add("qdrant", True, f"reachable at {host}:{port}")
    except OSError as exc:
        report.add(
            "qdrant",
            False,
            f"nothing listening at {host}:{port} ({exc}). Mem0 needs a Qdrant "
            "server; its local file mode deadlocks under async. "
            "docker compose -f docker-compose.test.yml up -d",
        )


def render(report: Report) -> str:
    lines = [f"{'PASS' if c.ok else 'FAIL'}  {c.name:<16} {c.detail}" for c in report.checks]
    lines.append("")
    lines.append("ready to serve" if report.ok else "not ready -- fix the FAIL lines above")
    return "\n".join(lines)


#: Marks the handler this module installed, so a reload adds nothing a second time.
_HANDLER_TAG = "memgw-cli"


def configure_logging(level: str = "info") -> None:
    """Turn the audit trail on.

    :mod:`memgw.observability` attaches a ``NullHandler`` and sets no level, which is
    what a library should do -- importing memgw must not change an unrelated
    program's output. The consequence is that a plain ``memgw serve`` emitted not one
    audit line: the logger had nowhere to write and no level to write at. Deciding
    that is the application's job, and this is the application.
    """
    log = logging.getLogger("memgw")
    log.setLevel(getattr(logging, level.upper(), logging.INFO))

    if any(getattr(h, "_memgw_tag", None) == _HANDLER_TAG for h in log.handlers):
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    handler._memgw_tag = _HANDLER_TAG  # type: ignore[attr-defined]
    log.addHandler(handler)


def asgi_app():
    """Build the app from the environment. ``uvicorn --factory memgw.cli:asgi_app``.

    Not ``async``. uvicorn calls a factory synchronously, so an async one hands it a
    coroutine, which it mistakes for an ASGI2 app -- and every request, ``/healthz``
    included, dies with ``'coroutine' object is not callable``."""
    from memgw.settings import Settings, create_app_from_settings

    # uvicorn imports this module fresh in its own process when reloading, so the
    # audit log has to be turned on here too, not only in main().
    configure_logging(os.environ.get("MEMGW_LOG_LEVEL", "info"))
    return create_app_from_settings(Settings.from_env())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="memgw", description="One API in front of any AI memory provider"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP gateway")
    serve.add_argument("--host", default=os.environ.get("MEMGW_HOST", "0.0.0.0"))
    serve.add_argument("--port", type=int, default=int(os.environ.get("MEMGW_PORT", "8080")))

    sub.add_parser("migrate", help="bring the catalog schema up to head")

    check = sub.add_parser("doctor", help="check the configuration before anything depends on it")
    check.add_argument(
        "--probe",
        action="store_true",
        help="also ask each provider to prove it works: writes to a throwaway "
        "subject and deletes it",
    )

    args = parser.parse_args(argv)
    configure_logging(os.environ.get("MEMGW_LOG_LEVEL", "info"))

    if args.command == "migrate":
        raise SystemExit(asyncio.run(_migrate()))

    if args.command == "doctor":
        report = asyncio.run(doctor(probe=args.probe))
        print(render(report))
        raise SystemExit(0 if report.ok else 1)

    if args.command == "serve":
        # Refuse to start on a configuration that is already provably broken --
        # but never on a provider outage. See preflight().
        report = asyncio.run(preflight())
        if not report.ok:
            print(render(report), file=sys.stderr)
            raise SystemExit(1)

        import uvicorn

        uvicorn.run(
            "memgw.cli:asgi_app",
            factory=True,
            host=args.host,
            port=args.port,
            log_level=os.environ.get("MEMGW_LOG_LEVEL", "info"),
        )
        raise SystemExit(0)

    raise SystemExit(2)


async def _migrate() -> int:
    """Apply the migrations, and say what moved."""
    from memgw.catalog import Catalog
    from memgw.settings import Settings

    settings = Settings.from_env()
    try:
        catalog = Catalog(settings.catalog_url)
    except Exception as exc:  # noqa: BLE001
        print(f"cannot open {_redact(settings.catalog_url)}: {_driver_hint(exc)}", file=sys.stderr)
        return 1
    try:
        before = await catalog.schema_state()
        if before.up_to_date:
            print(f"already {before.describe()}")
            return 0
        await catalog.upgrade()
        after = await catalog.schema_state()
        print(f"migrated {before.current or 'empty'} -> {after.current}")
        return 0 if after.up_to_date else 1
    except Exception as exc:  # noqa: BLE001 -- a migration failure is not a traceback
        print(f"migration failed: {_driver_hint(exc)}", file=sys.stderr)
        return 1
    finally:
        await catalog.close()


def _driver_hint(exc: Exception) -> str:
    """Name the package to install, when that is what went wrong.

    "No module named 'asyncpg'" is technically a complete answer and practically a
    search. The install command is the whole content of the message."""
    text = str(exc)
    for module, extra in (("asyncpg", "postgres"), ("aiosqlite", ""), ("psycopg", "postgres")):
        if module in text:
            install = f"pip install 'memgw[{extra}]'" if extra else f"pip install {module}"
            return f"the {module} driver is not installed -- {install}"
    return text


def _flagged_off(raw: str | None) -> bool:
    return (raw or "").strip().lower() in ("0", "false", "no", "off")


def _redact(url: str) -> str:
    """A database URL routinely carries a password, and doctor output gets pasted
    into issues."""
    if "://" not in url or "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.rpartition("@")
    return f"{scheme}://***@{host}"


if __name__ == "__main__":  # pragma: no cover
    main()
