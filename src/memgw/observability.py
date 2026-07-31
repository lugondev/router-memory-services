"""The audit trail.

A gateway that sits between an application and its users' memories and keeps no
record of who read what is worse than no gateway: it centralises the access
without centralising the accounting.

Standard library ``logging`` only, and every field on ``extra`` rather than
interpolated into the message. That way a plain deployment gets readable lines
for free, and a deployment with a JSON formatter gets structured events without
memgw having to pick a logging vendor for anyone.

Never logged: memory content, episode text, api keys. Subject ids are logged
because an access log that cannot say *whose* memory was read is not an access
log; deployments treating subject ids as personal data can filter on the field
name, which is why it is a field and not part of the message.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

log = logging.getLogger("memgw")

#: A library decides *what* to log, never *where*. Without this, Python's last-resort
#: handler prints warnings to stderr of any application that never configured
#: logging -- so importing memgw would change an unrelated program's output.
log.addHandler(logging.NullHandler())

#: Set by the HTTP layer and read here, so the core stays unaware there is an HTTP
#: layer at all -- embedded callers simply get ``None``.
current_request_id: ContextVar[str | None] = ContextVar("memgw_request_id", default=None)


@contextmanager
def observe(verb: str, *, tenant: str, subject: str | None = None, **fields: Any):
    """Time one verb and emit exactly one event for it, success or failure.

    Yields a mutable dict: whatever the body puts in it (the resolved provider, the
    number of results) lands on the log record. The provider is usually unknown when
    the call starts, and an event that cannot name the provider is not much of an
    audit trail.
    """
    extra: dict[str, Any] = {"provider": None, **fields}
    started = time.perf_counter()
    try:
        yield extra
    except Exception as exc:
        _emit(
            logging.WARNING,
            verb,
            tenant,
            subject,
            extra,
            started,
            outcome="error",
            error=type(exc).__name__,
            code=getattr(exc, "code", None),
        )
        raise
    _emit(logging.INFO, verb, tenant, subject, extra, started, outcome="ok")


def _emit(
    level: int,
    verb: str,
    tenant: str,
    subject: str | None,
    extra: dict[str, Any],
    started: float,
    **outcome: Any,
) -> None:
    payload = {
        "verb": verb,
        "tenant": tenant,
        "subject": subject,
        "request_id": current_request_id.get(),
        "ms": round((time.perf_counter() - started) * 1000, 2),
        **extra,
        **outcome,
    }
    log.log(level, "%s %s", verb, payload.get("outcome"), extra=payload)
