"""The gateway app.

One exception handler covers the whole error vocabulary, because every failure a
caller can act on is a ``GatewayError`` carrying its own status. FastAPI's default
422 for a malformed body is remapped to 400: 422 is reserved here for
``unsupported_capability``, and two different meanings on one status is how a
client ends up retrying something that will never work.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from memgw.catalog import new_id
from memgw.core import MemoryCore
from memgw.errors import GatewayError, InvalidRequest
from memgw.observability import current_request_id
from memgw.server.auth import ApiKeyAuth
from memgw.server.routes import router


def create_app(
    *,
    core: MemoryCore | None = None,
    core_factory: Callable[[], Awaitable[MemoryCore]] | None = None,
    api_keys: dict[str, str],
    docs: bool = True,
) -> FastAPI:
    """The gateway app.

    ``core`` for a core you already built; ``core_factory`` for one that has to be
    awaited. The factory form exists because building a core opens databases, and
    an ASGI app must be constructible **without a running event loop** -- uvicorn's
    ``--factory`` calls its factory synchronously. Opening connections in a
    temporary loop and then serving from a different one is the other way to write
    this, and it fails later and more confusingly.

    ``docs`` turns off ``/docs`` and ``/redoc``. They need no credential and leak
    only the shape of the API, but a gateway standing in front of other people's
    memories should be able to decline to advertise itself.
    """
    if (core is None) == (core_factory is None):
        raise ValueError("pass exactly one of core= / core_factory=")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if core_factory is not None:
            app.state.core = await core_factory()
        yield
        built = getattr(app.state, "core", None)
        if built is not None and core_factory is not None:
            # Only close what this app opened. A caller-supplied core has a lifetime
            # the caller owns.
            await built.catalog.close()

    app = FastAPI(
        title="memgw",
        version="0.1.0",
        summary="One API in front of any AI memory provider",
        docs_url="/docs" if docs else None,
        redoc_url="/redoc" if docs else None,
        lifespan=lifespan,
    )
    app.state.core = core
    app.state.auth = ApiKeyAuth(api_keys)

    @app.middleware("http")
    async def _request_id(request: Request, call_next):
        """One id per request, echoed back and put on every log record the request
        produces. Without it an access log of a concurrent gateway is a pile of
        unrelated lines, and correlating a caller's complaint with what the gateway
        did means guessing from timestamps."""
        incoming = request.headers.get("x-request-id")
        request_id = incoming or new_id("req")
        token = current_request_id.set(request_id)
        try:
            response = await call_next(request)
        finally:
            current_request_id.reset(token)
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(GatewayError)
    async def _gateway_error(request: Request, exc: GatewayError) -> JSONResponse:
        del request
        return JSONResponse(status_code=exc.status, content=exc.to_body())

    def _as_invalid_request(exc: RequestValidationError | ValidationError) -> JSONResponse:
        # ctx carries the raw exception object, which is not JSON -- and leaking a
        # validator's internals to a caller would be noise at best.
        errors = [
            {key: value for key, value in error.items() if key not in ("ctx", "url")}
            for error in exc.errors()
        ]
        err = InvalidRequest("request body is not valid", details={"errors": errors})
        return JSONResponse(status_code=err.status, content=err.to_body())

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        del request
        return _as_invalid_request(exc)

    @app.exception_handler(ValidationError)
    async def _model_error(request: Request, exc: ValidationError) -> JSONResponse:
        """Domain models validate inside the routes too -- Episode's "exactly one of
        messages/text" lives on the type, not on the request body, so the rule is
        stated once. Without this handler that rule would surface as a 500."""
        del request
        return _as_invalid_request(exc)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(router)
    return app
