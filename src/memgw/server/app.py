"""The gateway app.

One exception handler covers the whole error vocabulary, because every failure a
caller can act on is a ``GatewayError`` carrying its own status. FastAPI's default
422 for a malformed body is remapped to 400: 422 is reserved here for
``unsupported_capability``, and two different meanings on one status is how a
client ends up retrying something that will never work.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from memgw.core import MemoryCore
from memgw.errors import GatewayError, InvalidRequest
from memgw.server.auth import ApiKeyAuth
from memgw.server.routes import router


def create_app(*, core: MemoryCore, api_keys: dict[str, str]) -> FastAPI:
    app = FastAPI(
        title="memgw",
        version="0.1.0",
        summary="One API in front of any AI memory provider",
    )
    app.state.core = core
    app.state.auth = ApiKeyAuth(api_keys)

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
