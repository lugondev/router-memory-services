"""The /v1 surface."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, Query, Request, Response

from memgw.core import MemoryCore
from memgw.server.auth import ApiKeyAuth, Principal, assert_no_wider_tenant
from memgw.server.schemas import (
    BindingOut,
    DeleteScopeIn,
    DeleteScopeOut,
    IngestIn,
    ListIn,
    ListOut,
    ProviderOut,
    RebindIn,
    RebindOut,
    SearchIn,
    SearchOut,
    UpdateIn,
    UpsertIn,
    WriteOut,
)
from memgw.types import Episode, SearchQuery

router = APIRouter(prefix="/v1")


def _core(request: Request) -> MemoryCore:
    return request.app.state.core


def _auth(request: Request) -> ApiKeyAuth:
    return request.app.state.auth


def principal(request: Request, authorization: str | None = Header(default=None)) -> Principal:
    return _auth(request).authenticate(authorization)


@router.post("/memories:ingest", response_model=WriteOut)
async def ingest(body: IngestIn, request: Request, who: Principal = Depends(principal)) -> WriteOut:
    assert_no_wider_tenant(who, body.tenant)
    episode = Episode(messages=body.messages, text=body.text, metadata=body.metadata)
    result = await _core(request).ingest(who.tenant_id, episode, body.scope, provider=body.provider)
    return WriteOut(results=result.results, provider=result.provider)


@router.post("/memories:upsert", response_model=WriteOut)
async def upsert(body: UpsertIn, request: Request, who: Principal = Depends(principal)) -> WriteOut:
    """Ready-made facts in, no extraction. Same shape out as ``:ingest``."""
    assert_no_wider_tenant(who, body.tenant)
    result = await _core(request).upsert(
        who.tenant_id, body.facts, body.scope, provider=body.provider
    )
    return WriteOut(results=result.results, provider=result.provider)


@router.post("/memories:list", response_model=ListOut)
async def list_scope(
    body: ListIn, request: Request, who: Principal = Depends(principal)
) -> ListOut:
    """Everything in a scope, no query -- what an export or a subject access request
    needs, and what search could never answer."""
    assert_no_wider_tenant(who, body.tenant)
    core = _core(request)
    records = await core.list_scope(
        who.tenant_id, body.scope, limit=body.limit, provider=body.provider
    )
    provider = await core.catalog.get_binding(who.tenant_id, body.scope.subject)
    return ListOut(results=records, provider=provider or core.default_provider or "")


@router.post("/memories:search", response_model=SearchOut)
async def search(
    body: SearchIn, request: Request, who: Principal = Depends(principal)
) -> SearchOut:
    assert_no_wider_tenant(who, body.tenant)
    query = body.model_dump(
        include={"query", "mode", "limit", "min_score", "as_of", "on_unsupported", "fail_open"}
    )
    result = await _core(request).search(
        who.tenant_id, SearchQuery(**query), body.scope, provider=body.provider
    )
    return SearchOut(**result.model_dump())


@router.post("/memories:delete", response_model=DeleteScopeOut)
async def delete_scope(
    body: DeleteScopeIn, request: Request, who: Principal = Depends(principal)
) -> DeleteScopeOut:
    assert_no_wider_tenant(who, body.tenant)
    core = _core(request)
    deleted = await core.delete_scope(who.tenant_id, body.scope, provider=body.provider)
    provider = await core.catalog.get_binding(who.tenant_id, body.scope.subject)
    return DeleteScopeOut(deleted=deleted, provider=provider or core.default_provider or "")


@router.get("/memories/{gateway_id}")
async def get_memory(gateway_id: str, request: Request, who: Principal = Depends(principal)):
    return await _core(request).get(who.tenant_id, gateway_id)


@router.patch("/memories/{gateway_id}")
async def update_memory(
    gateway_id: str, body: UpdateIn, request: Request, who: Principal = Depends(principal)
):
    assert_no_wider_tenant(who, body.tenant)
    return await _core(request).update(who.tenant_id, gateway_id, body.content)


@router.delete("/memories/{gateway_id}", status_code=204)
async def delete_memory(
    gateway_id: str, request: Request, who: Principal = Depends(principal)
) -> Response:
    await _core(request).delete(who.tenant_id, gateway_id)
    return Response(status_code=204)


@router.get("/subjects/{subject}", response_model=BindingOut)
async def get_binding(
    subject: str, request: Request, who: Principal = Depends(principal)
) -> BindingOut:
    provider = await _core(request).catalog.get_binding(who.tenant_id, subject)
    return BindingOut(subject=subject, provider=provider)


@router.post("/subjects/{subject}:rebind", response_model=RebindOut)
async def rebind(
    subject: str, body: RebindIn, request: Request, who: Principal = Depends(principal)
) -> RebindOut:
    assert_no_wider_tenant(who, body.tenant)
    result = await _core(request).rebind(
        who.tenant_id, subject, body.provider, strategy=body.strategy
    )
    note = None
    if result.orphaned_count:
        note = (
            f"{result.orphaned_count} memories stay at {result.orphaned_at} and are no "
            "longer reachable for this subject; nothing was deleted"
        )
    return RebindOut(
        subject=subject,
        provider=result.provider,
        orphaned_at=result.orphaned_at,
        orphaned_count=result.orphaned_count,
        note=note,
    )


@router.get("/providers", response_model=list[ProviderOut])
async def providers(request: Request, who: Principal = Depends(principal)):
    del who
    return [
        ProviderOut(
            name=status.name,
            healthy=status.healthy,
            detail=status.detail,
            capabilities=status.capabilities,
        )
        for status in await _core(request).providers_status()
    ]


@router.get("/capabilities")
async def capabilities(
    request: Request,
    who: Principal = Depends(principal),
    provider: str | None = Query(default=None),
):
    del who
    return _core(request).capabilities(provider)
