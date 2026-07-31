# memgw — Memory Gateway

One library and one API in front of any AI memory provider: Mem0, a self-hosted
Postgres+pgvector store, and — as adapters land — Zep, Supermemory, Letta.

Two things drive the design, and they need different shapes:

| Goal | Mode |
| --- | --- |
| Integrate once, change provider by configuration | **embedded** — one provider, nothing to deploy |
| Put each end-user on a different provider | **proxy** — bindings, tenancy, one gateway |

Routing is not retrofitted into embedded mode. It needs state that outlives a
process, and state that outlives a process is a server.

## Install

```bash
pip install -e ".[dev]"          # library + tests
pip install -e ".[server]"       # + the HTTP gateway
pip install -e ".[mem0]"         # + the Mem0 adapter
pip install -e ".[postgres]"     # + Postgres drivers
```

## Embedded

```python
from memgw import Memory

mem = Memory(provider="pgvector", config={
    "url": "postgresql+asyncpg://…",   # or sqlite+aiosqlite:///memgw.db
    "embedder": my_embedder,           # see "You supply the embedder"
    "extractor": my_extractor,         # optional — see below
})

person = mem.scope("u_1", agent="lugo")
await person.ingest([{"role": "user", "content": "I drink black coffee"}], session="s_9")

hits = await person.search("coffee")               # across every session
one  = await person.search("coffee", session="s_9")  # this conversation only
```

## Proxy

```python
from memgw import Memory
mem = Memory(base_url="https://memgw.example.com", api_key="…")
```

Same methods, same exceptions. Code written against one mode works in the other.

Standing the gateway up is explicit rather than magic, because the self-hosted
adapter needs objects (an embedder) that no environment variable can express:

```python
from memgw.adapters.pgvector import PgvectorAdapter
from memgw.catalog import Catalog
from memgw.core import MemoryCore
from memgw.server import create_app

catalog = Catalog("postgresql+asyncpg://…")
await catalog.init()

app = create_app(
    core=MemoryCore(
        catalog=catalog,
        providers={"pgvector": PgvectorAdapter(…), "mem0": Mem0Adapter()},
        default_provider="pgvector",
        journal_enabled=False,
    ),
    api_keys={"sk-…": "tenant-a"},
)
```

Then `uvicorn yourmodule:app`. OpenAPI is served at `/openapi.json`.

## The contract

```
POST   /v1/memories:ingest       raw episode in, the provider extracts
POST   /v1/memories:upsert       ready-made facts in            → 501, see limits
POST   /v1/memories:search
GET    /v1/memories/{id}
PATCH  /v1/memories/{id}
DELETE /v1/memories/{id}
POST   /v1/memories:delete       by scope (bulk / erasure)
GET    /v1/subjects/{subject}    current binding
POST   /v1/subjects/{subject}:rebind
GET    /v1/providers
GET    /v1/capabilities?provider=
GET    /healthz
```

### Scope is three dimensions, never one key

`subject` (who the memory is about) · `agent` (which persona remembers) ·
`session` (which conversation) · `labels` (everything else).

A composite key is tempting because callers usually already hold one. `Memory
.parse_scope("t/u_1/s_9", "{tenant}/{subject}/{session}")` will split it — in the
client. What travels on the wire is always the triple, because collapsing it costs
three things: recall across sessions (an adapter can only map an opaque blob onto
one native dimension, and "everything about this user" is the query memory exists
for), tenant isolation, and erasure by subject.

`subject` is required and non-empty.

### The gateway picks the provider, you may only assert it

Resolution is `binding(tenant, subject)` → `tenant default` → `400
no_provider_resolved`. Only writes create a binding; a search never pins an
end-user to whatever the default happened to be.

A request may carry `provider`. It is checked, not obeyed: disagreeing with the
binding is `409 provider_mismatch` and no provider call is made. This exists
because ingesting to one backend and recalling from another returns an empty
result with no error — the nastiest failure in this system, and a certainty once
end-users are spread across providers.

### Capabilities are per instance, and degradation has a floor

`GET /v1/capabilities` reports what *this configured provider* can do — the same
class answers differently depending on how it was built. Three fields matter more
than they look:

- **`consistency`** — `eventual` means ingest-then-search returns nothing for a
  while. Mem0 and Zep both behave this way and neither says so plainly.
- **`metered_externally`** — the provider spends on its own LLM and embedder
  inside `add()`/`search()`. Your cost figures are incomplete, and the gateway
  cannot fix that, only disclose it.
- **`memory_model`** — `flat_facts` / `temporal_graph` / `memory_blocks` /
  `documents`. Providers differ in kind, not in feature checklist.

`on_unsupported` defaults to `reject`. With `degrade`, only `graph`, `temporal`
and `hybrid` may fall back to `semantic`, and the response says what was lost.

> Degradation may reduce **quality**. It never reduces **isolation** or
> **deletability** — a scope dimension the provider cannot honour, or a missing
> delete, is `422` under every setting.

## Adding an adapter

Implement `memgw.adapters.base.MemoryAdapter`, register it, and pass the
conformance suite:

```python
# tests/conformance/test_yours.py
from tests.conformance.suite import ConformanceSuite

class TestYoursConformance(ConformanceSuite):
    async def make_adapter(self):
        return YourAdapter(...)
```

Six checks, chosen by your own `capabilities()` — except **scope isolation, which
never skips, for any adapter, under any declaration**. The suite catches a
declaration that drifts from behaviour: a mode declared and then refused, a
consistency promise you cannot keep, a filter that leaks. It cannot verify that a
declared `graph` search really traverses a graph; that stays yours.

An adapter never sees a gateway id. It speaks `native_id` and the catalog bridges
the two.

## What the MVP does not do

Stated here rather than discovered in production:

- **You supply the embedder.** `memgw` ships none — `Embedder` and `Extractor` are
  protocols you implement. A pgvector instance built without an extractor reports
  `supports_ingest: false` and refuses `ingest` rather than storing raw transcript
  and calling it memory.
- **`upsert` returns `501`.** Specified and published now because the migration
  engine replays through the fact path, and adding a verb later would break the
  API.
- **`rebind` only does `fresh_start`.** The new binding takes effect, old memories
  **stay at the old provider and are not deleted**, and the response says how many
  were stranded. `strategy: "migrate"` returns `501` until the migration engine
  lands. Telling you the end-user will lose their memories beats pretending the
  move worked.
- **The pgvector adapter scans.** Embeddings are stored as JSON and scored exactly
  over the scope on both SQLite and Postgres. A native `vector` column with `<=>`
  ordering is a follow-up. Exact search is correct at per-end-user size; it is not
  correct at millions of rows per subject.
- **`min_score` is provider-relative.** Scores are not comparable across
  providers, which is also the real blocker for multi-provider merge.
- **No multi-provider fan-out, no read merge, no dashboard**, and no Zep,
  Supermemory or Letta adapters. Letta is last on purpose: its unit is the agent,
  so it will report a reduced `scope_dims` rather than pretend to have subjects.
- **One API key per tenant, backend to backend.** Running the library inside an
  end-user's app needs short-lived subject-scoped tokens; the contract reserves
  that a `subject` in the token wins over a `subject` in the body, but no token
  minting ships here.
- **The Mem0 conformance run is opt-in.** It needs real credentials; set
  `MEMGW_MEM0_TEST=1`. Its filter tests — the ones guarding the silent-empty-recall
  trap — always run and need no dependency.

## Tests

```bash
.venv/bin/pytest          # this package only
.venv/bin/ruff check src tests
```

Design: `docs/superpowers/specs/2026-07-31-memory-gateway-design.md`
Plan: `docs/superpowers/plans/2026-07-31-memory-gateway-mvp.md`
