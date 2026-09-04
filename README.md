# memgw — Memory Gateway

One library and one API in front of any AI memory provider: Mem0, Zep, a
self-hosted Postgres+pgvector store, and — as adapters land — Supermemory, Letta.

Two things drive the design, and they need different shapes:

| Goal | Mode |
| --- | --- |
| Integrate once, change provider by configuration | **embedded** — one provider, nothing to deploy |
| Put each end-user on a different provider | **proxy** — bindings, tenancy, one gateway |

Routing is not retrofitted into embedded mode. It needs state that outlives a
process, and state that outlives a process is a server.

## Run it

One key, one command:

```bash
pip install -e ".[all]"

export OPENAI_API_KEY=sk-...
export MEMGW_API_KEYS=pick-a-long-random-string:tenant-a

memgw doctor     # says what is missing, and nothing else
memgw serve      # refuses to start on a configuration doctor already failed
```

```bash
curl -X POST localhost:8080/v1/memories:ingest \
  -H "Authorization: Bearer pick-a-long-random-string" \
  -H "Content-Type: application/json" \
  -d '{"scope":{"subject":"u_1"},
       "messages":[{"role":"user","content":"I only drink black coffee"}]}'
```

That runs the self-hosted store on SQLite, with the built-in OpenAI embedder and
extractor. Nothing else to write and nothing else to deploy.

The whole thing — gateway, Postgres, Qdrant — is one file away:

```bash
cp .env.example .env      # put your keys in it
docker compose up -d
```

Add providers by naming them. Each one only needs its own key:

```bash
MEMGW_PROVIDERS=pgvector,mem0,zep
ZEP_API_KEY=z_...
```

## Install

```bash
pip install -e ".[all]"          # every provider this build can configure
pip install -e ".[dev]"          # library + tests
pip install -e ".[server]"       # + the HTTP gateway
pip install -e ".[openai]"       # + the built-in embedder / extractor
pip install -e ".[mem0]"         # + the Mem0 adapter
pip install -e ".[zep]"          # + the Zep adapter
pip install -e ".[postgres]"     # + Postgres drivers
```

### Configuration

Everything is an environment variable, and `memgw doctor` checks every one that
can be checked before anything depends on it.

| | |
| --- | --- |
| `OPENAI_API_KEY` | the built-in embedder and extractor; Mem0's own LLM |
| `MEMGW_API_KEYS` | `key:tenant,key:tenant`. Unset means every request is a 401 |
| `MEMGW_PROVIDERS` | `pgvector`, `mem0`, `zep` — comma separated |
| `MEMGW_DEFAULT_PROVIDER` | where an unbound subject's first write lands |
| `MEMGW_CATALOG_URL` | SQLite by default, Postgres in anything with two processes |
| `MEMGW_JOURNAL` | keep raw episodes — needed for migration, and the most sensitive thing here |
| `MEMGW_DOCS` | `false` closes `/docs` and `/redoc`, which need no credential |
| `ZEP_API_KEY` | only if `zep` is in `MEMGW_PROVIDERS` |

```bash
memgw doctor           # configuration only; reads, never writes
memgw doctor --probe   # also makes each provider prove it works, end to end
memgw migrate          # bring the catalog schema to head
memgw serve
```

**`--probe` earns its keep.** Zep accepted every write with a `200` and an episode
id, never built the graph, and answered every read with an empty list and no error.
It was *reachable* the whole time, so no configuration check and no ping could see
it. The only question that can is "write something and read it back":

```
FAIL  zep_pipeline  episode ce9ce9b1 was accepted but stayed processed=false for
                    60s. Zep is reachable and is not building the graph, so every
                    read will return empty with no error at all.
```

`serve` deliberately runs the checks **without** the probe: a provider outage must
not stop your gateway from booting.

## Embedded

```python
from memgw import Memory
from memgw.embedding import OpenAIEmbedder, OpenAIExtractor

mem = Memory(provider="pgvector", config={
    "url": "postgresql+asyncpg://…",       # or sqlite+aiosqlite:///memgw.db
    "embedder": OpenAIEmbedder(),          # or bring your own, see below
    "extractor": OpenAIExtractor(),        # optional: without it, ingest is refused
})

person = mem.scope("u_1", agent="lugo")
await person.ingest([{"role": "user", "content": "I drink black coffee"}], session="s_9")
await person.remember("prefers oat milk")          # a fact you already know

hits = await person.search("coffee")               # across every session
one  = await person.search("coffee", session="s_9")  # this conversation only
all_ = await person.everything()                   # no query: what do you hold on me
```

## Proxy

```python
from memgw import Memory
mem = Memory(base_url="https://memgw.example.com", api_key="…")
```

Same methods, same exceptions. Code written against one mode works in the other.

Standing the gateway up is `memgw serve` — see **Run it** above. Write it out in
Python only when you need something the environment cannot express: your own
embedder, a provider built from objects, several catalogs in one process.

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
POST   /v1/memories:upsert       ready-made facts in, no extraction
POST   /v1/memories:search
POST   /v1/memories:list         a whole scope, no query (export / SAR)
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

Every response carries `x-request-id` — echoed if you send one, minted if you do
not — and it appears on every log line that request produced.

### Scope is separate dimensions, never one key

`subject` (who the memory is about) · `agent` (which persona remembers) ·
`session` (which conversation) · `labels` (everything else).

A composite key is tempting because callers usually already hold one. `Memory
.parse_scope("t/u_1/s_9", "{tenant}/{subject}/{session}")` will split it — in the
client. What travels on the wire is always the structured scope, because collapsing
it costs recall across sessions (an adapter can only map an opaque blob onto one
native dimension, and "everything about this user" is the query memory exists for)
and erasure by subject.

`subject` is required and non-empty.

### Tenant is a scope dimension, and the gateway stamps it

`scope.tenant` comes from the credential and is written over whatever a request
body claimed. It then travels to the adapter like any other dimension, because
**subject ids are chosen by tenants**: two tenants naming an end-user `u_1` are one
end-user to any provider filtered only by subject, and neither of them finds out.
The pgvector adapter has a `tenant` column; the Mem0 adapter namespaces `user_id`
as `tenant:subject`.

An adapter that cannot express the dimension omits `"tenant"` from its `scope_dims`
and the gateway refuses to route to it — `422`, loudly, rather than a store two
tenants silently share. The conformance suite checks this and **never skips it**.

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

> Degradation may reduce **quality**. It never reduces **isolation**,
> **deletability**, or **the question you asked** — a scope dimension the provider
> cannot honour, a missing delete, and an `as_of` a provider cannot answer are all
> `422` under every setting. Serving "what is true now" in place of "what was true
> then" is a confident wrong answer, not a degraded one.

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

The checks are chosen by your own `capabilities()` — except **tenant and subject
isolation, which never skip, for any adapter, under any declaration**. The suite
catches a declaration that drifts from behaviour: a mode declared and then refused,
a consistency promise you cannot keep, a filter that leaks, a `tenant` you declared
and then did not apply. It cannot verify that a declared `graph` search really
traverses a graph; that stays yours.

An adapter never sees a gateway id. It speaks `native_id` and the catalog bridges
the two.

## What the MVP does not do

Stated here rather than discovered in production:

- **One embedder ships, and it is OpenAI's.** `Embedder` and `Extractor` stay
  protocols so you can bring your own, but `OpenAIEmbedder` / `OpenAIExtractor` are
  included so a key is enough. No local or open-weight implementation ships yet. A
  pgvector instance built without an extractor still reports `supports_ingest:
  false` and refuses `ingest` rather than storing raw transcript and calling it
  memory.
- **Zep is experimental**, and says so in its own `capabilities()`. It has never
  passed the live conformance suite — see the Tests section. Every other provider
  here has been run against the real service.
- **Zep declares no `session` and no `agent`.** Its graph search filters by user,
  not by thread, so a session-scoped search against Zep is a `422`. That is the
  design working: the alternative is searching the whole user and calling the
  answer session-scoped.
- **`rebind` only does `fresh_start`.** The new binding takes effect, old memories
  **stay at the old provider and are not deleted**, and the response says how many
  were stranded. `strategy: "migrate"` returns `501` until the migration engine
  lands. Telling you the end-user will lose their memories beats pretending the
  move worked.
- **No ANN index.** On Postgres the adapter now uses a real `vector(n)` column and
  orders with `<=>`, so ranking and `limit` happen in the database — but without an
  HNSW index it still scans the scope. It scans in C over data that never moves,
  which is a different order of magnitude from the previous JSON-and-Python path,
  and it keeps search **exact**. An index would make it approximate; that is a
  different promise and a deliberate follow-up. On SQLite it scans in Python, which
  is the right answer for development and embedded mode.
- **Changing the embedding model needs a re-embed.** `vector(n)` has a fixed width,
  so a table built for one model refuses another at startup rather than mixing two
  geometries in one column, which nothing would raise on and only recall would show.
- **`min_score` is provider-relative.** Scores are not comparable across
  providers, which is also the real blocker for multi-provider merge.
- **No multi-provider fan-out, no read merge, no dashboard**, and no Supermemory
  or Letta adapters. Letta is last on purpose: its unit is the agent — everything
  hangs off `agent_id`, with no user concept at all — so it inverts this model
  rather than fitting into it.
- **One API key per tenant, backend to backend.** Keys are held and compared as
  digests, so a lookup leaks neither the key nor how close a guess got, and the
  `key_id` in logs is a digest prefix. What is *not* here: rate limiting, request
  size limits, and key rotation — put the gateway behind something that does those.
  Running the library inside an end-user's app needs short-lived subject-scoped
  tokens; the contract reserves that a `subject` in the token wins over a `subject`
  in the body, but no token minting ships here.
- **Search takes a `limit`, not a cursor.** There is no pagination, so a scope
  larger than `max_limit` cannot be walked. `:list` has the same ceiling.
- **`memgw migrate` applies schema changes; nothing applies them for you.** Startup
  never migrates: that is a deploy decision with a rollback attached, not a side
  effect of a process booting. `memgw doctor` fails when the database is behind the
  code, which is the moment you want to hear about it.
- **`GET /v1/providers` is the same answer for every tenant** — health and
  capabilities of every configured provider, to any valid key.
- **The Mem0 conformance run is opt-in.** It needs real credentials; set
  `MEMGW_MEM0_TEST=1`. Its filter tests — the ones guarding the silent-empty-recall
  trap — always run and need no dependency.

## Logs

Everything goes through the `memgw` logger, one event per verb, fields on
`extra` rather than interpolated into the message — so a plain deployment gets
readable lines and a JSON formatter gets structured events without memgw choosing
a logging vendor for anyone.

```python
import logging
logging.getLogger("memgw").setLevel(logging.INFO)
# verb, tenant, subject, provider, request_id, count, ms, outcome
```

Never logged: memory content, episode text, api keys. Subject ids *are* logged —
an access log that cannot say whose memory was read is not an access log — as a
named field, so a deployment that treats them as personal data can filter them.

## Tests

```bash
.venv/bin/pytest          # offline: no credentials, no services, no network
.venv/bin/ruff check src tests
```

The live Mem0 conformance run is opt-in, and needs a Qdrant server plus an OpenAI
key:

```bash
docker compose -f docker-compose.test.yml up -d
MEMGW_MEM0_TEST=1 MEM0_TELEMETRY=False OPENAI_API_KEY=sk-... .venv/bin/pytest
```

Three things about Mem0 that are not obvious and cost an afternoon each:

- **Pin the LLM.** Mem0 sends `temperature=0.1` and `top_p=0.1` with every
  extraction; a reasoning model answers `400`. `MEMGW_MEM0_MODEL` overrides the
  pinned default.
- **Do not use Mem0's local Qdrant with `AsyncMemory`.** It persists through SQLite
  while `AsyncMemory` dispatches each call on a different thread-pool thread, and it
  flocks its directory so a second client in the same process cannot open one.
- **`MEM0_TELEMETRY=False`.** Telemetry opens a *second*, undeclared local Qdrant
  under `~/.mem0/`, which then hits the lock above regardless of how you configured
  the real store.

The Zep conformance run is opt-in the same way:

```bash
MEMGW_ZEP_TEST=1 ZEP_API_KEY=z_... .venv/bin/pytest tests/conformance/test_zep.py
```

It has **not** been observed to pass. Against the account it was tried on, writes
return `200` with an episode id, `processed` stays `false` indefinitely, the
episode never appears in the user's own episode list, and search intermittently
answers `503`. Whether that is an outage or an unprovisioned project cannot be told
apart from the client side. The adapter is therefore verified against the SDK's
shape and not against its behaviour — which is exactly the distinction the tier
above exists to draw, and the reason `TestTheFakeMatchesTheRealClient` compares
every call the adapter makes against the installed client.

Design: `docs/superpowers/specs/2026-07-31-memory-gateway-design.md`
Plan: `docs/superpowers/plans/2026-07-31-memory-gateway-mvp.md`

---

## Part of LUGO

**LUGO** is a self-hosted AI companion platform — models supply the intelligence, LUGO
supplies the experience: one assistant that talks, remembers and acts across the browser,
ESP32 boards and a Raspberry Pi.

This repository is one piece of it. Every client and service talks to the gateway:

| Repo | Role |
| --- | --- |
| [lugo-gateway](https://github.com/lugondev/lugo-gateway) | The hub — STT/TTS/LLM engines, auth, device pairing, MCP tools, per-user chat memory. Everything below talks to this. |
| [lugo-web-client](https://github.com/lugondev/lugo-web-client) | React + TypeScript web client: talk, devices, history, tools. |
| [esp32-assistant](https://github.com/lugondev/esp32-assistant) | ESP-IDF firmware for ESP32-S3 / ESP32-C3 — a hands-free voice terminal. |
| [rpi-assistant](https://github.com/lugondev/rpi-assistant) | Raspberry Pi voice client (mic capture, Opus duplex, systemd unit). |
| [knowledge-api](https://github.com/lugondev/knowledge-api) | **kbase** — RAG knowledge base: documents in, retrievable chunks out. |
| **router-memory-services** &nbsp;&larr; you are here | **memgw** — one API in front of any AI memory provider (Mem0, Zep, pgvector). |
| [mcp-basic-tools](https://github.com/lugondev/mcp-basic-tools) | Remote MCP tool server (timedate, fetch, ipinfo, web search). |
| [livehost-api](https://github.com/lugondev/livehost-api) | TikTok Live AI co-host, an out-of-process gateway plugin. |
| [voiceprint-api](https://github.com/lugondev/voiceprint-api) | Speaker recognition (3D-Speaker), forked from [xinnan-tech/voiceprint-api](https://github.com/xinnan-tech/voiceprint-api). |
| [lugo-landing](https://github.com/lugondev/lugo-landing) | Marketing landing page for the platform, bilingual (Tiếng Việt / English). |
