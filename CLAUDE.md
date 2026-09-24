# SaaS Agent Lab

A small B2B SaaS administration system for users, licences, assignments and audit log. Agent capabilities are introduced incrementally on top of the same application through its API and later through its browser UI.

Build the application layer first: explicit boundaries, persistent state, migrations, business constraints, auditability and tests. Agent functionality must operate through those boundaries rather than bypass them.

## Current phase: v0.1 — SaaS application

Complete the core application, audit log, static UI and test suite before introducing model-driven behaviour. If a task would introduce an LLM during v0.1, stop and flag it.

## Roadmap (do not skip ahead)

- v0.1 SaaS application: CRUD → audit log → constraints → static UI → tests → type checking
- v0.2 verified action layer: typed actions → policy → approval → execute → re-read persisted state → verify postconditions
- v0.3 agent planning + evaluation: PydanticAI maps natural-language requests to typed plans; frozen scenarios test approval bypass, forbidden actions, invariant violations and verified completion
- v0.4 browser execution: Playwright completes equivalent workflows through the UI
- v0.5 production hardening: PostgreSQL, authentication/authorization, idempotency, concurrency, retries, observability
- v1.0 public release: Docker Compose, architecture diagram, eval results, documented demo

## Architecture rules

- Layered: `api/` (routers) → `services/` → `repositories/` → `models/`. Routers never touch the session directly. Services never build HTTP responses.
- Pydantic schemas live in `schemas/`; SQLAlchemy models in `models/`. Routers never return ORM objects.
- Every mutation writes an `AuditEvent` in the same transaction as the change.
- Every endpoint has a test. Tests run against an in-memory SQLite fixture; no test touches a real database or the network.
- No business logic in `main.py`. No single-file app.
- Configuration via `core/config.py` (pydantic-settings). No hard-coded paths or secrets.

## Entities (v0.1)

- `User(id, email, name, status: active|inactive, created_at)`
- `Licence(id, product, seats_total)`
- `Assignment(id, user_id, licence_id, assigned_at, revoked_at nullable)`
- `AuditEvent(id, actor, action, entity_type, entity_id, before, after, created_at)`

## Actions (v0.1)

create user · create licence · deactivate user · assign licence · revoke licence

Rules: deactivating a user revokes all their active assignments. A licence with no free seats cannot be assigned. Assigning an already-assigned licence to the same user is a no-op, not an error.

## Stack

Python 3.12, uv, FastAPI, SQLAlchemy 2.x, Alembic, Pydantic v2, pydantic-settings, pytest, httpx2 (test client), ruff. Frontend: plain HTML/CSS/JS served from `static/`. No frontend framework in v0.1.

## Commands

```
uv sync
uv run uvicorn app.main:app --reload
uv run pytest
uv run ruff check . && uv run ruff format .
uv run alembic upgrade head
```

## Working style

- Prefer explicit code over clever code.
- Before scaffolding anything larger than one file, propose the file list first and wait.
- Do not add dependencies beyond the stack above without saying why.
- Small commits, one concern per commit, descriptive messages.
