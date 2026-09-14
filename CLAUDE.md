# SaaS Agent Lab

A small B2B SaaS administration system (users, licences, assignments, audit log) plus — later — an AI agent that operates it through the API and, later still, through the browser. The goal is ordinary backend engineering done well: layered architecture, databases, tests, failure handling. It is not an AI showcase. Keep it boring and correct.

## Current phase: v0.1 — the SaaS application, no agent

Nothing agent- or LLM-related gets built until CRUD, the audit log, the static UI and the test suite are complete. If a task would add agent code during v0.1, stop and say so.

## Roadmap (do not skip ahead)

- v0.1 full-stack SaaS: HTML/CSS/JS → FastAPI → SQLAlchemy/SQLite → CRUD → tests
- v0.2 action agent: plain-Python tool loop → explicit state → approval for destructive ops → audit → postcondition verification. No LangGraph.
- v0.3 browser agent: Playwright completes the same tasks through the UI
- v0.4 production engineering: Postgres, migrations, retries, idempotency, concurrency, background jobs, observability
- v1.0 public release: Docker Compose, CI, README, architecture diagram, write-up

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

create user · deactivate user · assign licence · revoke licence

Rules: deactivating a user revokes all their active assignments. A licence with no free seats cannot be assigned. Assigning an already-assigned licence to the same user is a no-op, not an error.

## Stack

Python 3.12, uv, FastAPI, SQLAlchemy 2.x, Alembic, Pydantic v2, pydantic-settings, pytest, httpx (test client), ruff. Frontend: plain HTML/CSS/JS served from `static/`. No frontend framework in v0.1.

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
