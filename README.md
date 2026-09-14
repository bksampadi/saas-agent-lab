# SaaS Agent Lab

A small B2B SaaS administration system — users, licences, licence assignments
and an audit log — built as a backend engineering project. Later versions add
an agent that operates the system through its API, then through the browser.

**Status:** v0.1 in progress. Project scaffold, data models and migrations are
in place; CRUD endpoints and the UI are next.

## Stack

Python 3.12 · FastAPI · SQLAlchemy 2 · Alembic · Pydantic v2 · SQLite · pytest · ruff · uv

## Run locally

```bash
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload
```

Check it's up at <http://127.0.0.1:8000/health>. Settings come from `SAL_*`
environment variables; see [`.env.example`](.env.example).

## Test and lint

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Tests run against in-memory SQLite and never touch a real database or the network.

## Layout

```
src/app/
  api/       HTTP routers
  schemas/   Pydantic request/response models
  models/    SQLAlchemy models
  core/      settings and database setup
alembic/     migrations
tests/
```

## Roadmap

- **v0.1** SaaS app: CRUD, audit log, static UI, tests
- **v0.2** API agent: tool loop, approval for destructive actions, postcondition checks
- **v0.3** Browser agent: the same tasks through the UI with Playwright
- **v0.4** Production concerns: Postgres, retries, idempotency, concurrency, observability
- **v1.0** Docker Compose, architecture diagram, write-up
