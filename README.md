# SaaS Agent Lab

A small B2B SaaS administration system — users, licences, licence assignments
and an audit log — built as a backend engineering project. Later versions add
an agent that operates the system through its API, then through the browser.

**Status:** v0.1 in progress. Users can be created, listed and fetched through
the API, and every change is written to the audit log. Licences, assignments,
user deactivation and the UI are next.

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

## API

| Method | Path | |
|---|---|---|
| `POST` | `/users` | Create a user (email is trimmed and lowercased) |
| `GET` | `/users` | List users, ordered by id |
| `GET` | `/users/{user_id}` | Fetch one user |

Mutating endpoints require an `X-Actor` header naming who is making the change;
it is recorded as the actor on the audit event. This is **trusted,
caller-supplied identity for audit provenance, not authentication**: any caller
can send any value. Read endpoints don't need it.

```bash
curl -X POST http://127.0.0.1:8000/users \
  -H "X-Actor: admin@example.com" -H "Content-Type: application/json" \
  -d '{"email": "ada@example.com", "name": "Ada Lovelace"}'
```

## Test and lint

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Tests run against in-memory SQLite and never touch a real database or the network.

## Layout

```
src/app/
  api/           HTTP routers and request dependencies
  schemas/       Pydantic request/response models
  services/      business rules; own the transaction (commit/rollback)
  repositories/  database queries; add and flush, never commit
  models/        SQLAlchemy models
  core/          settings and database setup
alembic/         migrations
tests/
```

Requests flow `api → services → repositories → models`. Every change and its
audit event are committed in the same transaction.

## Roadmap

- **v0.1** SaaS app: CRUD, audit log, static UI, tests
- **v0.2** API agent: tool loop, approval for destructive actions, postcondition checks
- **v0.3** Browser agent: the same tasks through the UI with Playwright
- **v0.4** Production concerns: Postgres, retries, idempotency, concurrency, observability
- **v1.0** Docker Compose, architecture diagram, write-up
