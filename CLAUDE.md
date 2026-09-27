# SaaS Agent Lab

A small B2B SaaS administration system for users, licences, assignments and audit log. Agent capabilities are introduced incrementally on top of the same application through its API and later through its browser UI.

Build the application layer first: explicit boundaries, persistent state, migrations, business constraints, auditability and tests. Agent functionality must operate through those boundaries rather than bypass them.

## Current state

- v0.1 SaaS application: complete.
- Day 1, deterministic agent execution foundation: complete. Persisted `AgentRun` and `ToolCall`, deterministic resolver, persisted resolved goal, goal-scoped executor with separate log and business transactions, audit actor `agent:run-<id>`, state-based verifier.
- Day 2A, natural-language intent extraction: complete. PydanticAI turns an instruction into a closed `ExtractedIntent` union (`EnsureAssignmentIntent | NeedsClarification | Unsupported`). Every model request is persisted as a `ModelCall`; model calls and tool calls share one ordered trace per run.
- Day 2B, bounded model-directed execution: complete. `AgentExecutor.decide` lets a model choose among four argument-free tools bound to the run's persisted goal (`agent/decision_tools.py`), executed through the Day 1 executor. The model concludes with a closed proposal (`GoalReached | NoActionNeeded | CannotProceed`). Id-free observations are persisted on each `ToolCall`, and the initial context on the run. Limits are enforced by application code (`step_limit`). The run's outcome comes from `decision.decision_outcome`, never from the proposal.
- Next, Day 2C: not started. The live-provider test belongs there.

LLM and model integration is permitted, within the authority boundary below. `harness.run_instruction` still runs the deterministic Day 1 steps after extraction; `harness.run_directed_instruction` hands the resolved run to a decision planner.

## Authority boundary

Models interpret intent and choose bounded observations and actions. Deterministic application code owns everything else:

- identity resolution (which user and licence rows are meant),
- goal scope (what a run may touch),
- domain rules and policy,
- transactions,
- whether a tool call succeeded,
- postcondition verification (whether the goal holds).

A model's output is never evidence that something happened. Only the verifier, reading committed state, can complete a run.

## Roadmap (do not skip ahead)

- v0.1 SaaS application: CRUD, audit log, constraints, request transactions, tests, type checking
- v0.2 agent execution foundation: persisted runs and tool calls, deterministic resolution, goal-scoped typed tools, traces, postcondition verification
- v0.3 autonomous planning and human control: natural-language requests (Day 2A done), bounded model decisions and planner-visible observations (Day 2B done), policy checks, approval checkpoints, cancellation
- v0.4 browser execution and operator UI: Playwright, agent runs and approvals in a web interface
- v0.5 agent evaluation: frozen scenarios, state-based success, counter-evidence and observation coverage, model comparison, latency, tokens, cost
- v0.6 production hardening: PostgreSQL, authentication/authorization, durable execution, idempotency, concurrency, retries, observability
- v1.0 public release

## Architecture rules

- Layered: `api/` (routers) → `services/` → `repositories/` → `models/`. Routers never touch the session directly. Services never build HTTP responses.
- Pydantic schemas live in `schemas/`; SQLAlchemy models in `models/`. Routers never return ORM objects.
- Every mutation writes an `AuditEvent` in the same transaction as the change.
- Every endpoint has a test. Tests run against an in-memory SQLite fixture; no test touches a real database or the network.
- No business logic in `main.py`. No single-file app.
- Configuration via `core/config.py` (pydantic-settings). No hard-coded paths or secrets.

## Agent rules

- `agent/` holds the executor, resolver, verifier, tools, observations, the decision contract (limits, context, outcome table), harness and planners. The executor opens its own short sessions: log transactions (runs, tool calls, model calls) and business transactions are never open at the same time, and no session is open while a model runs.
- A model reaches the application only through a planner interface (`agent/planner.py`) with a fake-able implementation. Model output is a closed Pydantic union with `extra="forbid"`: no id fields, no free-form fields.
- Every model request, accepted, rejected or failed, is persisted as a `ModelCall` as soon as it has an outcome. Retries are bounded by explicit constants.
- `AgentRun.last_sequence_no` orders the trace and nothing else. It is not a limit.
- Tests never send a real model request: `tests/conftest.py` sets `pydantic_ai.models.ALLOW_MODEL_REQUESTS = False`. Use `FunctionModel`, `TestModel` or a fake planner. No test needs an API key.
- Default model: `SAL_PLANNER_MODEL`, `anthropic:claude-sonnet-5`. Model comparison belongs to evaluation, not to defaults.

## Decision-stage rules

- Model-facing tools take no arguments. The model chooses the capability; `GoalBoundTools` builds the call from the run's persisted goal and sends it through `AgentExecutor.call_decision_tool`. Never add an entity argument to a model-facing tool.
- Keep three representations of a tool call's result distinct:
  1. the internal executor result: may contain entity and database ids; used only by deterministic application code;
  2. the model-visible observation (`agent/observations.py`): a separate closed DTO with no entity or database ids, only what is intentionally shown to the model;
  3. the persisted model-visible observation (`ToolCall.observation`): the exact, deterministic serialization of (2), written with the call's outcome and returned to the model as that same text.
  Never use the internal result itself as the model observation, and never build model-visible text from `str(exception)`: rejections are closed reason codes.
- The initial model context (instructions and prompt) contains no ids and is persisted verbatim on the run (`decision_context`) before the first request.
- Execution limits are separate, explicit counters: `MAX_DECISION_MODEL_REQUESTS`, `MAX_DECISION_READ_CALLS` and `MAX_DECISION_MUTATION_CALLS` (`agent/decision.py`), counted from persisted rows. A mutation attempt counts even if rejected. Reaching a limit is an exception that stops the loop and ends the run FAILED (`step_limit`), never an observation. Never derive a limit from `last_sequence_no`.
- A domain rejection of an allowed call is an observation: the model may react to it.
- The model's proposal is recorded on the run and never sets its status. A `CannotProceed` reason blocks a run only when the application confirms it against current state (`verifier.check_block`).
- No forced precondition sweep: nothing is read on the model's behalf.

## Entities

- `User(id, email, name, status: active|inactive, created_at)`
- `Licence(id, product, seats_total)`
- `Assignment(id, user_id, licence_id, assigned_at, revoked_at nullable)`
- `AuditEvent(id, actor, action, entity_type, entity_id, before, after, created_at)`
- Agent: `AgentRun`, `ToolCall`, `ModelCall` (see `models/`)

## Actions

Implemented: create user · create licence · assign licence · revoke assignment. Not yet implemented: deactivate user (when it is, deactivating a user revokes all their active assignments).

Rules: a licence with no free seats cannot be assigned. Assigning a licence the user already actively holds is rejected with 409 Conflict (checked before seat capacity); once revoked, it can be assigned again.

## Stack

Python 3.12, uv, FastAPI, SQLAlchemy 2.x, Alembic, Pydantic v2, pydantic-settings, PydanticAI (`pydantic-ai-slim[anthropic]`), pytest, httpx2 (test client), ruff, pyright. No frontend yet; the operator UI is planned for v0.4.

## Commands

```
uv sync
uv run uvicorn app.main:app --reload
uv run pytest
uv run ruff check . && uv run ruff format .
uv run pyright
uv run alembic upgrade head
```

## Working style

- Prefer explicit code over clever code.
- Before scaffolding anything larger than one file, propose the file list first and wait.
- Do not add dependencies beyond the stack above without saying why.
- Small commits, one concern per commit, descriptive messages.
