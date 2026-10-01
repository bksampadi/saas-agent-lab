# SaaS Agent Lab

A small B2B SaaS administration system for users, licences, assignments and audit log. Agent capabilities are introduced incrementally on top of the same application through its API and later through its browser UI.

Build the application layer first: explicit boundaries, persistent state, migrations, business constraints, auditability and tests. Agent functionality must operate through those boundaries rather than bypass them.

## Current state

Latest release: `v0.3.0`. `main` is in development toward the next milestone, version `0.4.0.dev0`, and must not be released until approval work gives `AWAITING_APPROVAL` a way out.

- SaaS application: users, licences, assignments and revocation behind a layered FastAPI API; every mutation writes its audit event in the same transaction.
- Agent runs: `POST /agent-runs` runs one natural-language instruction synchronously through `AgentExecutor.run`, the only execution path:
  1. receive: the run is persisted `RECEIVED` before any model is called;
  2. extract: a PydanticAI intent planner returns a closed `ExtractedIntent` (`EnsureAssignmentIntent | NeedsClarification | Unsupported`); extracted text not found in the instruction is refused;
  3. resolve: the deterministic resolver (`agent/resolver.py`) turns the extracted email and product into row ids, or the run needs clarification;
  4. decide: a PydanticAI decision planner chooses among four argument-free tools (`agent/tools.py`). `AgentExecutor.call_tool` builds each call from the run's persisted goal, checks the run's limits and then policy (`agent/policy.py`), and only then runs it in its own business transaction. The model concludes with a closed proposal (`GoalReached | NoActionNeeded | CannotProceed`);
  5. verify: the verifier (`agent/verifier.py`) reads committed state, and `decision.decision_outcome` sets the run's outcome; the proposal never does.
- Policy: each licence has an `agent_policy` (`allow | require_approval | deny`, default `allow`). Deny ends the run `BLOCKED` (`policy_denied`); require_approval holds the call unrun and pauses the run `AWAITING_APPROVAL`. Neither opens a business transaction. Nothing leaves `AWAITING_APPROVAL` yet.
- Trace: every model request (`ModelCall`) and tool call (`ToolCall`) is persisted, in one order per run. `GET /agent-runs/{run_id}` returns the run, its decision context, its verification and its trace, id-free (`schemas/agent_runs.py`).
- One opt-in live test (`tests/live/`, marker `live`) drives a real model end to end.
- Next: approval checkpoints, then cancellation (see Roadmap). Not started.

LLM and model integration is permitted, within the authority boundary below.

## Authority boundary

Models interpret intent and choose bounded observations and actions. Deterministic application code owns everything else:

- identity resolution (which user and licence rows are meant),
- the goal (a run's tools act on its persisted user and licence, never on ids a model names),
- domain rules and policy,
- transactions,
- whether a tool call succeeded,
- postcondition verification (whether the goal holds).

A model's output is never evidence that something happened. Only the verifier, reading committed state, can complete a run.

## Roadmap (do not skip ahead)

Released:

- v0.1.0 SaaS application: CRUD, audit log, constraints, request transactions, tests, type checking
- v0.2.0 agent execution foundation: persisted runs and tool calls, deterministic resolution, goal-scoped typed tools, traces, postcondition verification
- v0.3.0 natural-language planning and bounded execution: intent extraction; bounded model-directed execution with id-free observations, deterministic verification and persisted model and tool traces; agent-run HTTP API and opt-in live-provider smoke test

Next milestone, in development on `main` (version `0.4.0.dev0`):

1. policy checks (on `main`)
2. approval checkpoints
3. cancellation

Later, in this order, not yet numbered:

4. browser execution and operator UI: Playwright, agent runs and approvals in a web interface
5. agent evaluation: frozen scenarios, state-based success, counter-evidence and observation coverage, model comparison, latency, tokens, cost
6. production hardening: PostgreSQL, authentication/authorization, durable execution, idempotency, concurrency, retries, observability

Then a public release.

## Architecture rules

- Layered: `api/` (routers) → `services/` → `repositories/` → `models/`. Routers never touch the session directly. Services never build HTTP responses.
- Pydantic schemas live in `schemas/`; SQLAlchemy models in `models/`. Routers never return ORM objects.
- Every mutation writes an `AuditEvent` in the same transaction as the change.
- Every endpoint has a test. Tests run against an in-memory SQLite fixture; no test touches a real database or the network.
- No business logic in `main.py`. No single-file app.
- Configuration via `core/config.py` (pydantic-settings). No hard-coded paths or secrets.

## Agent rules

- `agent/` holds the executor (`AgentExecutor.run`, each step and its transactions), the resolver, policy, the tools (what each does, and what the model is shown of it), the verifier, the decision contract (limits, context, outcome table), the planner boundary with its PydanticAI planners, and `runs.py` (start and read a run, for the API). The executor opens its own short sessions: log transactions (runs, tool calls, model calls) and business transactions are never open at the same time, and no session is open while a model runs.
- Every status change happens in the executor, after it checks the status it starts from; the lifecycle is written on `AgentRunStatus`. A run ends only through `_end`, and an ended run never changes. Which reason goes with which ending is a CHECK constraint.
- Policy is evaluated when a call is admitted (`_start_call`'s log transaction), after the run's limits and before any business transaction. Deny and require_approval never open a business transaction, and the model is asked nothing more.
- A run with a `STARTED` tool call (no recorded outcome: its change may or may not have committed) is never continued or verified.
- `ToolCall.tool_name` records the application operation (`tools.RECORDED_NAMES`); the model and the public trace use the tool's own name.
- A model reaches the application only through a planner interface (`agent/planner.py`) with a fake-able implementation. Model output is a closed Pydantic union with `extra="forbid"`: no id fields, no free-form fields.
- Every model request, accepted, rejected or failed, is persisted as a `ModelCall` as soon as it has an outcome. Retries are bounded by explicit constants.
- `AgentRun.last_sequence_no` orders the trace and nothing else. It is not a limit.
- Tests never send a real model request: `tests/conftest.py` sets `pydantic_ai.models.ALLOW_MODEL_REQUESTS = False`. Use `FunctionModel`, `TestModel` or a fake planner. No test needs an API key. The one exception is the live smoke test (`tests/live/`): deselected by default (`-m 'not live'` in addopts), skipped without the provider key, and the only place the guard is lifted (`override_allow_model_requests`). Never add a second one, and never retry it. Shared test helpers live in `tests/support.py` (imported as `support`), shared fixtures in the `conftest.py` files.
- Agent-run routes (`api/agent_runs.py`) are transport only and plain `def`. They take no request session or transaction: `agent/runs.py` gets the session factory (`deps.get_session_factory`). Planners are dependencies (`get_intent_planner`, `get_decision_planner`); API tests override them with scripted models.
- The public API shows a run only through `schemas/agent_runs.py`: never a resolved id, tool arguments, internal results, error messages or raw `outcome_detail`. Expected agent outcomes are `201` with the outcome in the body, never an HTTP error. Verification is read from the `outcome_detail` persisted at the end of the run, never recomputed from current state.
- Default model: `SAL_PLANNER_MODEL`, `anthropic:claude-sonnet-5`. Model comparison belongs to evaluation, not to defaults.

## Decision-stage rules

- Model-facing tools take no arguments. The model chooses the capability by name; `AgentExecutor.call_tool` builds the call from the run's persisted goal. Never add an entity argument to a model-facing tool.
- Keep three representations of a tool call's result distinct:
  1. the internal executor result: may contain entity and database ids; used only by deterministic application code;
  2. the model-visible observation (`tools.observe`): a separate closed DTO with no entity or database ids, only what is intentionally shown to the model;
  3. the persisted model-visible observation (`ToolCall.observation`): the exact, deterministic serialization of (2), written with the call's outcome and returned to the model as that same text.
  Never use the internal result itself as the model observation, and never build model-visible text from `str(exception)`: rejections are closed reason codes.
- The initial model context (instructions and prompt) contains no ids and is persisted verbatim on the run (`decision_context`) before the first request.
- Execution limits are separate, explicit counters: `MAX_DECISION_MODEL_REQUESTS`, `MAX_DECISION_READ_CALLS` and `MAX_DECISION_MUTATION_CALLS` (`agent/decision.py`), counted from persisted rows. A mutation attempt counts even if rejected. Reaching a limit ends the run FAILED (`step_limit`) in the transaction that counts it and raises `DecisionStopped`, which stops the loop: never an observation. Never derive a limit from `last_sequence_no`.
- A domain rejection of an allowed call is an observation: the model may react to it.
- The model's proposal is recorded on the run and never sets its status. A `CannotProceed` reason blocks a run only when the application confirms it against current state (`verifier.check_block`).
- No forced precondition sweep: nothing is read on the model's behalf.

## Entities

- `User(id, email, name, status: active|inactive, created_at)`
- `Licence(id, product, seats_total, agent_policy: allow|require_approval|deny)`
- `Assignment(id, user_id, licence_id, assigned_at, revoked_at nullable)`
- `AuditEvent(id, actor, action, entity_type, entity_id, before, after, created_at)`
- Agent: `AgentRun`, `ToolCall`, `ModelCall` (see `models/`)

## Actions

Implemented: create user · create licence · assign licence · revoke assignment. Not yet implemented: deactivate user (when it is, deactivating a user revokes all their active assignments).

Rules: a licence with no free seats cannot be assigned. Assigning a licence the user already actively holds is rejected with 409 Conflict (checked before seat capacity); once revoked, it can be assigned again.

## Stack

Python 3.12, uv, FastAPI, SQLAlchemy 2.x, Alembic, Pydantic v2, pydantic-settings, PydanticAI (`pydantic-ai-slim[anthropic]`), pytest, httpx2 (test client), ruff, pyright. No frontend yet; the operator UI comes after policy checks, approvals and cancellation (see Roadmap).

## Commands

```
uv sync
uv run uvicorn app.main:app --reload
uv run pytest
uv run pytest -m live   # opt-in: real, billable model requests; needs ANTHROPIC_API_KEY
uv run ruff check . && uv run ruff format .
uv run pyright
uv run alembic upgrade head
```

## Working style

- Prefer explicit code over clever code.
- Before scaffolding anything larger than one file, propose the file list first and wait.
- Do not add dependencies beyond the stack above without saying why.
- Small commits, one concern per commit, descriptive messages.
