# Mini-Town Architecture Implementation TODO

## Scope and principles

- Merge `CONTEXT_AND_STATE_ARCHITECTURE.md` and `INTERACTION_ARCHITECTURE_PLAN.md`.
- Implement in stages; run focused tests after every stage, then the full backend suite and frontend build.
- Preserve complete runtime state. LLM requests receive task-specific working context only.
- Build or select an intention first; query relevant locations/resources afterward; let the LLM choose only from code-validated candidates.
- Reuse current `MentalState`, `MemoryStore`, `RelationshipStore`, `DialogueStore`, `FactLedger`, `ResourceStore`, `InteractionEngine`, `FeasibilityResult`, `ActionExpectation`, `ActionOutcome`, `FactEvent`, `Task`, and existing frontend event fields where practical.
- Keep the world interface intentionally small for now. Do not implement a World Agent decision-maker, world stimulus queues, opportunities, or constraints in this pass.
- Keep legacy API and WebSocket fields when they still support the feature; add new fields compatibly before considering cleanup.
- Keep test databases isolated so one run cannot affect the next run.

## Stage 0: Baseline and test isolation

- [x] Record the baseline: backend tests and frontend production build.
- [x] Add a shared temporary database fixture for tests.
- [x] Ensure all SQLite-backed stores used by a test Engine share that test database.
- [x] Ensure tests never depend on `backend/data/town.db`.
- [ ] Close LLM clients, database connections, and background tasks after tests.
- [x] Verify `SimulationEngine.reset()` clears every runtime subsystem.
- [x] Add a test proving a second Engine does not inherit the first Engine's state.

Verification: focused isolation tests, full backend suite, frontend build.

## Stage 1: State ownership and MentalState

- [x] Keep `MentalState` as the complete runtime mental state.
- [x] Keep `context_dict()` for API/UI/debugging, but remove it from the default LLM hot path.
- [x] Add task-oriented mental summaries for action, planning, and dialogue contexts.
- [x] Audit implicit goal activation in `add_goal()`.
- [x] Separate pure intention reads from expiration mutations in `open_intentions()`.
- [x] Improve intention relevance selection beyond first type/location match.
- [ ] Document which goals, intentions, beliefs, plans, and habits are runtime-only versus restart-persistent.

Verification: cognition, intention, plan continuity, memory, and dialogue regression tests.

## Stage 2: Active considerations and intention-first decisions

- [ ] Add a small code-owned active-consideration model for commitments, urgent needs, plan steps, intentions, tasks, stimuli, and blocking facts.
- [x] Compute priority and urgency in code for commitments, urgent needs, plan steps, and open intentions.
- [ ] Use the complete priority order: safety/permissions, due commitments, urgent needs, current plan step, important open intentions, social context, habits/flexible activity.
- [x] Define a compact intention proposal schema without requiring an LLM-invented location.
- [ ] Allow fixed commitments and urgent needs to create system intentions without an LLM call.
- [x] Preserve deterministic fallback behavior.

Verification: priority, multiple-intention, urgent-need, fixed-commitment, and fallback tests.

## Stage 3: Unified WorldQuery

- [x] Add a small `WorldQuery` using the existing WorldPack, `LOCATION_MAP`, permission checks, pathfinding, `ResourceStore`, and observation snapshot.
- [x] Query locations after an intention is selected.
- [x] Filter by permission, reachability, distance, weather cost, task relevance, resources, facilities, and nearby agents.
- [x] Always retain the current location and required fixed-commitment location when applicable.
- [x] Return a bounded set, normally at most six relevant locations.
- [x] Keep "not retrieved" distinct from "does not exist".
- [x] Leave World Agent/stimulus/opportunity/constraint behavior unimplemented.

Verification: permissions, reachability, fixed-target retention, weather cost, resource relevance, and candidate limit tests.

## Stage 4: ContextService and task-specific contexts

- [x] Add a single `ContextService` instead of letting each caller assemble a full context.
- [x] Bind requests to agent ID, task type, simulation timestamp, state version, intention ID, plan ID, and plan step ID.
- [x] Define task-specific field allowlists for action decisions, daily plans, life plans, routine selection, dialogue, reflection, and memory formation.
- [x] Keep current time, location, status, needs, hard constraints, and current plan step available where required.
- [x] Reuse and extend `MemoryStore.retrieve()` for task/intention semantic relevance, while retaining recent/important/location retrieval.
- [x] Reuse relationship pair queries rather than loading the full relationship graph.
- [x] Filter future and agent-invisible facts before building context.
- [x] Track selected intention/location IDs, omitted sections, and input character size in bounded context metrics.
- [x] Keep existing API/UI complete mental-state responses compatible.

Verification: schema, permission, future-fact, prompt-size, retrieval, and frontend API compatibility tests.

## Stage 5: Integrate intention-first planning

- [x] Refactor the compatibility `decide_action()` path to propose an intention before location lookup.
- [x] Query WorldQuery after intention selection on that path.
- [x] Resolve selected-intention location views before normal daily/life-plan generation; final action feasibility remains code-validated.
- [x] Keep candidate selection restricted to code-generated candidates where already implemented.
- [x] Re-run `InteractionEngine.validate()` immediately before execution.
- [ ] Keep life-plan steps focused on intention/outcome rather than prematurely fixed locations where possible; current compatibility schema still accepts explicit validated locations.
- [x] Keep high-frequency calls in chat mode and low-frequency thinking calls unchanged unless measurement justifies a change.

Verification: call ordering, unknown-intention, candidate restriction, fallback, and multi-day behavior tests.

## Stage 6: State versions and stale LLM results

- [x] Add an Engine `state_version`.
- [x] Increment it at tick boundaries for request diagnostics.
- [x] Store request timestamp/version/agent/intention/plan metadata on deferred tasks.
- [x] Re-check agent, intention, plan step, location, and time on LLM completion; permissions/resources are revalidated at action commit.
- [x] Drop and requeue stale planning, daily-plan, and dialogue work; stale routine choices use deterministic fallback.
- [x] Use deterministic fallback for stale routine/commitment paths where an applicable fallback exists.
- [x] Prevent reset from allowing old tasks to mutate the new run.
- [x] Add stale/retry/fallback trace metadata.

Verification: delayed LLM, resource changes, movement, completed intention, reset, and out-of-order completion tests.

## Stage 7: Unified interaction lifecycle

- [x] Audit overlap among `FeasibilityResult`, `ActionExpectation`, `ActionOutcome`, `FactEvent`, `Task`, and frontend events.
- [x] Introduce one lightweight canonical `InteractionRecord`/`InteractionStore` without replacing existing validation and outcome views.
- [x] Carry `interaction_id` through action, outcome, fact, dialogue, relationship update, and frontend event paths.
- [x] Preserve existing external fields while adding compatibility fields.
- [x] Support current movement, activity, and dialogue paths; service/resource coverage reuses the same action commit path.
- [x] Keep existing validation, expectation, outcome, fact, task, and dialogue models as compatible views while the canonical interaction record covers their shared identity/lifecycle.

Verification: interaction lifecycle, success/failure effects, dialogue, resource, source, and API compatibility tests.

## Stage 8: Persistent cognition boundary

- [x] Keep memory, relationships, and dialogue persistence in their current Stores.
- [x] Decide the current boundary: goals, intentions, beliefs, current episode, recent outcomes, recent interactions, life plans, and habits are restart-persistent; instant weather and movement are runtime-only.
- [x] Add dedicated cognition persistence for the confirmed restart-persistent state.
- [x] Keep instant weather and runtime state separate from durable behavioral evidence.
- [x] Persist effects and sources when transient state changes long-term behavior.
- [x] Test restart recovery and reset isolation.

Verification: restart, reset, long-term memory, habit, intention, belief, and future-fact tests.

## Stage 9: Frontend causal display

- [x] Extend frontend event types with optional interaction/source/parent/outcome fields.
- [x] Preserve existing event types and fields.
- [x] Link backend action/dialogue/fact/memory-source/relationship records by interaction ID; frontend event types and feed carry the ID, with full interaction detail UI deferred.
- [x] Display concise cause and outcome information.
- [x] Fix pause incorrectly clearing trace and relationship panels.
- [x] Fix control requests that can leave buttons pending after a network error.

Verification: TypeScript build, API compatibility, event linking, pause/reset, and WebSocket ordering tests.

## Stage 10: Full regression and calibration

- [x] Run the complete backend suite.
- [x] Run the frontend production build.
- [x] Run fallback multi-day simulation.
- [x] Run mocked/live-mode multi-day planning scenarios.
- [x] Test memory-driven dialogue and resource interactions.
- [x] Test failures and replanning.
- [x] Measure input character size through ContextService metrics and LLM request metadata; token/latency baseline remains for live deployment.
- [ ] Compare repeated/invalid actions and meaningful follow-up interactions before and after the change.

## Acceptance criteria

- Complete state is never discarded because prompt context is trimmed.
- Each data type has one defined source of truth.
- LLM receives task-relevant context, not the complete world and complete mental state by default.
- Intentions precede relevant location/resource queries where LLM choice is needed.
- Code owns permissions, facts, priorities, feasibility, and effects.
- Stale LLM results cannot commit invalid actions.
- Interaction IDs connect actions, outcomes, facts, durable effects, and frontend events.
- Existing APIs, fallback behavior, and frontend build remain functional.
- World Agent remains a future query/interface extension, not a decision-maker.
