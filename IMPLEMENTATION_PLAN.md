# Mini-Town Behavior Upgrade

## Progress

- [x] 0. Establish baseline and LLM observability
  - Baseline: 54 backend tests passed.
  - Added prompt-free request metadata in `LLMProvider`.
- [x] 1. Add DeepSeek task-level thinking mode
  - `chat` mode sends `thinking.type=disabled` and keeps `temperature`.
  - `thinking` mode sends `thinking.type=enabled`, `reasoning_effort`, and omits `temperature`.
  - Existing tool-call turns preserve `reasoning_content`.
  - Verification: `58 passed` in `backend/tests`.
- [x] 2. Refactor schedules into fixed commitments and flexible routine blocks
  - [x] Add `TimeWindow` and `RoutineBlock` schema, validation, and Agent access.
  - [x] Add declarative blocks for every default-world resident.
  - [x] Use blocks only to replace legacy flexible slots; explicit commitments and urgent needs remain dominant.
- [x] 3. Add persistent daily coarse-grained plans
  - [x] Add `DailyPlan` and non-overlapping block parser/state transitions.
  - [x] Add daily-plan generation prompt interface.
  - [x] Store/reuse daily plans in the realtime engine and state broadcast without blocking simulation.
- [x] 4. Add candidate generation, deterministic scoring, and LLM selection
  - [x] Generate accessible routine-block candidates and score distance/social context.
  - [x] Use the top deterministic candidate as the compatibility fallback.
  - [x] Add non-blocking chat-model selection limited to scored candidate IDs, with deterministic fallback.
- [x] 5. Add behavior inertia and habit decay
  - [x] Learn only from voluntary routine-block actions.
  - [x] Reinforce success, penalize failure, decay by simulated time, and cap active habits.
- [x] 6. Add differentiated continuous emotion dynamics
  - [x] Add personality-sensitive event evaluation and baseline decay.
  - [x] Keep neutral emotion internal instead of writing it into every memory.
  - [x] Add focused emotion regression tests.
- [x] 7. Add structured reasoning-based memory formation
  - [x] High-value memories use a thinking-mode, fact-bounded interpretation call.
  - [x] Derived future intentions are persisted into MentalState.
  - [x] Validate belief updates and expose structured provenance in the UI.
- [x] 8. Connect memory, relationships, and dialogue triggers
  - [x] Co-location increases familiarity and idle co-located pairs can talk.
  - [x] Memory-origin intentions can supply a dialogue opening and public trigger metadata.
  - [x] Rank dialogue topics by shared-memory and relationship relevance.
- [x] 9. Expose behavioral causes in the frontend
  - [x] Show current action reason and learned behavior tendencies.
  - [x] Show memory-intention dialogue triggers in the event feed.
- [x] 10. Run multi-day regression and calibrate parameters
  - [x] Add multi-day routine/habit regression.
  - [x] Add memory-driven dialogue and failure-path scenarios.
  - [x] Verify a memory intention opens dialogue and is retained as public trigger metadata.

## Verification rule

Each step must add focused tests, pass those tests, then pass the full backend suite before the checklist advances.
