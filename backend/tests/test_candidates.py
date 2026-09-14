import asyncio

from backend.town.candidates import build_routine_candidates
from backend.town.engine import SimulationEngine
from backend.town.routine_planner import RoutinePlanner
from backend.town.daily_plan import DailyPlan, DailyPlanBlock
from backend.town.schedule import RoutineBlock, TimeWindow


def run(coro):
    return asyncio.run(coro)


def test_flexible_routine_candidates_are_accessible_and_ranked(tmp_path):
    async def scenario():
        engine = SimulationEngine()
        engine.llm.fallback = True
        db_path = str(tmp_path / "town.db")
        engine.memory.db_path = db_path
        engine.relationship_store.db_path = db_path
        engine.trace.db_path = db_path
        await engine.init()
        agent = next(item for item in engine.agents if item.id == "mei")
        engine.agents = [agent]
        engine.hour, engine.minute = 18, 0

        candidates = build_routine_candidates(agent, engine)

        assert candidates
        assert all(candidate.action["source"] == "routine_block" for candidate in candidates)
        assert all(candidate.action["location"] in {"bookstore", "park", "restaurant", "home_mei", "cafe"}
                   for candidate in candidates)
        assert candidates == sorted(candidates, key=lambda candidate: (-candidate.score, candidate.id))
        assert all("当前处于下班后的个人时间时间区块" in candidate.reasons for candidate in candidates)

    run(scenario())


def test_flexible_block_overrides_legacy_slot_but_not_explicit_meal(tmp_path):
    async def scenario():
        engine = SimulationEngine()
        engine.llm.fallback = True
        db_path = str(tmp_path / "town.db")
        engine.memory.db_path = db_path
        engine.relationship_store.db_path = db_path
        engine.trace.db_path = db_path
        await engine.init()
        agent = next(item for item in engine.agents if item.id == "mei")
        engine.agents = [agent]
        planner = RoutinePlanner(engine.tasks)

        agent.routine_blocks.append(RoutineBlock(
            id="mei_lunch_flexible", window=TimeWindow(690, 780), intent="午间自由安排",
            candidate_locations=("cafe",), candidate_activities=("聊天",),
        ))
        engine.hour, engine.minute = 12, 0
        meal = planner.next_action(agent, engine)
        assert meal["source"] == "routine"
        assert meal["interaction_type"] == "consume"

        engine.hour, engine.minute = 18, 0
        flexible = planner.next_action(agent, engine)
        assert flexible["source"] == "routine_block"
        assert flexible["routine_block_id"] == "mei_evening"

    run(scenario())


def test_realtime_routine_choice_uses_ready_llm_selection(tmp_path):
    async def scenario():
        engine = SimulationEngine()
        engine.llm.fallback = False
        db_path = str(tmp_path / "town.db")
        engine.memory.db_path = db_path
        engine.relationship_store.db_path = db_path
        engine.trace.db_path = db_path
        await engine.init()
        agent = next(item for item in engine.agents if item.id == "mei")
        engine.agents = [agent]
        engine._realtime_llm = True
        engine.hour, engine.minute = 18, 0
        selected = []

        async def create_daily(planned_agent, planned_engine, _context, _reason=""):
            return DailyPlan(planned_agent.id, planned_engine.day, "晚间安排", [
                DailyPlanBlock("evening", TimeWindow(1020, 1260), "个人时间")
            ], planned_engine.get_sim_timestamp(), planned_engine.get_sim_timestamp() + 180)

        async def choose(planned_agent, planned_engine, candidates):
            selected.append([candidate.id for candidate in candidates])
            return dict(candidates[-1].action, selection_reason="想换个地方")

        engine.planner.create_daily_plan = create_daily
        engine.planner.select_routine_candidate = choose
        await engine.tick()
        routine_task = engine._routine_choice_tasks[agent.id][1]
        await routine_task
        await engine.tick()

        assert selected
        assert agent.state.status in {"MOVING", "ACTING"}
        assert agent._movement_source == "routine_block" or agent._activity_source == "routine_block"
        await engine.shutdown()

    run(scenario())


def test_no_candidates_outside_a_flexible_block(tmp_path):
    async def scenario():
        engine = SimulationEngine()
        engine.llm.fallback = True
        db_path = str(tmp_path / "town.db")
        engine.memory.db_path = db_path
        engine.relationship_store.db_path = db_path
        engine.trace.db_path = db_path
        await engine.init()
        agent = next(item for item in engine.agents if item.id == "mei")
        engine.agents = [agent]
        engine.hour, engine.minute = 10, 0

        assert build_routine_candidates(agent, engine) == []

    run(scenario())
