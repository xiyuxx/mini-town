import asyncio

from backend.town.daily_plan import DailyPlan, DailyPlanBlock
from backend.town.engine import SimulationEngine
from backend.town.schedule import TimeWindow


def run(coro):
    return asyncio.run(coro)


def test_realtime_engine_creates_and_reuses_one_daily_plan(tmp_path):
    async def scenario():
        engine = SimulationEngine()
        engine.llm.fallback = False
        db_path = str(tmp_path / "town.db")
        engine.memory.db_path = db_path
        engine.relationship_store.db_path = db_path
        engine.trace.db_path = db_path
        await engine.init()
        agent = engine.agents[0]
        engine.agents = [agent]
        engine._realtime_llm = True
        calls = []

        async def create_daily_plan(planned_agent, planned_engine, _context, replan_reason=""):
            calls.append((planned_agent.id, planned_engine.day, replan_reason))
            return DailyPlan(
                agent_id=planned_agent.id,
                day=planned_engine.day,
                focus="保留生活节奏",
                blocks=[DailyPlanBlock(
                    id="morning", window=TimeWindow(360, 720), intent="处理晨间安排",
                )],
                created_at=planned_engine.get_sim_timestamp(),
                review_at=planned_engine.get_sim_timestamp() + 180,
            )

        engine.planner.create_daily_plan = create_daily_plan
        await engine.tick()
        task = engine._daily_plan_tasks[agent.id][1]
        await task
        await engine.tick()

        assert agent.daily_plan is not None
        assert agent.daily_plan.focus == "保留生活节奏"
        assert calls == [(agent.id, 1, "")]
        assert engine.get_state()["dailyPlans"][0]["agentId"] == agent.id

        await engine.tick()
        assert calls == [(agent.id, 1, "")]
        await engine.shutdown()

    run(scenario())
