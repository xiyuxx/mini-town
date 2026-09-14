import asyncio

from backend.town.engine import SimulationEngine


def test_multi_day_flexible_routine_forms_bounded_habits(tmp_path):
    async def scenario():
        engine = SimulationEngine()
        engine.llm.fallback = True
        db = str(tmp_path / "town.db")
        engine.memory.db_path = db
        engine.relationship_store.db_path = db
        engine.trace.db_path = db
        await engine.init()
        mei = next(agent for agent in engine.agents if agent.id == "mei")
        engine.agents = [mei]
        engine._realtime_llm = True

        for _ in range(600):
            await engine.tick()

        habits = engine.habits.for_agent(mei.id)
        assert engine.day >= 3
        assert len(habits) <= engine.habits.max_per_agent
        assert all(habit.voluntary_count > 0 for habit in habits)
        assert all(0.0 <= habit.strength <= 1.0 for habit in habits)
        await engine.shutdown()

    asyncio.run(scenario())
