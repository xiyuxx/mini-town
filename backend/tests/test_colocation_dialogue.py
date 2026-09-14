import asyncio

from backend.town.engine import SimulationEngine


def test_idle_agents_in_same_location_can_start_dialogue_without_same_cell(tmp_path):
    async def scenario():
        engine = SimulationEngine()
        engine.llm.fallback = True
        db = str(tmp_path / "town.db")
        engine.memory.db_path = db
        engine.relationship_store.db_path = db
        engine.trace.db_path = db
        await engine.init()
        first, second = engine.agents[:2]
        engine.agents = [first, second]
        first.state.current_location = second.state.current_location = "park"
        first.state.x, first.state.y = 9, 8
        second.state.x, second.state.y = 12, 9
        await engine._update_colocation(engine.get_sim_time_str())
        engine._queue_colocated_dialogues()
        events = await engine._run_group_dialogues(engine.get_sim_time_str())
        assert any(event["type"] == "dialogue_start" for event in events)
        relationship = await engine.relationship_store.get(first.id, second.id)
        assert relationship.familiarity > 0

    asyncio.run(scenario())
