import asyncio

from backend.town.cognition import Intention
from backend.town.engine import SimulationEngine


def test_memory_intention_becomes_dialogue_opening_and_event_trigger(tmp_path):
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
        first.state.status = second.state.status = "IDLE"
        first.mental_state.add_intention(Intention(
            description="问问对方上次提到的计划有没有进展", source="memory",
            created_at=engine.get_sim_timestamp(), target_location="park",
        ))
        engine._queue_encounter_dialogue(first, second)
        events = await engine._run_group_dialogues(engine.get_sim_time_str())
        line = next(event for event in events if event["type"] == "dialogue_line")
        assert "问问对方" in line["content"]
        assert line["trigger"]["type"] == "memory_intention"

    asyncio.run(scenario())
