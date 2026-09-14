import asyncio

from backend.town.cognition import Belief, Intention
from backend.town.engine import SimulationEngine


def test_behavior_relevant_cognition_survives_engine_restart(tmp_path):
    async def scenario():
        db_path = str(tmp_path / "persistent" / "town.db")
        first = SimulationEngine()
        first.memory.db_path = db_path
        await first.init()
        agent = first.agents[0]
        agent.mental_state.add_intention(Intention(
            description="之后去公园", source="test", created_at=first.get_sim_timestamp(),
            target_location="park", interaction_type="wait",
        ))
        agent.mental_state.add_belief(Belief(
            proposition="公园适合散步", confidence=0.8, status="inferred",
            learned_at=first.get_sim_timestamp(), last_confirmed_at=first.get_sim_timestamp(),
        ))
        first.habits.record(
            agent.id,
            {"source": "routine_block", "routine_block_id": "r", "location": "park", "content": "散步"},
            first.get_sim_timestamp(), True,
        )
        await first._persist_cognition()
        await first.shutdown()

        second = SimulationEngine()
        second.memory.db_path = db_path
        await second.init()
        restored = second.agents[0]
        assert any(item.description == "之后去公园" for item in restored.mental_state.intentions)
        assert any(item.proposition == "公园适合散步" for item in restored.mental_state.beliefs)
        assert second.habits.for_agent(restored.id)[0].activity == "散步"

        await second.reset()
        third = SimulationEngine()
        third.memory.db_path = db_path
        await third.init()
        assert third.agents[0].mental_state.intentions == []
        assert third.agents[0].mental_state.beliefs == []
        assert third.habits.for_agent(third.agents[0].id) == []
        await second.shutdown()
        await third.shutdown()

    asyncio.run(scenario())
