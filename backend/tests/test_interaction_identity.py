import asyncio

from backend.town.engine import SimulationEngine


def test_dialogue_lifecycle_uses_dialogue_id_as_interaction_id(tmp_path):
    async def scenario():
        engine = SimulationEngine()
        engine.llm.fallback = True
        engine.memory.db_path = str(tmp_path / "dialogue-interaction" / "town.db")
        await engine.init()
        first, second = engine.agents[:2]
        second.state.current_location = first.state.current_location
        second.state.x, second.state.y = first.state.x, first.state.y
        await engine._execute_decision(first, {
            "action": "talk", "target": second.id, "content": "你好",
        }, engine.get_sim_time_str())
        events = await engine._run_group_dialogues(engine.get_sim_time_str())
        start = next(event for event in events if event["type"] == "dialogue_start")
        interaction = await engine.interaction_store.get(start["interactionId"])
        assert interaction is not None
        assert interaction.id == start["dialogueId"]
        assert interaction.status == "executing"

        for _ in range(20):
            if not engine.dialogue._active_dialogues:
                break
            await engine._advance_dialogues(engine.get_sim_time_str())
        interaction = await engine.interaction_store.get(start["dialogueId"])
        assert interaction is not None
        assert interaction.status == "completed"
        assert interaction.fact_ids
        relationship = await engine.relationship_store.get(first.id, second.id)
        assert relationship.last_interaction_id == start["dialogueId"]
        await engine.shutdown()
    asyncio.run(scenario())


def test_action_fact_and_event_share_interaction_id(tmp_path):
    async def scenario():
        engine = SimulationEngine()
        engine.memory.db_path = str(tmp_path / "interaction" / "town.db")
        await engine.init()
        agent = engine.agents[0]
        events = await engine._execute_decision(agent, {
            "action": "act", "content": "整理房间", "duration_minutes": 5,
            "source": "test",
        }, engine.get_sim_time_str())
        assert events and events[0]["interactionId"]
        event_id = events[0]["interactionId"]
        fact = engine.fact_ledger.get(events[0]["factId"])
        assert fact is not None
        assert fact.interaction_id == event_id
        await engine.shutdown()
    asyncio.run(scenario())
