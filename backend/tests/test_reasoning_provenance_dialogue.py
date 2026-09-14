import asyncio

from backend.town.cognition import FactEvent, Intention
from backend.town.engine import SimulationEngine


def make_engine(tmp_path):
    async def create():
        engine = SimulationEngine()
        engine.llm.fallback = True
        db = str(tmp_path / "town.db")
        engine.memory.db_path = db
        engine.relationship_store.db_path = db
        engine.trace.db_path = db
        await engine.init()
        return engine

    return create


def test_belief_updates_require_visible_fact_provenance(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)()
        agent = engine.agents[0]
        visible = engine.fact_ledger.add(FactEvent(
            sim_time=engine.get_sim_time_str(), sim_timestamp=engine.get_sim_timestamp(),
            type="observation", agent_id=agent.id, location=agent.state.current_location,
            details={"content": "门已开"}, known_by=[agent.id],
        ))
        await engine._execute_decision(agent, {
            "action": "noop",
            "mental_update": {"belief_updates": [
                {"proposition": "门已开", "status": "observed", "confidence": 1.0,
                 "source_fact_ids": [visible.id]},
                {"proposition": "里面有人", "status": "observed", "confidence": 1.0,
                 "source_fact_ids": ["fact_not_visible"]},
            ]},
        }, engine.get_sim_time_str())
        beliefs = {belief.proposition: belief for belief in agent.mental_state.beliefs}
        assert beliefs["门已开"].status == "observed"
        assert beliefs["门已开"].source_fact_ids == [visible.id]
        assert beliefs["里面有人"].status == "uncertain"
        assert beliefs["里面有人"].confidence <= 0.35
        assert beliefs["里面有人"].source_fact_ids == []

    asyncio.run(scenario())


def test_dialogue_topic_ranks_shared_memory_and_relationship_context(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)()
        first, second = engine.agents[:2]
        shared = await engine.memory.add(
            first.id, "park", "我们约好下周一起修理长椅。", "dialogue", importance=9,
            sim_time=engine.get_sim_time_str(), participants=[second.id],
            unresolved=True, formation_score=0.9,
        )
        intention = first.mental_state.add_intention(Intention(
            description="问问修理长椅的计划有没有进展", source="memory",
            created_at=engine.get_sim_timestamp(), target_location="park",
            source_ids=[shared.id],
        ))
        intents = [
            {"initiator": first, "target_id": second.id, "location": "park",
             "content": "问问修理长椅的计划有没有进展", "memory_intention_id": intention.id},
            {"initiator": first, "target_id": second.id, "location": "park",
             "content": "在公园碰见了，打个招呼。"},
        ]
        selected = await engine._rank_dialogue_topics(first, second, intents)
        assert selected is intents[0]
        assert selected["topic"]["sharedMemoryIds"] == [shared.id]
        assert selected["topic"]["relevanceScore"] > 0

    asyncio.run(scenario())


def test_memory_intention_does_not_trigger_remote_dialogue(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)()
        first, second = engine.agents[:2]
        first.state.current_location = "park"
        second.state.current_location = "cafe"
        first.state.status = second.state.status = "IDLE"
        first.mental_state.add_intention(Intention(
            description="问问对方计划有没有进展", source="memory",
            created_at=engine.get_sim_timestamp(), target_location="park",
        ))
        engine._queue_encounter_dialogue(first, second)
        assert await engine._run_group_dialogues(engine.get_sim_time_str()) == []
        assert engine.dialogue.participant_ids() == set()

    asyncio.run(scenario())
