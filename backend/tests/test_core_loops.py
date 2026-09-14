import asyncio
import random

from backend.town.embedding import EmbeddingProvider
from backend.town.engine import SimulationEngine
from backend.town.memory import MemoryStore


def run(coro):
    return asyncio.run(coro)


def test_sim_time_memory_retrieval_and_decay(tmp_path):
    async def scenario():
        store = MemoryStore()
        store.db_path = str(tmp_path / "town.db")
        store.set_embedding_provider(EmbeddingProvider())
        await store.init_db()
        old = await store.add("a", "park", "很久以前的一次散步", "observation", 8,
                              sim_time="第1天 06:00")
        recent = await store.add("a", "park", "今天在公园遇到了朋友", "dialogue", 9,
                                 sim_time="第3天 06:00")
        memories = await store.get_important(
            "a", hours=24, current_sim_timestamp=2 * 1440 + 6 * 60
        )
        assert [memory.id for memory in memories] == [recent.id]
        await store.apply_decay("a", 5 * 1440 + 6 * 60)
        all_memories = await store.get_all_for_agent("a")
        decayed = next(memory for memory in all_memories if memory.id == old.id)
        assert decayed.importance < 8

    run(scenario())


def test_embedding_single_batch_and_semantic_search(tmp_path):
    async def scenario():
        provider = EmbeddingProvider()
        one = await provider.embed_one("公园下棋")
        many = await provider.embed_many(["公园下棋", "诊所看病"])
        assert provider.cosine_similarity(one, many[0]) > provider.cosine_similarity(one, many[1])

        store = MemoryStore()
        store.db_path = str(tmp_path / "town.db")
        store.set_embedding_provider(provider)
        await store.init_db()
        await store.add("a", "park", "我在公园和老王下棋", "dialogue", 8,
                        sim_time="第1天 08:00")
        result = await store.semantic_search(
            "a", "公园下棋", provider, current_sim_timestamp=9 * 60
        )
        assert result and "下棋" in result[0].content

    run(scenario())


def test_talk_intent_runs_dialogue_and_updates_relationship(tmp_path):
    async def scenario():
        random.seed(7)
        engine = SimulationEngine()
        engine.llm.fallback = True
        engine.memory.db_path = str(tmp_path / "town.db")
        engine.relationship_store.db_path = str(tmp_path / "town.db")
        engine.trace.db_path = str(tmp_path / "town.db")
        await engine.init()
        first, second = engine.agents[:2]
        second.state.current_location = first.state.current_location
        second.state.x, second.state.y = first.state.x, first.state.y
        await engine._execute_decision(first, {
            "action": "talk", "target": second.id,
            "content": "早啊，今天一起去公园吗？",
        }, engine.get_sim_time_str())
        assert len(engine._dialogue_intents) == 1
        events = await engine._run_group_dialogues(engine.get_sim_time_str())
        assert "dialogue_line" in [event["type"] for event in events]
        assert await engine.memory.get_recent(second.id, limit=10) == []
        assert engine.dialogue.participant_ids() == {first.id, second.id}

        for _ in range(20):
            if not engine.dialogue._active_dialogues:
                break
            await engine._advance_dialogues(engine.get_sim_time_str())

        memories = await engine.memory.get_recent(second.id, limit=10)
        assert any(memory.type == "dialogue" and "公园" in memory.content for memory in memories)
        relationship = await engine.relationship_store.get(first.id, second.id)
        assert relationship.interaction_count == 1
        stored = await engine.dialogue_store.list_recent()
        assert stored and stored[0]["status"] == "completed"
        detail = await engine.dialogue_store.get(stored[0]["id"])
        assert detail and len(detail["messages"]) >= 2
        assert second.id in first.mental_state.recent_interactions
        assert first.id in second.mental_state.recent_interactions
        recent = await engine.dialogue_store.get_recent_between(first.id, second.id)
        assert recent and recent["id"] == detail["id"]

    run(scenario())


def test_relationship_values_are_rounded_on_storage_and_serialization(tmp_path):
    async def scenario():
        engine = SimulationEngine()
        engine.llm.fallback = True
        db_path = str(tmp_path / "town.db")
        engine.memory.db_path = db_path
        engine.relationship_store.db_path = db_path
        engine.trace.db_path = db_path
        await engine.init()
        await engine.relationship_store.record_interaction(
            "mei", "hua", "一起讨论画稿", valence=0.8000000000000002,
            affinity_delta=0.6400000000000001,
            trust_delta=0.16000000000000003,
            sim_time="第1天 09:50",
        )
        relationship = await engine.relationship_store.get("mei", "hua")
        data = relationship.to_dict()
        assert relationship.affinity == 0.64
        assert data["affinity"] == 0.6
        assert data["anchors"][0]["valence"] == 0.8

    run(scenario())


def test_action_mental_update_persists_into_next_decision_context(tmp_path):
    async def scenario():
        engine = SimulationEngine()
        engine.llm.fallback = True
        db_path = str(tmp_path / "town.db")
        engine.memory.db_path = db_path
        engine.relationship_store.db_path = db_path
        engine.trace.db_path = db_path
        await engine.init()
        doctor = next(agent for agent in engine.agents if agent.id == "li")
        await engine._execute_decision(doctor, {
            "action": "act", "content": "继续晨跑两圈", "duration_minutes": 10,
            "mental_update": {
                "active_goal": "完成晨跑",
                "attention": ["呼吸", "剩余两圈"],
                "intentions": [{
                    "description": "继续晨跑两圈", "source": "spoken",
                    "target_location": "park", "public": True,
                }],
            },
        }, engine.get_sim_time_str())
        context = doctor.mental_state.context_dict(engine.get_sim_timestamp())
        assert context["active_goal"] == "完成晨跑"
        assert context["intentions"][0]["description"] == "继续晨跑两圈"
        assert doctor.state.current_action == "继续晨跑两圈"

    run(scenario())


def test_reflection_keeps_source_memory_ids(tmp_path):
    async def scenario():
        engine = SimulationEngine()
        engine.llm.fallback = True
        engine.memory.db_path = str(tmp_path / "town.db")
        engine.relationship_store.db_path = str(tmp_path / "town.db")
        engine.trace.db_path = str(tmp_path / "town.db")
        await engine.init()
        agent = engine.agents[0]
        for hour in range(6, 10):
            await engine.memory.add(
                agent.id, "park", f"重要经历{hour}", "observation", 7,
                sim_time=f"第1天 {hour:02d}:00",
            )
        engine._last_reflection_timestamp = 5 * 60
        engine.day, engine.hour, engine.minute = 1, 10, 0

        async def fake_reflect(persona, memories):
            return ["我应该更认真维护重要关系。"]

        engine.llm.reflect = fake_reflect
        await engine._run_reflections(engine.get_sim_time_str())
        memories = await engine.memory.get_recent(agent.id, limit=20)
        reflection = next(memory for memory in memories if memory.type == "reflection")
        assert len(reflection.source_ids) == 4

    run(scenario())
