import asyncio
import json

from backend.town.cognition import Goal, Intention, MentalState
from backend.town.engine import SimulationEngine


def run(coro):
    return asyncio.run(coro)


def test_engines_do_not_share_persistent_state(tmp_path):
    async def scenario():
        first_db = tmp_path / "first" / "town.db"
        second_db = tmp_path / "second" / "town.db"

        first = SimulationEngine()
        first.memory.db_path = str(first_db)
        await first.init()
        await first.memory.add(
            "wang", "park", "第一轮测试留下的记忆", "observation", 8,
            sim_time="第1天 06:00",
        )
        await first.relationship_store.record_interaction(
            "wang", "mei", "第一轮测试互动", sim_time="第1天 06:05",
        )
        await first.trace.log("第1天 06:05", "wang", "test", "first_run")

        second = SimulationEngine()
        second.memory.db_path = str(second_db)
        await second.init()

        assert await second.memory.get_all_for_agent("wang") == []
        relationship = await second.relationship_store.get("wang", "mei")
        assert relationship.interaction_count == 0
        assert await second.trace.get_recent() == []
        assert first.memory.db_path != second.memory.db_path

        await first.shutdown()
        await second.shutdown()

    run(scenario())


def test_reset_clears_persistent_and_runtime_state(tmp_path):
    async def scenario():
        engine = SimulationEngine()
        engine.memory.db_path = str(tmp_path / "reset" / "town.db")
        await engine.init()
        agent = engine.agents[0]
        initial_position = (agent.state.x, agent.state.y, agent.state.current_location)
        resource = next(iter(engine.resources.resources.values()))
        initial_quantity = resource.quantity

        engine._encountered_pairs.add(("wang", "mei"))
        engine._active_encounter_pairs.add(("wang", "mei"))
        engine._active_colocation_groups["park"] = ("wang", "mei")
        engine._last_events = [{"type": "test"}]
        resource.quantity = 0
        agent.state.x += 1
        agent.state.current_location = "in_transit"
        engine.habits.record(
            agent.id,
            {
                "source": "routine_block",
                "routine_block_id": "test",
                "location": "park",
                "content": "散步",
            },
            engine.get_sim_timestamp(),
            success=True,
        )
        await engine.memory.add(
            agent.id, "park", "需要在重置时清除的记忆", "observation", 8,
            sim_time=engine.get_sim_time_str(),
        )
        await engine.relationship_store.record_interaction(
            "wang", "mei", "需要在重置时清除的关系", sim_time=engine.get_sim_time_str(),
        )
        await engine.trace.log(engine.get_sim_time_str(), agent.id, "test", "reset_state")

        await engine.reset()

        assert engine._encountered_pairs == set()
        assert engine._active_encounter_pairs == set()
        assert engine._active_colocation_groups == {}
        assert engine._last_events == []
        assert engine.habits.for_agent(agent.id) == []
        assert await engine.memory.get_all_for_agent(agent.id) == []
        assert (await engine.relationship_store.get("wang", "mei")).interaction_count == 0
        assert await engine.trace.get_recent() == []
        reset_agent = engine.agents[0]
        assert (reset_agent.state.x, reset_agent.state.y, reset_agent.state.current_location) == initial_position
        assert next(iter(engine.resources.resources.values())).quantity == initial_quantity

        await engine.shutdown()

    run(scenario())


def test_mental_state_reads_do_not_expire_intentions(tmp_path):
    state = MentalState()
    intention = state.add_intention(Intention(
        description="去公园散步", source="test", created_at=0,
        target_location="park", interaction_type="wait", expected_until=10,
    ))

    assert state.open_intentions(11) == []
    assert intention.status == "open"
    assert state.expire_intentions(11) == [intention.id]
    assert intention.status == "expired"


def test_task_mental_summary_is_bounded_but_keeps_current_plan_data():
    state = MentalState()
    for index in range(12):
        state.add_goal(Goal(
            description=f"目标{index}", source="test", created_at=0,
            importance=0.5, status="proposed",
        ))
    state.add_intention(Intention(
        description="完成当前安排", source="test", created_at=0,
        target_location="park", interaction_type="wait",
    ))

    full = state.context_dict(0)
    action = state.summary_for("action_decision", 0)
    planning = state.summary_for("daily_plan", 0)

    assert action["active_goal"]
    assert action["intentions"][0]["description"] == "完成当前安排"
    assert len(action["goals"]) <= 5
    assert len(json.dumps(action, ensure_ascii=False)) < len(json.dumps(full, ensure_ascii=False))
    assert len(planning["goals"]) <= 8


def test_relevant_intention_prefers_matching_location_and_action():
    state = MentalState()
    park = state.add_intention(Intention(
        description="在公园散步", source="test", created_at=0,
        target_location="park", interaction_type="wait",
    ))
    cafe = state.add_intention(Intention(
        description="在咖啡馆休息", source="test", created_at=0,
        target_location="cafe", interaction_type="wait",
    ))

    selected = state.relevant_intention({
        "interaction_type": "wait", "location": "cafe",
    })

    assert selected is cafe
    assert selected is not park


def test_engine_binds_all_sqlite_stores_to_memory_database(tmp_path):
    async def scenario():
        engine = SimulationEngine()
        db_path = tmp_path / "shared" / "town.db"
        engine.memory.db_path = str(db_path)
        await engine.init()

        assert engine.relationship_store.db_path == str(db_path)
        assert engine.trace.db_path == str(db_path)
        assert engine.dialogue_store.db_path == str(db_path)

        await engine.shutdown()

    run(scenario())
