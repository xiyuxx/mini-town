import asyncio
import json

from backend.town.dialogue import DialogueSession
from backend.town.engine import SimulationEngine


def run(coro):
    return asyncio.run(coro)


def test_planning_context_uses_task_summary_and_bounded_locations(tmp_path):
    async def scenario():
        engine = SimulationEngine()
        engine.memory.db_path = str(tmp_path / "context" / "town.db")
        await engine.init()
        agent = engine.agents[0]
        full_mental_state = agent.mental_state.context_dict(engine.get_sim_timestamp())
        scoped = await agent.build_planning_context(
            engine.get_sim_time_str(), engine, task_type="life_plan",
        )
        assert scoped["request"]["agent_id"] == agent.id
        assert scoped["request"]["task_type"] == "life_plan"
        assert scoped["request"]["state_version"] == engine.state_version
        assert len(scoped["known_locations"]) <= 6
        assert len(scoped["mental_state"]["goals"]) <= 6
        assert "current_episode" not in scoped["mental_state"]
        assert len(json.dumps(scoped["mental_state"], ensure_ascii=False)) < len(json.dumps(full_mental_state, ensure_ascii=False))
        assert scoped["active_considerations"]
        metric = engine.context.recent_metrics(1)[0]
        assert metric["agentId"] == agent.id
        assert metric["taskType"] == "life_plan"
        assert metric["inputChars"] > 0
        assert "known_locations" not in metric["omittedSections"]
        await engine.shutdown()
    run(scenario())


def test_dialogue_context_uses_compact_mental_state_and_pair_relationship(tmp_path):
    async def scenario():
        engine = SimulationEngine()
        engine.memory.db_path = str(tmp_path / "dialogue-context" / "town.db")
        await engine.init()
        engine.llm.fallback = False
        speaker, other = engine.agents[:2]
        other.state.current_location = speaker.state.current_location
        captured = {}
        async def fake_function_call_loop(system, user, tools, registry, max_calls=5):
            captured["user"] = user
            return {"content": "你好", "action": "end", "emoji": "🙂"}
        engine.llm.function_call_loop = fake_function_call_loop
        session = DialogueSession("dlg_test", [speaker, other], [speaker, other], speaker.state.current_location)
        reply = await engine.dialogue._generate_reply(session, speaker, [other], engine.trace, engine.get_sim_time_str())
        assert reply[0] == "你好"
        assert "current_episode" not in captured["user"]
        assert "与参与者的关系" in captured["user"]
        assert other.name in captured["user"]
        await engine.shutdown()
    run(scenario())


def test_intention_context_excludes_location_catalog_and_proposal_is_location_free(tmp_path):
    async def scenario():
        engine = SimulationEngine()
        engine.memory.db_path = str(tmp_path / "intention" / "town.db")
        await engine.init()
        agent = engine.agents[0]
        context = await agent.build_planning_context(engine.get_sim_time_str(), engine, task_type="intention_proposal")
        assert "known_locations" not in context
        async def fake_json_call(system, payload, mode="chat", task="json_call"):
            assert task == "intention_proposal"
            return {"purpose": "去一个适合散步的地方", "interaction_type": "wait"}
        engine.planner._json_call = fake_json_call
        proposal = await engine.planner.propose_intention(agent, engine, context)
        assert proposal["purpose"] == "去一个适合散步的地方"
        assert "location" not in proposal
        await engine.shutdown()
    run(scenario())


def test_decide_action_queries_locations_after_intention(tmp_path):
    async def scenario():
        engine = SimulationEngine()
        engine.memory.db_path = str(tmp_path / "intent-first" / "town.db")
        await engine.init()
        engine.llm.fallback = False
        agent = engine.agents[0]
        calls = []
        async def propose(current_agent, current_engine, context):
            calls.append(("intention", "known_locations" in context))
            return {"purpose": "去公园散步", "interaction_type": "wait", "target_location": "park"}
        async def decide(current_agent, current_engine, context):
            calls.append(("action", "known_locations" in context))
            return {"action": "act", "content": "散步", "location": "park"}
        engine.planner.propose_intention = propose
        engine.planner.decide = decide
        result = await agent.decide_action(engine.get_sim_time_str(), engine)
        assert result["location"] == "park"
        assert calls == [("intention", False), ("action", True)]
        await engine.shutdown()
    run(scenario())


def test_context_task_views_do_not_share_unneeded_fields(tmp_path):
    async def scenario():
        engine = SimulationEngine()
        engine.memory.db_path = str(tmp_path / "task-views" / "town.db")
        await engine.init()
        agent = engine.agents[0]
        routine = await agent.build_planning_context(engine.get_sim_time_str(), engine, task_type="routine_selection")
        reflection = await agent.build_planning_context(engine.get_sim_time_str(), engine, task_type="reflection")
        assert "known_locations" not in routine
        assert "inventory" not in routine
        assert "known_locations" not in reflection
        assert "location" not in reflection
        assert "mental_state" in routine
        assert "mental_state" in reflection
        await engine.shutdown()
    run(scenario())


def test_life_plan_resolves_intention_before_building_location_context(tmp_path):
    async def scenario():
        engine = SimulationEngine()
        engine.memory.db_path = str(tmp_path / "life-plan-order" / "town.db")
        await engine.init()
        engine.llm.fallback = False
        agent = engine.agents[0]
        calls = []
        async def fake_propose(current_agent, current_engine, context):
            calls.append(("propose", "known_locations" in context))
            return {"purpose": "找地方散步", "interaction_type": "wait"}
        async def fake_build(current_agent, task_type, current_engine, intention=None, sim_time_str=None):
            calls.append(("build", task_type, intention["purpose"]))
            return {"sim_time": current_engine.get_sim_time_str(), "known_locations": [{"id": "park"}]}
        async def fake_json_call(system, payload, mode="chat", task="json_call"):
            calls.append(("plan", task, payload.get("selected_intention", {}).get("purpose")))
            return {"focus": "散步", "motive": "放松", "review_after_minutes": 60, "steps": [{"description": "去公园散步", "action": {"interaction_type": "wait", "location": "park"}}]}
        engine.planner.propose_intention = fake_propose
        engine.context.build = fake_build
        engine.planner._json_call = fake_json_call
        plan = await engine.planner.create_life_plan(agent, engine, {"sim_time": engine.get_sim_time_str()})
        assert plan.current_step.action["location"] == "park"
        assert calls == [("propose", False), ("build", "life_plan", "找地方散步"), ("plan", "life_plan", "找地方散步")]
        await engine.shutdown()
    run(scenario())
