"""Social consequences: interacting must change later behaviour.

Covers the three loops that were missing: a conversation satisfies the social
need, what gets said becomes knowledge the listener can pass on, and stored
relationships change whom an agent looks for.
"""

import asyncio
import json

from backend.config import config
from backend.town.candidates import build_routine_candidates
from backend.town.considerations import collect_active_considerations
from backend.town.engine import SimulationEngine
from backend.town.relationship import Relationship
from backend.town.tools import ToolCall


def run(coro):
    return asyncio.run(coro)


async def make_engine(tmp_path):
    engine = SimulationEngine()
    engine.llm.fallback = True
    engine.embedding_provider.fallback = True
    db_path = str(tmp_path / "town.db")
    engine.memory.db_path = db_path
    engine.relationship_store.db_path = db_path
    engine.trace.db_path = db_path
    await engine.init()
    return engine


async def settle(engine):
    for _ in range(6):
        await asyncio.sleep(0)
    pending = [task for _, task in engine._dialogue_turn_tasks.values() if not task.done()]
    pending += [task for task in engine._background_tasks if not task.done()]
    if pending:
        await asyncio.wait(pending, timeout=5)


async def park_pair(engine):
    """Two agents in the park, idle, on separate cells, kept in place."""
    chosen = engine.agents[:2]
    engine.agents = list(chosen)
    engine.routine_planner.next_action = lambda _agent, _engine: None
    for index, agent in enumerate(chosen):
        agent.state.current_location = "park"
        agent.state.x, agent.state.y = 9 + index, 8
        agent.state.status = "IDLE"
    engine._realtime_llm = True
    return chosen


def test_conversation_satisfies_the_social_need(tmp_path):
    """Talking restores the social need; before, only decay could move it."""
    async def scenario():
        engine = await make_engine(tmp_path)
        first, second = await park_pair(engine)
        before = {agent.id: agent.state.needs["social"] for agent in (first, second)}
        session_id = ""
        settled_after_end = 0
        for _ in range(30):
            await engine.tick()
            await settle(engine)
            if engine.dialogue._active_dialogues:
                session_id = next(iter(engine.dialogue._active_dialogues.values())).id
                settled_after_end = 0
            elif session_id:
                # The engine commits the closing turn on the next tick, which is
                # also where the conversation meets the social need.
                settled_after_end += 1
                if settled_after_end >= 2:
                    break

        assert session_id
        for agent in (first, second):
            assert agent.state.needs["social"] > before[agent.id] + 10

    run(scenario())


def test_a_told_fact_becomes_known_to_the_listener_and_survives_restart(tmp_path):
    """Saying something out loud is how the listener comes to know it."""
    async def scenario():
        engine = await make_engine(tmp_path)
        first, second = await park_pair(engine)
        engine.llm.fallback = False
        fact = await engine._record_fact(
            first, "rumour", "park", {"note": "今晚广场有集市"},
        )
        assert fact.known_by == [first.id]

        async def fake_loop(_system, _user, _definitions, registry, max_calls=3):
            await registry.execute(ToolCall(
                id="call-1", name="dialogue_reply",
                arguments={
                    "content": "听说今晚广场有集市。", "action": "end", "emoji": "🙂",
                    "mental_update": {}, "referenced_fact_ids": [fact.id],
                },
            ))
            return {"content": "听说今晚广场有集市。", "action": "end"}

        async def fake_support(_reply, _facts):
            return {"supported": True}

        async def fake_evaluation(_text, _names):
            return {"summary": "聊了集市的事", "valence": 0.2,
                    "affinity_delta": 0.1, "trust_delta": 0.1}

        async def fake_encoding(*_args, **_kwargs):
            return "听说明晚广场有集市"

        engine.llm.function_call_loop = fake_loop
        engine.llm.validate_claim_support = fake_support
        engine.llm.evaluate_dialogue = fake_evaluation
        engine.llm.encode_dialogue_memory = fake_encoding

        sim_time = engine.get_sim_time_str()
        await engine._update_colocation(sim_time)
        await engine._queue_colocated_dialogues()
        await engine._run_group_dialogues(sim_time)
        await engine._advance_dialogues(sim_time)

        assert second.id in engine.fact_ledger.get(fact.id).known_by

        # Knowledge is durable: a fresh ledger warmed from storage still has it.
        restored = await engine.fact_store.load_recent()
        stored = next(item for item in restored if item.id == fact.id)
        assert second.id in stored.known_by
        assert json.loads(json.dumps(stored.details)) == {"note": "今晚广场有集市"}

    run(scenario())


def test_social_motive_names_the_agent_worth_seeking(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        first, second = engine.agents[:2]
        first.state.needs["social"] = 10
        await engine.relationship_store.upsert(Relationship(
            agent_a=first.id, agent_b=second.id, familiarity=4.0, affinity=2.0,
        ))

        social = next(
            item for item in collect_active_considerations(first, engine)
            if item.source_id == f"need:{first.id}:social"
        )

        assert social.details["target_agent_id"] == second.id
        assert second.name in social.description

    run(scenario())


def test_candidates_prefer_a_place_with_a_familiar_face(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        friend = engine.agents[1]
        chosen = None
        for _ in range(60):
            await engine.tick()
            for agent in engine.agents:
                if build_routine_candidates(agent, engine):
                    chosen = agent
                    break
            if chosen:
                break
        assert chosen is not None, "no flexible routine candidates in the first sim hours"

        before = build_routine_candidates(chosen, engine)
        target_location = before[0].action["location"]
        friend.state.current_location = target_location
        await engine.relationship_store.upsert(Relationship(
            agent_a=chosen.id, agent_b=friend.id, familiarity=6.0, affinity=3.0,
        ))
        after = build_routine_candidates(chosen, engine)
        same = next(item for item in after if item.id == before[0].id)

        assert same.score > before[0].score
        assert any("熟人" in reason for reason in same.reasons)

    run(scenario())
