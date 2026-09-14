"""Seeing the room you are in is bookkeeping; looking closely is a decision."""

import asyncio

from backend.town.engine import SimulationEngine


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


def place(engine, agent_id, location):
    agent = next(item for item in engine.agents if item.id == agent_id)
    agent.state.current_location = location
    agent.state.status = "IDLE"
    return agent


def test_walking_in_is_seeing_the_room(tmp_path):
    """No action needed: the agent that is somewhere can see what is there."""
    async def scenario():
        engine = await make_engine(tmp_path)
        agent = place(engine, "wang", "cafe")
        agent._last_observation = None

        await engine.tick()

        record = agent._last_observation
        assert record is not None, "standing somewhere must be enough to have seen it"
        assert record["location"] == "cafe"
        assert record["signature"]

    run(scenario())


def test_the_view_follows_the_agent(tmp_path):
    """An observation of where it used to be is worse than none."""
    async def scenario():
        engine = await make_engine(tmp_path)
        agent = place(engine, "wang", "cafe")
        await engine.tick()
        assert agent._last_observation["location"] == "cafe"

        agent.state.current_location = "park"
        await engine.tick()

        assert agent._last_observation["location"] == "park"

    run(scenario())


def test_standing_still_does_not_make_the_room_go_stale(tmp_path):
    """Half an hour behind the same counter is still knowing who is behind it."""
    async def scenario():
        engine = await make_engine(tmp_path)
        agent = place(engine, "wang", "cafe")
        await engine.tick()
        stale = engine.get_sim_timestamp() - 40 * 60
        agent._last_observation["observed_at"] = stale

        await engine.tick()

        assert agent._last_observation["observed_at"] > stale

    run(scenario())


def test_a_view_that_did_not_move_is_not_rewritten(tmp_path):
    """Change detection must not decay into writing the same thing every tick."""
    async def scenario():
        engine = await make_engine(tmp_path)
        agent = place(engine, "wang", "cafe")

        engine._refresh_observations()      # nothing changed since the last look
        settled = agent._last_observation["observed_at"]
        engine._refresh_observations()

        assert agent._last_observation["observed_at"] == settled

    run(scenario())


def test_a_change_in_the_room_is_seen_at_once(tmp_path):
    """Somebody walking in is a change, and it must not wait for the clock."""
    async def scenario():
        engine = await make_engine(tmp_path)
        agent = place(engine, "wang", "cafe")
        newcomer = place(engine, "mei", "park")
        await engine.tick()
        before = agent._last_observation["signature"]

        newcomer.state.current_location = "cafe"
        await engine.tick()

        assert agent._last_observation["signature"] != before
        assert [item["id"] for item in agent._last_observation["nearby_agents"]] == ["mei"]

    run(scenario())


def test_an_agent_that_walked_in_can_order_from_the_counter(tmp_path):
    """The failure this replaces: refused at the counter for not having looked."""
    async def scenario():
        engine = await make_engine(tmp_path)
        item = next(
            resource for resource in engine.resources.at_location("cafe")
            if engine.interactions._service_facility_id(resource)
        )
        facility_id = engine.interactions._service_facility_id(item)
        staff_duty = next(duty for duty in engine.resources.scene.duties if duty.facility_id == facility_id)
        buyer = place(engine, "wang", "cafe")
        place(engine, staff_duty.agent_id, "cafe")

        await engine.tick()

        assert engine.interactions.service_needs_a_look(buyer, "cafe", item, engine) is False

    run(scenario())


def test_a_talking_agent_is_told_what_is_around_it(tmp_path):
    """The dialogue layer reads the same record, not a second description."""
    async def scenario():
        engine = await make_engine(tmp_path)
        speaker = place(engine, "wang", "cafe")
        listener = place(engine, "mei", "cafe")
        await engine.tick()
        entity = next(
            item for item in engine.resources.at_location("cafe") if item.name
        )

        captured = {}

        async def fake_loop(system, user, tools, registry, **kwargs):
            captured["prompt"] = user
            return {"content": "嗯。", "action": "end"}

        engine.llm.fallback = False
        engine.llm.function_call_loop = fake_loop
        sim_time = engine.get_sim_time_str()
        await engine.dialogue.start_dialogue(
            [speaker, listener], "早啊。", "cafe", engine.trace, sim_time,
        )
        await engine.dialogue.advance_all(engine.trace, sim_time)

        prompt = captured["prompt"]
        assert "我此刻看到的周围环境" in prompt
        assert entity.name in prompt, "the room the speaker is standing in"
        assert "小美" in prompt and "在做什么" in prompt
        assert str(speaker._last_observation["observed_at"]) in prompt

    run(scenario())
