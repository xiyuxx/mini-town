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


def test_a_fresh_record_is_left_alone(tmp_path):
    """Refreshing is not free, so it happens on arrival, on change, and on a clock."""
    async def scenario():
        engine = await make_engine(tmp_path)
        agent = place(engine, "wang", "cafe")
        await engine.tick()
        fresh = agent._last_observation["observed_at"]

        await engine.tick()

        assert agent._last_observation["observed_at"] == fresh

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
