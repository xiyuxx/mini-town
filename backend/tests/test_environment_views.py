"""One record per agent, one declared projection per reader."""

import asyncio

from backend.town.engine import SimulationEngine
from backend.town.environment import ENVIRONMENT_VIEWS, view


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


def test_the_planning_view_keeps_the_world_and_drops_the_bookkeeping(tmp_path):
    """A plan cites ids, so it keeps the world; the hash is the engine's business."""
    async def scenario():
        engine = await make_engine(tmp_path)
        agent = next(item for item in engine.agents if item.id == "wang")
        agent.state.current_location = "cafe"
        record = engine.interactions.observation_snapshot(agent, engine)

        planned = view(record, "planning")

        assert "signature" not in planned, "the model has no use for the hash"
        assert planned == {key: value for key, value in record.items() if key != "signature"}

    run(scenario())


def test_the_dialogue_view_hides_what_a_person_would_not_say(tmp_path):
    """Names and what people are doing — no ids, no capabilities, no signature."""
    async def scenario():
        engine = await make_engine(tmp_path)
        speaker = next(item for item in engine.agents if item.id == "wang")
        listener = next(item for item in engine.agents if item.id == "mei")
        for agent in (speaker, listener):
            agent.state.current_location = "cafe"
            agent.state.x, agent.state.y = 8, 6
        record = engine.interactions.observation_snapshot(speaker, engine)
        assert record["entities"], "the café must have something in it for this to mean anything"

        seen = view(record, "dialogue")

        assert "signature" not in seen
        assert "available_processes" not in seen
        assert listener.id in {item["id"] for item in seen["nearby_agents"]}
        assert set(seen["nearby_agents"][0]) == {"id", "status", "action"}
        for entity in seen["entities"]:
            assert set(entity) <= {"name", "kind"}
            assert "id" not in entity

    run(scenario())


def test_an_undeclared_reader_is_refused(tmp_path):
    """A new layer declares what it sees instead of quietly receiving everything."""
    async def scenario():
        engine = await make_engine(tmp_path)
        record = engine.interactions.observation_snapshot(engine.agents[0], engine)

        try:
            view(record, "some_layer_that_does_not_exist")
            raise AssertionError("an undeclared reader must not be served")
        except KeyError:
            pass

    run(scenario())


def test_a_record_that_was_never_taken_projects_to_nothing(tmp_path):
    """No record means no view — every reader, not just the one that asked."""
    async def scenario():
        engine = await make_engine(tmp_path)
        record = None

        for reader in ENVIRONMENT_VIEWS:
            assert view(record, reader) == {}
        # A live agent that has never observed still gets a view of a real place.
        fresh = next(item for item in engine.agents)
        assert "location" in view(
            engine.interactions.observation_snapshot(fresh, engine), "dialogue",
        )

    run(scenario())
