import asyncio

from backend.town.engine import SimulationEngine


def test_state_version_is_exposed_and_advances_per_tick(tmp_path):
    async def scenario():
        engine = SimulationEngine()
        engine.memory.db_path = str(tmp_path / "version" / "town.db")
        await engine.init()
        initial = engine.state_version
        await engine.tick()
        assert engine.state_version == initial + 1
        assert engine.get_state()["stateVersion"] == engine.state_version
        await engine.reset()
        assert engine.state_version == 0
        await engine.shutdown()
    asyncio.run(scenario())
