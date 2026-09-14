"""The world reports what actually happened, and people act on it.

Covers the replacement for the old flavour-event generator: weather and
infrastructure changes are real state, they are announced, they are learned by
whoever is affected, and they change where free agents choose to go.
"""

import asyncio

from backend.town.candidates import build_routine_candidates
from backend.town.engine import SimulationEngine
from backend.town.world import is_outdoor


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


def clear_weather():
    return {"condition": "晴", "temperature": 25.0, "wind": "微风",
            "description": "晴朗，阳光明媚", "travel_cost": 0.0, "sensory_salience": 0.1}


def rain_weather():
    return {"condition": "大雨", "temperature": 18.0, "wind": "和风",
            "description": "大雨滂沱", "travel_cost": 0.7, "sensory_salience": 0.7}


def test_rain_moves_free_time_indoors(tmp_path):
    """Weather that makes travel expensive is the same weather that shelters."""
    async def scenario():
        engine = await make_engine(tmp_path)
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

        engine.weather = clear_weather()
        dry = {item.id: item for item in build_routine_candidates(chosen, engine)}
        engine.weather = rain_weather()
        wet = {item.id: item for item in build_routine_candidates(chosen, engine)}
        assert set(dry) == set(wet)

        raised = [item for key, item in wet.items() if item.score > dry[key].score]
        lowered = [item for key, item in wet.items() if item.score < dry[key].score]
        assert raised and all(not is_outdoor(item.action["location"]) for item in raised)
        assert lowered and all(is_outdoor(item.action["location"]) for item in lowered)

    run(scenario())


def test_weather_change_is_announced(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        await engine.tick()
        engine.town_agent.weather_engine._condition = "小雨"
        events = engine.town_agent._weather_events(
            {"condition": "小雨", "description": "下着小雨"},
        )
        engine.town_agent.weather_engine._condition = "晴"
        assert len(events) == 1
        assert events[0]["type"] == "weather_change"
        assert "小雨" in events[0]["description"]
        # Repeating the same condition is not news.
        assert engine.town_agent._weather_events({"condition": "小雨"}) == []

    run(scenario())


def test_service_failure_is_announced_and_learned_by_everyone(tmp_path):
    """A town-wide outage is noticed by all; a closed street by whoever is there."""
    async def scenario():
        engine = await make_engine(tmp_path)
        normal = {"power": "normal", "water": "normal", "road_main": "open"}
        assert engine.town_agent._infrastructure_events(normal) == []
        power = engine.town_agent._infrastructure_events(
            {"power": "outage", "water": "normal", "road_main": "open"},
        )
        assert power and power[0]["description"] == "停电了"

        await engine._record_town_events(power)
        fact = next(item for item in engine.fact_ledger.known_for(engine.agents[0].id, limit=10)
                    if item.type == "infrastructure")
        assert set(fact.known_by) == {agent.id for agent in engine.agents}

        street = engine.town_agent._infrastructure_events(
            {"power": "outage", "water": "normal", "road_main": "closed"},
        )
        assert street and street[0]["description"] == "主街封闭"
        assert street[0]["location"] == "road_main"

    run(scenario())


def test_taking_the_last_one_is_reported(tmp_path):
    """A shortage is the result of a real purchase, so it is announced."""
    async def scenario():
        engine = await make_engine(tmp_path)
        agent = engine.agents[0]
        engine.agents = [agent]
        coffee = engine.resources.get("food_cafe")
        coffee.quantity = 1
        agent.state.current_location = "cafe"
        agent.state.x, agent.state.y = 6, 4
        agent.state.status = "ACTING"
        agent._activity_ticks = 1
        agent._activity_desc = "喝咖啡"
        agent._activity_source = "test"
        agent._activity_started_at = engine.get_sim_timestamp()
        agent._activity_action = {
            "interaction_type": "consume", "resource_id": "food_cafe", "quantity": 1,
        }
        agent._activity_effects = [
            {"type": "resource_change", "resource_id": "food_cafe", "delta": -1},
        ]

        events = await engine._process_acting(agent, engine.get_sim_time_str())

        stock = next(event for event in events if event["type"] == "out_of_stock")
        assert "阳光咖啡馆" in stock["content"] and "可食用食物" in stock["content"]
        fact = next(
            item for item in engine.fact_ledger.known_for(agent.id, limit=10)
            if item.type == "out_of_stock"
        )
        assert fact.details["resource_id"] == "food_cafe"
        assert fact.location == "cafe"

    run(scenario())


def test_closed_main_street_blocks_actions_there(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        agent = engine.agents[0]
        engine.agents = [agent]
        agent.state.status = "IDLE"
        engine.infrastructure = {"power": "normal", "water": "normal", "road_main": "normal"}
        reachable = engine.interactions.validate(agent, {"interaction_type": "move", "location": "park"}, engine)
        assert reachable.feasible

        engine.infrastructure = {"power": "normal", "water": "normal", "road_main": "closed"}
        blocked = engine.interactions.validate(agent, {"interaction_type": "move", "location": "road_main"}, engine)
        assert not blocked.feasible
        assert any("主街" in reason for reason in blocked.reasons)

    run(scenario())


def test_outage_stops_service_but_not_sitting_there(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        agent = engine.agents[0]
        agent.state.current_location = "cafe"
        agent.state.x, agent.state.y = 6, 4
        agent.state.status = "IDLE"
        for other in engine.agents:
            if other.id != agent.id:
                other.state.current_location = "home_mei"
        engine.infrastructure = {"power": "normal", "water": "normal", "road_main": "open"}
        inspection = engine.interactions.validate(
            agent, {"interaction_type": "inspect", "content": "查看咖啡吧台"}, engine,
        )
        engine.interactions.apply_effects(
            agent, inspection.resolved_action, inspection.expected_effects,
            engine.get_sim_timestamp(),
        )
        me = next(item for item in engine.agents if item.id == "mei")
        me.state.current_location = "cafe"
        inspection = engine.interactions.validate(
            agent, {"interaction_type": "inspect", "content": "查看咖啡吧台"}, engine,
        )
        engine.interactions.apply_effects(
            agent, inspection.resolved_action, inspection.expected_effects,
            engine.get_sim_timestamp(),
        )
        served = engine.interactions.validate(
            agent, {"interaction_type": "request_service", "resource_id": "food_cafe"}, engine,
        )
        assert served.feasible

        engine.infrastructure = {"power": "outage", "water": "normal", "road_main": "open"}
        blocked = engine.interactions.validate(
            agent, {"interaction_type": "request_service", "resource_id": "food_cafe"}, engine,
        )
        assert not blocked.feasible
        assert any("服务中断" in reason for reason in blocked.reasons)
        waiting = engine.interactions.validate(
            agent, {"interaction_type": "wait", "content": "在咖啡馆坐一会儿"}, engine,
        )
        assert waiting.feasible

    run(scenario())
