from backend.town.engine import SimulationEngine
from backend.town.world_query import WorldQuery


def test_world_query_returns_bounded_accessible_reachable_locations():
    engine = SimulationEngine()
    engine._init_agents()
    agent = next(item for item in engine.agents if item.id == "wang")
    query = WorldQuery(engine.world_pack, engine.resources)

    locations = query.locations_for_intention(
        agent, engine, {"interaction_type": "wait", "target_location": "park"}, limit=3,
    )

    ids = [item["id"] for item in locations]
    assert len(locations) == 3
    assert "home_mei" not in ids
    assert "park" in ids
    assert all(item["accessible"] and item["reachable"] for item in locations)


def test_world_query_uses_intention_target_agent_to_rank_location():
    engine = SimulationEngine()
    engine._init_agents()
    agent = next(item for item in engine.agents if item.id == "mei")
    target = next(item for item in engine.agents if item.id == "hua")
    target.state.current_location = "cafe"
    query = WorldQuery(engine.world_pack, engine.resources)

    locations = query.locations_for_intention(
        agent, engine, {
            "interaction_type": "communicate",
            "target_agent_id": target.id,
        }, limit=1,
    )

    assert locations[0]["id"] == "cafe"
    assert "目标角色在此处" in locations[0]["reasons"]


def test_world_query_reuses_observation_snapshot_and_reports_constraints():
    engine = SimulationEngine()
    engine._init_agents()
    agent = engine.agents[0]
    query = WorldQuery(engine.world_pack, engine.resources)

    assert query.visible_environment(agent, engine) == engine.interactions.observation_snapshot(agent, engine)
    engine.weather = {"condition": "大雨", "travel_cost": 0.8}
    constraints = query.current_constraints(agent, engine)
    assert {item["type"] for item in constraints} >= {"travel_cost"}
