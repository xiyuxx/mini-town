import asyncio

from backend.town.engine import SimulationEngine
from backend.town.interactions import InteractionEngine


async def _engine_async():
    engine = SimulationEngine()
    engine.llm.fallback = True
    engine.embedding_provider.fallback = True
    await engine.init()
    agent = next(item for item in engine.agents if item.id == "hua")
    agent.state.current_location = "restaurant"
    agent.state.x, agent.state.y = 16, 4
    agent.state.status = "IDLE"
    return engine, agent


def _engine():
    return asyncio.run(_engine_async())


def test_take_put_and_consume_use_same_generic_resource_flow():
    engine, agent = _engine()
    store = engine.resources
    food = store.get("food_cafe")
    food.location = "restaurant"
    food.quantity = 1
    food.properties = {"nutrition": 20, "energy": 3}
    interaction = InteractionEngine(store)

    take = interaction.validate(agent, {"interaction_type": "take", "resource_id": food.id}, engine)
    assert take.feasible
    interaction.apply_effects(agent, take.resolved_action, take.expected_effects)
    assert food.location == f"agent:{agent.id}"

    consume = interaction.validate(agent, {"interaction_type": "consume", "resource_id": food.id}, engine)
    assert consume.feasible
    hunger_before = agent.state.needs["hunger"]
    interaction.apply_effects(agent, consume.resolved_action, consume.expected_effects)
    assert food.quantity == 0
    assert agent.state.needs["hunger"] < hunger_before


def test_transfer_uses_scenario_container_capacity_and_kind_rules():
    engine, agent = _engine()
    interaction = InteractionEngine(engine.resources)
    ingredient = engine.resources.get("ingredient_restaurant")

    transfer = interaction.validate(agent, {
        "interaction_type": "transfer",
        "resource_id": ingredient.id,
        "quantity": 2,
        "destination": "meal_ingredients_storage",
    }, engine)
    assert transfer.feasible
    interaction.apply_effects(agent, transfer.resolved_action, transfer.expected_effects)
    assert ingredient.location == "restaurant"
    stored = next(item for item in engine.resources.resources.values()
                  if item.container_id == "meal_ingredients_storage")
    assert stored.quantity == 2
    assert engine.resources.container_usage("meal_ingredients_storage") == 2

    agent.state.current_location = "bookstore"
    ingredient.location = "bookstore"
    wrong_kind = interaction.validate(agent, {
        "interaction_type": "transfer",
        "resource_id": ingredient.id,
        "quantity": 1,
        "destination": "archive_storage",
    }, engine)
    assert not wrong_kind.feasible
    assert any("不接收" in reason for reason in wrong_kind.reasons)


def test_process_requires_device_and_input_then_adds_output():
    engine, agent = _engine()
    interaction = InteractionEngine(engine.resources)
    ingredient = engine.resources.get("ingredient_restaurant")
    assert ingredient.quantity > 0

    action = {"interaction_type": "operate", "process_id": "prepare_meal"}
    result = interaction.validate(agent, action, engine)
    assert result.feasible
    before = engine.resources.get("food_restaurant").quantity
    applied = interaction.apply_effects(agent, result.resolved_action, result.expected_effects)
    assert engine.resources.get("food_restaurant").quantity == before + 1
    assert ingredient.quantity == 11
    assert result.resolved_action["anchor_id"] == "meal_prep_anchor"
    assert any(effect["type"] == "entity_used" for effect in applied)
    assert any(effect["type"] == "anchor_used" for effect in applied)


def test_process_is_blocked_when_device_state_is_invalid():
    engine, agent = _engine()
    device = engine.resources.get("device_restaurant_cooker")
    device.state["operational"] = False
    result = engine.interactions.validate(
        agent, {"interaction_type": "operate", "process_id": "prepare_meal"}, engine
    )
    assert not result.feasible
    assert any("设备状态" in reason for reason in result.reasons)


def test_inspect_records_snapshot_and_blocks_unchanged_repeat():
    engine, agent = _engine()
    first = engine.interactions.validate(
        agent, {"interaction_type": "inspect", "content": "查看环境"}, engine
    )
    assert first.feasible
    applied = engine.interactions.apply_effects(
        agent, first.resolved_action, first.expected_effects, engine.get_sim_timestamp()
    )
    observation = next(effect for effect in applied if effect["type"] == "observation")
    assert observation["snapshot"]["location"] == "restaurant"
    assert observation["snapshot"]["entities"]
    assert agent._last_observation["observed_at"] == engine.get_sim_timestamp()

    repeated = engine.interactions.validate(
        agent, {"interaction_type": "inspect", "content": "再次查看环境"}, engine
    )
    assert not repeated.feasible
    assert any("环境没有变化" in reason for reason in repeated.reasons)

    engine.resources.get("ingredient_restaurant").quantity -= 1
    changed = engine.interactions.validate(
        agent, {"interaction_type": "inspect", "content": "查看变化"}, engine
    )
    assert changed.feasible


def test_inspect_completion_creates_environment_observed_fact():
    async def scenario():
        engine, agent = await _engine_async()
        result = engine.interactions.validate(
            agent, {"interaction_type": "inspect", "content": "查看餐厅", "duration_minutes": 5}, engine
        )
        action = dict(result.resolved_action)
        action["validated_effects"] = result.expected_effects
        await engine._execute_decision(agent, action, engine.get_sim_time_str())
        events = await engine._process_acting(agent, engine.get_sim_time_str())
        facts = engine.fact_ledger.known_for(agent.id, limit=20)
        observed = next(fact for fact in facts if fact.type == "environment_observed")
        completed = next(fact for fact in facts if fact.type == "activity_completed")
        assert observed.details["entities"]
        assert observed.details["observed_at"] == engine.get_sim_timestamp()
        assert observed.id in completed.source_ids
        assert "看到" in events[0]["outcome"]

    asyncio.run(scenario())


def test_environment_is_visible_in_engine_state():
    engine, _ = _engine()
    state = engine.get_state()
    assert "environment" in state
    restaurant = state["environment"]["locations"]["restaurant"]
    assert any(item["id"] == "device_restaurant_cooker" for item in restaurant["entities"])
    assert any(item["id"] == "prepare_meal" for item in restaurant["processes"])
    assert state["environment"]["inventories"]["hua"] == []
    assert any(anchor["id"] == "meal_prep_anchor" for anchor in state["scene"]["anchors"])
    assert any(route["id"] == "outside_supply_route" for route in state["scene"]["routes"])
    assert any(duty["agentId"] == "liu" for duty in state["scene"]["duties"])


def test_cafe_service_requires_observation_staff_and_acquisition():
    engine, agent = _engine()
    agent.state.current_location = "cafe"
    agent.state.x, agent.state.y = 6, 4
    mei = next(item for item in engine.agents if item.id == "mei")
    mei.state.current_location = "home_mei"

    unobserved = engine.interactions.validate(
        agent, {"interaction_type": "request_service", "resource_id": "food_cafe"}, engine
    )
    assert not unobserved.feasible
    assert any("先查看" in reason for reason in unobserved.reasons)

    inspection = engine.interactions.validate(
        agent, {"interaction_type": "inspect", "content": "查看咖啡吧台"}, engine
    )
    assert inspection.feasible
    engine.interactions.apply_effects(
        agent, inspection.resolved_action, inspection.expected_effects, engine.get_sim_timestamp()
    )
    unattended = engine.interactions.validate(
        agent, {"interaction_type": "request_service", "resource_id": "food_cafe"}, engine
    )
    assert not unattended.feasible
    assert any("工作人员" in reason for reason in unattended.reasons)

    mei.state.current_location = "cafe"
    refreshed_inspection = engine.interactions.validate(
        agent, {"interaction_type": "inspect", "content": "再次查看咖啡吧台"}, engine
    )
    assert refreshed_inspection.feasible
    engine.interactions.apply_effects(
        agent, refreshed_inspection.resolved_action,
        refreshed_inspection.expected_effects, engine.get_sim_timestamp(),
    )
    service = engine.interactions.validate(
        agent, {"interaction_type": "request_service", "resource_id": "food_cafe"}, engine
    )
    assert service.feasible
    applied = engine.interactions.apply_effects(agent, service.resolved_action, service.expected_effects)
    acquired_id = next(effect["moved_resource_id"] for effect in applied if effect["type"] == "resource_move")
    assert engine.resources.get(acquired_id).location == f"agent:{agent.id}"

    consume = engine.interactions.validate(
        agent, {"interaction_type": "consume", "resource_id": acquired_id}, engine
    )
    assert consume.feasible


def test_closed_or_sold_out_cafe_service_returns_structured_blocker():
    engine, agent = _engine()
    agent.state.current_location = "cafe"
    mei = next(item for item in engine.agents if item.id == "mei")
    mei.state.current_location = "cafe"

    inspection = engine.interactions.validate(
        agent, {"interaction_type": "inspect", "content": "查看咖啡吧台"}, engine
    )
    engine.interactions.apply_effects(
        agent, inspection.resolved_action, inspection.expected_effects, engine.get_sim_timestamp()
    )

    facility = engine.resources.get("cafe_service_facility")
    facility.state["operational"] = False
    closed = engine.interactions.validate(
        agent, {"interaction_type": "request_service", "resource_id": "food_cafe"}, engine
    )
    assert not closed.feasible
    assert "服务台当前未开放" in closed.reasons

    facility.state["operational"] = True
    engine.resources.get("food_cafe").quantity = 0
    sold_out = engine.interactions.validate(
        agent, {"interaction_type": "request_service", "resource_id": "food_cafe"}, engine
    )
    assert not sold_out.feasible
    assert "请求的商品当前不可用" in sold_out.reasons
