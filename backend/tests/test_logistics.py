import copy
from pathlib import Path

import pytest

from backend.town.engine import SimulationEngine
from backend.town.interactions import ResourceStore
from backend.town.logistics import LogisticsEngine
from backend.town.scene import SceneIndex
from backend.town.tasks import TaskStore
from backend.town.world_pack import DEFAULT_WORLD, WorldPack, _validate


def make_pack(**order_changes):
    data = copy.deepcopy(DEFAULT_WORLD.data)
    order = {
        "id": "test_order",
        "resource_kind": "package",
        "quantity": 4,
        "destination_location": "restaurant",
        "destination_container_id": "meal_ingredients_storage",
        "route_id": "outside_supply_route",
        "carrier_id": "mei",
        "requested_at": 0,
    }
    order.update(order_changes)
    data["orders"] = [order]
    _validate(data)
    return WorldPack(Path("test-world.json"), data)


def make_logistics(**order_changes):
    pack = make_pack(**order_changes)
    resources = ResourceStore(pack)
    resources.reset()
    tasks = TaskStore()
    logistics = LogisticsEngine(pack, resources, SceneIndex(pack), tasks)
    logistics.reset()
    return logistics, resources, tasks


def test_external_order_reaches_container_and_completes_tasks():
    logistics, resources, tasks = make_logistics()

    first_events = logistics.advance(0)
    second_events = logistics.advance(180)
    order = logistics.orders["test_order"]
    shipment = logistics.shipments[order.shipment_id]
    stored = resources.get(shipment.resource_id)

    assert order.status == "delivered"
    assert shipment.status == "received"
    assert stored is not None
    assert stored.container_id == "meal_ingredients_storage"
    assert stored.location == "restaurant"
    assert {event["type"] for event in first_events} >= {
        "order_confirmed", "shipment_planned", "shipment_dispatched",
    }
    assert {event["type"] for event in second_events} >= {
        "shipment_arrived", "shipment_received",
    }
    assert tasks.tasks[f"logistics:receive:{shipment.id}"].status == "completed"


def test_local_supply_shortage_fails_without_creating_shipment():
    logistics, _resources, _tasks = make_logistics(
        source_location="restaurant", resource_kind="ingredient", quantity=999,
    )

    logistics.advance(0)

    order = logistics.orders["test_order"]
    assert order.status == "failed"
    assert "数量不足" in order.failure_reason
    assert order.shipment_id == ""
    assert logistics.shipments == {}


def test_route_and_container_references_are_validated():
    with pytest.raises(ValueError, match="unsupported|不支持"):
        make_pack(resource_kind="unknown_kind")
    with pytest.raises(ValueError, match="container|容器"):
        make_pack(destination_container_id="missing_container")
    with pytest.raises(ValueError, match="carrier|承运"):
        make_pack(carrier_id="missing_agent")


def test_capacity_shortage_blocks_arrived_shipment_without_delivery():
    data = copy.deepcopy(DEFAULT_WORLD.data)
    data["containers"] = copy.deepcopy(data["containers"])
    next(item for item in data["containers"] if item["id"] == "meal_ingredients_storage")["capacity"] = 1
    data["orders"] = [{
        "id": "capacity_order",
        "resource_kind": "package",
        "quantity": 4,
        "destination_location": "restaurant",
        "destination_container_id": "meal_ingredients_storage",
        "route_id": "outside_supply_route",
        "carrier_id": "mei",
        "requested_at": 0,
    }]
    _validate(data)
    pack = WorldPack(Path("capacity-world.json"), data)
    resources = ResourceStore(pack)
    resources.reset()
    tasks = TaskStore()
    logistics = LogisticsEngine(pack, resources, SceneIndex(pack), tasks)
    logistics.reset()

    logistics.advance(0)
    logistics.advance(180)

    order = logistics.orders["capacity_order"]
    shipment = logistics.shipments[order.shipment_id]
    assert order.status == "shipping"
    assert shipment.status == "blocked"
    assert "容量" in shipment.failure_reason
    assert tasks.tasks[f"logistics:receive:{shipment.id}"].status == "blocked"


def test_engine_state_exposes_logistics():
    engine = SimulationEngine()
    state = engine.get_state()
    assert "logistics" in state
    assert state["logistics"]["orders"]
    assert state["logistics"]["orders"][0]["id"] == "meal_ingredient_restock_day1"
