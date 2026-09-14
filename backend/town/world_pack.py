"""Load and validate declarative world content."""

from dataclasses import dataclass
import json
import os
from pathlib import Path

from .schedule import compile_routine_blocks

DEFAULT_WORLD_PATH = Path(__file__).resolve().parents[1] / "worlds" / "default.json"


@dataclass(frozen=True)
class WorldPack:
    path: Path
    data: dict

    @property
    def id(self) -> str:
        return str(self.data["id"])

    @property
    def name(self) -> str:
        return str(self.data.get("name", self.id))

    @property
    def setting(self) -> str:
        return str(self.data.get("setting", ""))

    @property
    def agents(self) -> list[dict]:
        return list(self.data.get("agents", []))

    @property
    def locations(self) -> list[dict]:
        return list(self.data.get("locations", []))

    @property
    def resources(self) -> list[dict]:
        return list(self.data.get("resources", []))

    def routine_blocks_for(self, agent_id: str) -> list[dict]:
        """Return flexible routine blocks keyed outside legacy slot schedules."""
        blocks = self.data.get("routine_blocks", {})
        if not isinstance(blocks, dict):
            return []
        return list(blocks.get(agent_id, []))

    @property
    def entities(self) -> list[dict]:
        return list(self.data.get("entities", []))

    @property
    def zones(self) -> list[dict]:
        return list(self.data.get("zones", []))

    @property
    def anchors(self) -> list[dict]:
        return list(self.data.get("anchors", []))

    @property
    def containers(self) -> list[dict]:
        return list(self.data.get("containers", []))

    @property
    def routes(self) -> list[dict]:
        return list(self.data.get("routes", []))

    @property
    def duties(self) -> list[dict]:
        return list(self.data.get("duties", []))

    @property
    def orders(self) -> list[dict]:
        return list(self.data.get("orders", []))

    @property
    def processes(self) -> list[dict]:
        return list(self.data.get("processes", []))

    @property
    def weather(self) -> dict:
        return dict(self.data.get("weather", {}))

    @property
    def festivals(self) -> list[dict]:
        return list(self.data.get("festivals", []))

    @property
    def infrastructure(self) -> dict:
        return dict(self.data.get("infrastructure", {}))

    @property
    def simulation(self) -> dict:
        return dict(self.data.get("simulation", {}))

    def initial_weather(self) -> dict:
        weather = self.weather
        initial = str(weather.get("initial", ""))
        state = dict(weather.get("states", {}).get(initial, {}))
        return {
            "condition": initial,
            "temperature": float(weather.get("season_base_temperature", {}).get("summer", 20)),
            "wind": next(iter(state.get("winds", [""])), ""),
            "description": str(state.get("description", initial)),
            "travel_cost": float(state.get("travel_cost", 0.0)),
            "sensory_salience": float(state.get("sensory_salience", 0.0)),
        }


def load_world_pack(path: str | Path | None = None) -> WorldPack:
    selected = Path(path or os.getenv("WORLD_PACK_PATH") or DEFAULT_WORLD_PATH).resolve()
    with selected.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    _validate(data)
    return WorldPack(selected, data)


def _validate(data: dict) -> None:
    for field in ("id", "map", "locations", "agents", "resources", "weather"):
        if field not in data:
            raise ValueError(f"world pack missing required field: {field}")
    width = int(data["map"].get("width", 0))
    height = int(data["map"].get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("world map dimensions must be positive")

    location_ids = _unique_ids(data["locations"], "location")
    agent_ids = _unique_ids(data["agents"], "agent")
    resource_ids = _unique_ids(data["resources"], "resource")
    entity_ids = _unique_ids(data.get("entities", []), "entity")
    process_ids = _unique_ids(data.get("processes", []), "process")
    zone_ids = _unique_ids(data.get("zones", []), "zone")
    anchor_ids = _unique_ids(data.get("anchors", []), "anchor")
    container_ids = _unique_ids(data.get("containers", []), "container")
    route_ids = _unique_ids(data.get("routes", []), "route")
    order_ids = _unique_ids(data.get("orders", []), "order")
    claimed_ids: dict[str, str] = {}
    for kind, ids in (
        ("location", location_ids), ("agent", agent_ids), ("resource", resource_ids),
        ("entity", entity_ids), ("process", process_ids), ("zone", zone_ids),
        ("anchor", anchor_ids), ("container", container_ids), ("route", route_ids),
        ("order", order_ids),
    ):
        for item_id in ids:
            previous = claimed_ids.setdefault(item_id, kind)
            if previous != kind:
                raise ValueError(f"{kind} id overlaps {previous} id: {item_id}")
    for location in data["locations"]:
        x, y = int(location["x"]), int(location["y"])
        w, h = int(location["width"]), int(location["height"])
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
            raise ValueError(f"location outside map: {location['id']}")
        access = location.get("access", {})
        for agent_id in access.get("allowed_agents", []):
            if agent_id and agent_id not in agent_ids:
                raise ValueError(f"location {location['id']} references unknown agent {agent_id}")

    # Roads and transit platforms may cross the map; destination buildings may not.
    destinations = [location for location in data["locations"] if not location.get("pass_through", False)]
    for index, first in enumerate(destinations):
        fx, fy = int(first["x"]), int(first["y"])
        fw, fh = int(first["width"]), int(first["height"])
        for second in destinations[index + 1:]:
            sx, sy = int(second["x"]), int(second["y"])
            sw, sh = int(second["width"]), int(second["height"])
            if fx < sx + sw and sx < fx + fw and fy < sy + sh and sy < fy + fh:
                raise ValueError(
                    f"locations overlap: {first['id']} and {second['id']}"
                )
    for agent in data["agents"]:
        if agent.get("location") not in location_ids:
            raise ValueError(f"agent {agent['id']} has unknown initial location")
        initial_needs = agent.get("initial_needs", {})
        if initial_needs and not isinstance(initial_needs, dict):
            raise ValueError(f"agent {agent['id']} initial_needs must be an object")
        for name, value in initial_needs.items():
            if name not in {"energy", "hunger", "social"}:
                raise ValueError(f"agent {agent['id']} has unknown initial need: {name}")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"agent {agent['id']} initial need {name} must be numeric") from exc
            if not 0 <= numeric <= 100:
                raise ValueError(f"agent {agent['id']} initial need {name} must be between 0 and 100")
        for item in agent.get("schedule", []):
            location = item.get("location") if isinstance(item, dict) else item[3]
            if location not in location_ids:
                raise ValueError(f"agent {agent['id']} schedule references unknown location {location}")
    routine_block_data = data.get("routine_blocks", {})
    if routine_block_data and not isinstance(routine_block_data, dict):
        raise ValueError("routine_blocks must map agent ids to block lists")
    for agent_id, raw_blocks in routine_block_data.items():
        if agent_id not in agent_ids:
            raise ValueError(f"routine_blocks references unknown agent {agent_id}")
        try:
            routine_blocks = compile_routine_blocks(raw_blocks)
        except ValueError as exc:
            raise ValueError(f"agent {agent_id} has invalid routine_blocks: {exc}") from exc
        for block in routine_blocks:
            unknown = [location for location in block.candidate_locations if location not in location_ids]
            if unknown:
                raise ValueError(
                    f"agent {agent_id} routine block {block.id} references unknown location {unknown[0]}"
                )
    for zone in data.get("zones", []):
        if zone.get("location") not in location_ids:
            raise ValueError(f"zone {zone['id']} references unknown location")
        if int(zone.get("capacity", 0)) < 0:
            raise ValueError(f"zone {zone['id']} capacity cannot be negative")
    for anchor in data.get("anchors", []):
        location_id = anchor.get("location")
        if location_id not in location_ids:
            raise ValueError(f"anchor {anchor['id']} references unknown location")
        if anchor.get("zone") and anchor["zone"] not in zone_ids:
            raise ValueError(f"anchor {anchor['id']} references unknown zone")
        if anchor.get("zone"):
            zone = next(item for item in data["zones"] if item["id"] == anchor["zone"])
            if zone["location"] != location_id:
                raise ValueError(f"anchor {anchor['id']} zone belongs to another location")
        location = next(item for item in data["locations"] if item["id"] == location_id)
        x, y = int(anchor.get("x", -1)), int(anchor.get("y", -1))
        if not (int(location["x"]) <= x < int(location["x"]) + int(location["width"])
                and int(location["y"]) <= y < int(location["y"]) + int(location["height"])):
            raise ValueError(f"anchor {anchor['id']} must be inside its location")
    for container in data.get("containers", []):
        if container.get("location") not in location_ids:
            raise ValueError(f"container {container['id']} references unknown location")
        if container.get("zone") and container["zone"] not in zone_ids:
            raise ValueError(f"container {container['id']} references unknown zone")
        if container.get("zone"):
            zone = next(item for item in data["zones"] if item["id"] == container["zone"])
            if zone["location"] != container["location"]:
                raise ValueError(f"container {container['id']} zone belongs to another location")
        anchor_id = container.get("anchor", container.get("anchor_id", ""))
        if anchor_id and anchor_id not in anchor_ids:
            raise ValueError(f"container {container['id']} references unknown anchor")
        if anchor_id:
            anchor = next(item for item in data["anchors"] if item["id"] == anchor_id)
            if anchor["location"] != container["location"]:
                raise ValueError(f"container {container['id']} anchor belongs to another location")
        if float(container.get("capacity", 0)) < 0:
            raise ValueError(f"container {container['id']} capacity cannot be negative")
    for route in data.get("routes", []):
        endpoint = route.get("endpoint_anchor_id", route.get("endpointAnchorId", ""))
        if endpoint and endpoint not in anchor_ids:
            raise ValueError(f"route {route['id']} references unknown endpoint anchor")
    for resource in data["resources"]:
        if resource.get("location") not in location_ids:
            raise ValueError(f"resource {resource['id']} references unknown location")
        properties = resource.get("properties", {})
        if properties and not isinstance(properties, dict):
            raise ValueError(f"resource {resource['id']} properties must be an object")
        service_facility_id = str(properties.get("service_facility_id", "")) if properties else ""
        if service_facility_id and service_facility_id not in entity_ids:
            raise ValueError(f"resource {resource['id']} references unknown service facility")
        if service_facility_id:
            facility = next(item for item in data.get("entities", []) if item["id"] == service_facility_id)
            if facility.get("location") != resource.get("location"):
                raise ValueError(f"resource {resource['id']} service facility is at another location")
            if "service" not in facility.get("capabilities", []):
                raise ValueError(f"resource {resource['id']} facility cannot provide service")
        container_id = resource.get("container", resource.get("container_id", ""))
        if container_id and container_id not in container_ids:
            raise ValueError(f"resource {resource['id']} references unknown container")
    for entity in data.get("entities", []):
        if entity.get("location") not in location_ids:
            raise ValueError(f"entity {entity['id']} references unknown location")
        anchor_id = entity.get("anchor", entity.get("anchor_id", ""))
        if anchor_id and anchor_id not in anchor_ids:
            raise ValueError(f"entity {entity['id']} references unknown anchor")
    for duty in data.get("duties", []):
        agent_id = duty.get("agent_id", duty.get("agentId", ""))
        facility_id = duty.get("facility_id", duty.get("facilityId", ""))
        if agent_id not in agent_ids:
            raise ValueError("duty references unknown agent")
        if facility_id not in entity_ids:
            raise ValueError("duty references unknown facility")
    for order in data.get("orders", []):
        if float(order.get("quantity", 0)) <= 0:
            raise ValueError(f"order {order['id']} quantity must be positive")
        route_id = str(order.get("route_id", order.get("routeId", "")))
        destination = str(order.get("destination_location", order.get("destinationLocation", "")))
        container_id = str(order.get("destination_container_id", order.get("destinationContainerId", "")))
        carrier_id = str(order.get("carrier_id", order.get("carrierId", "")))
        resource_kind = str(order.get("resource_kind", order.get("resourceKind", "")))
        if route_id not in route_ids:
            raise ValueError(f"order {order['id']} references unknown route")
        if destination not in location_ids:
            raise ValueError(f"order {order['id']} references unknown destination")
        if container_id not in container_ids:
            raise ValueError(f"order {order['id']} references unknown destination container")
        container = next(item for item in data.get("containers", []) if item["id"] == container_id)
        if container.get("location") != destination:
            raise ValueError(f"order {order['id']} destination container is at another location")
        route = next(item for item in data.get("routes", []) if item["id"] == route_id)
        supported = route.get("supported_resource_kinds", route.get("supportedResourceKinds", []))
        if resource_kind not in supported:
            raise ValueError(f"order {order['id']} resource kind is unsupported by route")
        accepts = container.get("accepts_kinds", container.get("acceptsKinds", []))
        if accepts and resource_kind not in accepts:
            raise ValueError(f"order {order['id']} resource kind is rejected by container")
        if carrier_id and carrier_id not in agent_ids:
            raise ValueError(f"order {order['id']} references unknown carrier")
        source = str(order.get("source_location", order.get("sourceLocation", "")))
        if source and source not in location_ids:
            raise ValueError(f"order {order['id']} references unknown source")
    for process in data.get("processes", []):
        if int(process.get("duration_minutes", 0)) <= 0:
            raise ValueError(f"process {process['id']} duration must be positive")
        if not process.get("outputs"):
            raise ValueError(f"process {process['id']} must declare outputs")
        anchor_capability = str(process.get("required_anchor_capability", ""))
        if anchor_capability and not any(anchor_capability in item.get("capabilities", []) for item in data.get("anchors", [])):
            raise ValueError(f"process {process['id']} requires unknown anchor capability")
        for item in [*process.get("inputs", []), *process.get("outputs", [])]:
            if not str(item.get("kind", "")).strip() or float(item.get("quantity", 0)) <= 0:
                raise ValueError(f"process {process['id']} has invalid input or output")


def _unique_ids(items: list[dict], kind: str) -> set[str]:
    ids = [str(item.get("id", "")) for item in items]
    if any(not item_id for item_id in ids):
        raise ValueError(f"{kind} id cannot be empty")
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate {kind} id")
    return set(ids)


DEFAULT_WORLD = load_world_pack()
