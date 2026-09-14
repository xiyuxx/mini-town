"""Scenario-neutral spatial and functional indexes.

The simulation kernel addresses places, interaction anchors, containers, routes,
and duties. World-specific labels such as a bookshop or a cargo airlock remain
content in a WorldPack rather than Python control flow.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .world_pack import WorldPack


@dataclass(frozen=True)
class SceneZone:
    id: str
    location_id: str
    name: str = ""
    tags: tuple[str, ...] = ()
    capacity: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id, "location": self.location_id, "name": self.name,
            "tags": list(self.tags), "capacity": self.capacity,
        }


@dataclass(frozen=True)
class SceneAnchor:
    id: str
    location_id: str
    x: int
    y: int
    name: str = ""
    zone_id: str = ""
    capabilities: tuple[str, ...] = ()
    capacity: int = 1

    def to_dict(self) -> dict:
        return {
            "id": self.id, "location": self.location_id, "x": self.x, "y": self.y,
            "name": self.name, "zone": self.zone_id,
            "capabilities": list(self.capabilities), "capacity": self.capacity,
        }


@dataclass(frozen=True)
class SceneContainer:
    id: str
    location_id: str
    name: str = ""
    anchor_id: str = ""
    zone_id: str = ""
    capacity: float = 0.0
    accepts_kinds: tuple[str, ...] = ()
    access: dict = field(default_factory=lambda: {"mode": "public"})

    def to_dict(self) -> dict:
        return {
            "id": self.id, "location": self.location_id, "name": self.name,
            "anchor": self.anchor_id, "zone": self.zone_id, "capacity": self.capacity,
            "acceptsKinds": list(self.accepts_kinds), "access": dict(self.access),
        }


@dataclass(frozen=True)
class SceneRoute:
    id: str
    name: str = ""
    route_type: str = "local"
    endpoint_anchor_id: str = ""
    supported_resource_kinds: tuple[str, ...] = ()
    travel_minutes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "type": self.route_type,
            "endpointAnchorId": self.endpoint_anchor_id,
            "supportedResourceKinds": list(self.supported_resource_kinds),
            "travelMinutes": dict(self.travel_minutes),
        }


@dataclass(frozen=True)
class Duty:
    agent_id: str
    facility_id: str
    capabilities: tuple[str, ...] = ()
    schedule_template: str = ""

    def to_dict(self) -> dict:
        return {
            "agentId": self.agent_id, "facilityId": self.facility_id,
            "capabilities": list(self.capabilities), "scheduleTemplate": self.schedule_template,
        }


class SceneIndex:
    """Read-only lookup indexes compiled from a declarative world pack."""

    def __init__(self, world_pack: "WorldPack"):
        self.zones: dict[str, SceneZone] = {
            item.id: item for item in (
                SceneZone(
                    id=str(raw["id"]), location_id=str(raw["location"]),
                    name=str(raw.get("name", raw["id"])),
                    tags=tuple(str(tag) for tag in raw.get("tags", [])),
                    capacity=max(0, int(raw.get("capacity", 0))),
                )
                for raw in world_pack.zones
            )
        }
        self.anchors: dict[str, SceneAnchor] = {
            item.id: item for item in (
                SceneAnchor(
                    id=str(raw["id"]), location_id=str(raw["location"]),
                    x=int(raw["x"]), y=int(raw["y"]),
                    name=str(raw.get("name", raw["id"])),
                    zone_id=str(raw.get("zone", "")),
                    capabilities=tuple(str(capability) for capability in raw.get("capabilities", [])),
                    capacity=max(1, int(raw.get("capacity", 1))),
                )
                for raw in world_pack.anchors
            )
        }
        self.containers: dict[str, SceneContainer] = {
            item.id: item for item in (
                SceneContainer(
                    id=str(raw["id"]), location_id=str(raw["location"]),
                    name=str(raw.get("name", raw["id"])),
                    anchor_id=str(raw.get("anchor", raw.get("anchor_id", ""))),
                    zone_id=str(raw.get("zone", "")),
                    capacity=max(0.0, float(raw.get("capacity", 0))),
                    accepts_kinds=tuple(str(kind) for kind in raw.get("accepts_kinds", raw.get("acceptsKinds", []))),
                    access=dict(raw.get("access", {"mode": "public"})),
                )
                for raw in world_pack.containers
            )
        }
        self.routes: dict[str, SceneRoute] = {
            item.id: item for item in (
                SceneRoute(
                    id=str(raw["id"]), name=str(raw.get("name", raw["id"])),
                    route_type=str(raw.get("type", "local")),
                    endpoint_anchor_id=str(raw.get("endpoint_anchor_id", raw.get("endpointAnchorId", ""))),
                    supported_resource_kinds=tuple(str(kind) for kind in raw.get("supported_resource_kinds", raw.get("supportedResourceKinds", []))),
                    travel_minutes=dict(raw.get("travel_minutes", raw.get("travelMinutes", {}))),
                )
                for raw in world_pack.routes
            )
        }
        self.duties: list[Duty] = [
            Duty(
                agent_id=str(raw.get("agent_id") or raw.get("agentId", "")),
                facility_id=str(raw.get("facility_id") or raw.get("facilityId", "")),
                capabilities=tuple(str(capability) for capability in raw.get("capabilities", [])),
                schedule_template=str(raw.get("schedule_template", raw.get("scheduleTemplate", ""))),
            )
            for raw in world_pack.duties
        ]

    def anchors_at(self, location_id: str, capability: str = "") -> list[SceneAnchor]:
        return [
            anchor for anchor in self.anchors.values()
            if anchor.location_id == location_id
            and (not capability or capability in anchor.capabilities)
        ]

    def containers_at(self, location_id: str) -> list[SceneContainer]:
        return [container for container in self.containers.values() if container.location_id == location_id]

    def duties_for(self, agent_id: str) -> list[Duty]:
        return [duty for duty in self.duties if duty.agent_id == agent_id]

    def to_dict(self) -> dict:
        return {
            "zones": [zone.to_dict() for zone in self.zones.values()],
            "anchors": [anchor.to_dict() for anchor in self.anchors.values()],
            "containers": [container.to_dict() for container in self.containers.values()],
            "routes": [route.to_dict() for route in self.routes.values()],
            "duties": [duty.to_dict() for duty in self.duties],
        }
