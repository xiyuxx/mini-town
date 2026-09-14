"""Data-driven world entities, processes, feasibility, and deterministic effects."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import uuid

from .world import LOCATION_MAP, bfs_path, can_enter, is_outdoor, location_center, location_name
from .world_pack import DEFAULT_WORLD, WorldPack
from .scene import SceneIndex
from ..config import config

INTERACTION_TYPES = {
    "move", "consume", "rest", "communicate", "inspect", "request_service", "use_resource",
    "produce", "transfer", "take", "put", "operate", "work_on_goal", "wait",
}


@dataclass
class Resource:
    id: str
    kind: str
    location: str
    quantity: float
    properties: dict = field(default_factory=dict)
    owner_id: str | None = None
    availability: str = "available"
    known_by: list[str] = field(default_factory=list)
    name: str = ""
    entity_type: str = "resource"
    portable: bool = True
    capabilities: list[str] = field(default_factory=list)
    components: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)
    anchor_id: str = ""
    container_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeasibilityResult:
    option_id: str
    feasible: bool
    reasons: list[str] = field(default_factory=list)
    estimated_minutes: int = 0
    expected_effects: list[dict] = field(default_factory=list)
    resolved_action: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ResourceStore:
    """Runtime store for resources, objects, devices, and process definitions."""

    def __init__(self, world_pack: WorldPack = DEFAULT_WORLD):
        self.world_pack = world_pack
        self.scene = SceneIndex(world_pack)
        self.resources: dict[str, Resource] = {}
        self.processes: dict[str, dict] = {}

    def add(self, resource: Resource) -> Resource:
        self.resources[resource.id] = resource
        return resource

    def get(self, resource_id: str) -> Resource | None:
        return self.resources.get(resource_id)

    def at_location(self, location: str, kind: str = "", available_only: bool = True) -> list[Resource]:
        return [
            resource for resource in self.resources.values()
            if resource.location == location
            and (not kind or resource.kind == kind)
            and (not available_only or resource.availability == "available")
            and resource.quantity > 0
        ]

    def accessible_to(self, location: str, agent_id: str, kind: str = "") -> list[Resource]:
        inventory = f"agent:{agent_id}"
        return [
            resource for resource in self.resources.values()
            if resource.location in {location, inventory}
            and (not kind or resource.kind == kind)
            and resource.availability == "available"
            and resource.quantity > 0
        ]

    def visible_at(self, location: str, agent_id: str) -> list[dict]:
        result = []
        for resource in self.at_location(location):
            if resource.known_by and agent_id not in resource.known_by:
                continue
            result.append(resource.to_dict())
        return result

    def inventory(self, agent_id: str) -> list[dict]:
        return [item.to_dict() for item in self.at_location(f"agent:{agent_id}")]

    def container_usage(self, container_id: str) -> float:
        return sum(
            resource.quantity for resource in self.resources.values()
            if resource.container_id == container_id
        )

    def scene_state(self) -> dict:
        scene = self.scene.to_dict()
        for container in scene["containers"]:
            used = self.container_usage(str(container["id"]))
            container["usedCapacity"] = used
            container["availableCapacity"] = max(0.0, float(container["capacity"]) - used)
        return scene

    def processes_at(self, location: str) -> list[dict]:
        capabilities = {
            capability
            for entity in self.at_location(location)
            for capability in entity.capabilities
            if entity.availability == "available"
        }
        return [
            dict(process) for process in self.processes.values()
            if (not process.get("required_capability")
                or process.get("required_capability") in capabilities)
            and (not process.get("required_anchor_capability")
                 or self.scene.anchors_at(location, str(process["required_anchor_capability"])))
        ]

    def world_state(self, agent_ids: list[str]) -> dict:
        locations = {}
        for location_id in LOCATION_MAP:
            locations[location_id] = {
                "entities": [item.to_dict() for item in self.at_location(location_id)],
                "processes": self.processes_at(location_id),
            }
        return {
            "locations": locations,
            "inventories": {agent_id: self.inventory(agent_id) for agent_id in agent_ids},
        }

    def reset(self):
        self.resources.clear()
        self.processes = {
            str(raw["id"]): dict(raw) for raw in self.world_pack.processes
        }
        for raw in [*self.world_pack.resources, *self.world_pack.entities]:
            self.add(Resource(
                id=str(raw["id"]),
                kind=str(raw.get("kind", raw.get("type", "object"))),
                location=str(raw["location"]),
                quantity=float(raw.get("quantity", 1)),
                properties=dict(raw.get("properties", {})),
                owner_id=raw.get("owner_id"),
                availability=str(raw.get("availability", "available")),
                known_by=[str(item) for item in raw.get("known_by", [])],
                name=str(raw.get("name", raw["id"])),
                entity_type=str(raw.get("entity_type", "resource")),
                portable=bool(raw.get("portable", raw.get("entity_type", "resource") == "resource")),
                capabilities=[str(item) for item in raw.get("capabilities", [])],
                components=dict(raw.get("components", {})),
                state=dict(raw.get("state", {})),
                anchor_id=str(raw.get("anchor", raw.get("anchor_id", ""))),
                container_id=str(raw.get("container", raw.get("container_id", ""))),
            ))

    def reset_default_town(self):
        self.reset()


class InteractionEngine:
    def __init__(self, resources: ResourceStore):
        self.resources = resources
        self.scene = resources.scene

    # What each action needs to name, and what the validator says when it is
    # absent: the same words, because a step refused here and a step refused on
    # arrival are refused for the same reason.
    REQUIRED_REFERENCES = {
        "move": (("location", "目标地点不存在"),),
        "take": (("resource_id", "资源不存在"),),
        "put": (("resource_id", "资源不存在"),),
        "transfer": (("resource_id", "资源不存在"),),
        "use_resource": (("resource_id", "资源不存在"),),
        "request_service": (("resource_id", "请求服务必须指定真实商品"),),
        "operate": (("process_id", "过程不存在"),),
        "produce": (("product_kind", "没有声明产物类型"),),
        "communicate": (("target", "对话对象不存在"),),
    }

    def missing_references(self, action: dict, agent, engine) -> list[dict]:
        """References this action needs that are absent from it or unknown here.

        Position-independent on purpose: it judges a step written for a place
        the agent has not reached yet, so a plan is refused before it is
        committed instead of failing once the agent arrives. Reasons match what
        execution-time validation says, so a step is refused for the same cause
        whether it is planned or attempted.
        """
        kind = str(action.get("interaction_type") or action.get("action") or "")
        problems: list[dict] = []

        def unknown(field: str, value, reason: str) -> None:
            problems.append({"field": field, "value": str(value), "reason": reason})

        for field, reason in self.REQUIRED_REFERENCES.get(kind, ()):
            if field == "location":
                present = (action.get("location") or action.get("location_id")
                           or action.get("target_location"))
            else:
                present = action.get(field)
            if not str(present or "").strip():
                # The validator looks the empty id up and fails the same way.
                unknown(field, "", reason)
        location = str(action.get("location") or action.get("location_id")
                       or action.get("target_location") or "")
        if kind == "move" and location and location not in LOCATION_MAP:
            unknown("location", location, "目标地点不存在")
        resource_id = str(action.get("resource_id", ""))
        if resource_id and resource_id not in self.resources.resources:
            unknown("resource_id", resource_id, "资源不存在")
        process_id = str(action.get("process_id", ""))
        if process_id and process_id not in self.resources.processes:
            unknown("process_id", process_id, "过程不存在")
        anchor_id = str(action.get("anchor_id") or action.get("target_anchor_id") or "")
        if anchor_id and anchor_id not in self.scene.anchors:
            unknown("anchor_id", anchor_id, "交互锚点不存在")
        container_id = str(action.get("container_id") or action.get("destination_container_id") or "")
        if container_id and container_id not in self.scene.containers:
            unknown("container_id", container_id, "目标容器不存在")
        if kind == "communicate":
            target_id = str(action.get("target", ""))
            if target_id and target_id not in {item.id for item in engine.agents}:
                unknown("target", target_id, "对话对象不存在")
        if kind == "work_on_goal":
            goal_id = str(action.get("goal_id", ""))
            actionable = {
                goal.id for goal in agent.mental_state.goals
                if goal.status in {"proposed", "active", "blocked", "suspended"}
            }
            if goal_id and goal_id not in actionable:
                unknown("goal_id", goal_id, "目标不存在、已完成或不再可推进")
            try:
                progress = float(action["progress_delta"])
            except (KeyError, TypeError, ValueError):
                unknown("progress_delta", action.get("progress_delta", ""),
                        "推进目标必须声明progress_delta")
            else:
                if not 0 < progress <= 1:
                    unknown("progress_delta", progress, "progress_delta必须大于0且不超过1")
        return problems

    def observation_snapshot(self, agent, engine, target_id: str = "") -> dict:
        location = agent.state.current_location
        visible = self.resources.visible_at(location, agent.id)
        if target_id:
            visible = [item for item in visible if item.get("id") == target_id]
        nearby = sorted(
            [
                {"id": item.id, "status": item.state.status, "action": item.state.current_action}
                for item in engine.agents
                if item.id != agent.id and item.state.current_location == location
            ],
            key=lambda item: item["id"],
        )
        entities = sorted(
            [
                {
                    "id": item.get("id"), "name": item.get("name"),
                    "kind": item.get("kind"), "quantity": item.get("quantity"),
                    "availability": item.get("availability"),
                    "capabilities": sorted(item.get("capabilities", [])),
                    "state": item.get("state", {}),
                }
                for item in visible
            ],
            key=lambda item: str(item.get("id", "")),
        )
        processes = [] if target_id else sorted(
            [
                {"id": item.get("id"), "name": item.get("name", item.get("id"))}
                for item in self.resources.processes_at(location)
            ],
            key=lambda item: str(item.get("id", "")),
        )
        snapshot = {
            "location": location,
            "target_id": target_id,
            "entities": entities,
            "nearby_agents": nearby,
            "available_processes": processes,
            "weather": getattr(engine, "weather", {}),
        }
        encoded = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        snapshot["signature"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
        return snapshot

    def normalize(self, action: dict) -> dict:
        normalized = dict(action or {})
        interaction_type = str(normalized.get("interaction_type") or normalized.get("action") or "wait")
        aliases = {"act": "wait", "talk": "communicate", "reflect": "wait"}
        interaction_type = aliases.get(interaction_type, interaction_type)
        normalized["interaction_type"] = interaction_type
        normalized["action"] = "move" if interaction_type == "move" else "talk" if interaction_type == "communicate" else "act"
        normalized.setdefault("action_id", f"action_{uuid.uuid4().hex[:12]}")
        normalized.setdefault("content", normalized.get("description", ""))
        return normalized

    def validate(self, agent, action: dict, engine, option_id: str = "selected") -> FeasibilityResult:
        action = self.normalize(action)
        kind = action["interaction_type"]
        if kind == "move":
            raw_target = str(
                action.get("location") or action.get("location_id")
                or action.get("target_location") or ""
            )
            if raw_target and raw_target not in LOCATION_MAP:
                matching_ids = [
                    location.id for location in LOCATION_MAP.values()
                    if location.name == raw_target
                ]
                raw_target = matching_ids[0] if len(matching_ids) == 1 else raw_target
            if not raw_target:
                supported_ids = {
                    str(item) for item in action.get("supports_goal_ids", [])
                }
                goal_locations = {
                    str(condition.get("location", ""))
                    for goal in agent.mental_state.goals
                    if goal.id in supported_ids
                    for condition in goal.success_conditions
                    if condition.get("type") == "at_location" and condition.get("location")
                }
                if not goal_locations:
                    commitment = agent.get_current_commitment(
                        engine.day, engine.hour, engine.minute
                    )
                    if commitment and commitment.get("status") != "completed":
                        target = str(commitment.get("location", ""))
                        if target and target != agent.state.current_location:
                            goal_locations.add(target)
                if len(goal_locations) == 1:
                    raw_target = next(iter(goal_locations))
            if raw_target:
                action["location"] = raw_target
        reasons: list[str] = []
        effects: list[dict] = []
        duration = max(5, int(action.get("duration_minutes") or 15))
        location = agent.state.current_location
        inventory = f"agent:{agent.id}"
        anchor_id = str(action.get("anchor_id") or action.get("target_anchor_id") or "")
        anchor = self.scene.anchors.get(anchor_id) if anchor_id else None
        if anchor_id and not anchor:
            reasons.append("交互锚点不存在")
        elif anchor and anchor.location_id != location:
            reasons.append("必须到达交互锚点所在地点才能使用")
        elif anchor:
            action["anchor_id"] = anchor.id

        if kind not in INTERACTION_TYPES:
            reasons.append(f"未知交互类型: {kind}")
        if agent.state.status != "IDLE":
            reasons.append(f"当前状态{agent.state.status}不能开始新行动")
        reasons.extend(self._infrastructure_reasons(agent, kind, action, engine))

        if kind == "move":
            target_location = str(action.get("location", ""))
            if target_location not in LOCATION_MAP:
                reasons.append("目标地点不存在")
            elif not can_enter(agent.id, target_location):
                reasons.append("没有进入目标地点的权限")
            elif target_location == location:
                reasons.append("角色已经在目标地点")
            else:
                cx, cy = location_center(target_location)
                path = bfs_path(agent.state.x, agent.state.y, cx, cy, agent_id=agent.id, target_location_id=target_location)
                if not path:
                    reasons.append("当前没有可达路径")
                else:
                    duration = max(5, len(path) * 5)
                    effects.append({"type": "location_change", "location": target_location})
        elif kind == "consume":
            resource = self._resolve_accessible_resource(action, location, agent.id, "food")
            if not resource:
                reasons.append("当前位置或携带物中没有可消费资源")
            else:
                quantity = max(0.01, float(action.get("quantity", 1)))
                service_facility_id = self._service_facility_id(resource)
                if (
                    service_facility_id
                    and resource.location == location
                    and not self._is_service_staff(agent.id, service_facility_id, location, engine)
                ):
                    reasons.append("店内商品需要先向服务台请求，不能直接取用")
                elif resource.quantity < quantity:
                    reasons.append("资源数量不足")
                else:
                    action.update(resource_id=resource.id, quantity=quantity)
                    effects.extend([
                        {"type": "need_change", "need": "hunger", "delta": -float(resource.properties.get("nutrition", 0))},
                        {"type": "need_change", "need": "energy", "delta": float(resource.properties.get("energy", 0))},
                        {"type": "resource_change", "resource_id": resource.id, "delta": -quantity},
                    ])
        elif kind == "request_service":
            resource = self.resources.get(str(action.get("resource_id", "")))
            if not resource:
                reasons.append("请求服务必须指定真实商品")
            elif resource.location != location:
                reasons.append("商品不在当前服务场所")
            elif resource.availability != "available" or resource.quantity <= 0:
                reasons.append("请求的商品当前不可用")
            else:
                facility_id = self._service_facility_id(resource)
                facility = self.resources.get(facility_id) if facility_id else None
                if not facility or facility.location != location:
                    reasons.append("当前商品没有可用的服务台")
                elif facility.availability != "available" or facility.state.get("operational") is False:
                    reasons.append("服务台当前未开放")
                elif not self._has_observed_staff(agent, location, facility_id, engine):
                    reasons.append("尚未观察到可服务工作人员；请先查看店内情况")
                else:
                    staff = self._available_service_staff(facility_id, location, engine)
                    if not staff:
                        reasons.append("观察到服务台当前无人值守")
                    else:
                        quantity = max(0.01, float(action.get("quantity", 1)))
                        if resource.quantity < quantity:
                            reasons.append("商品库存不足")
                        else:
                            action.update(
                                resource_id=resource.id, service_facility_id=facility_id,
                                service_staff_id=staff[0].id, quantity=quantity,
                            )
                            duration = max(5, min(duration, 10))
                            effects.extend([
                                {
                                    "type": "resource_move", "resource_id": resource.id,
                                    "quantity": quantity, "destination": inventory,
                                },
                                {
                                    "type": "service_fulfilled", "facility_id": facility_id,
                                    "staff_id": staff[0].id, "resource_id": resource.id,
                                },
                            ])
        elif kind == "rest":
            quality = max(0.1, min(1.0, float(action.get("quality", 0.5))))
            effects.append({"type": "need_change", "need": "energy", "delta": round(duration * quality * 0.5, 2)})
        elif kind == "communicate":
            target_id = str(action.get("target", ""))
            target = next((item for item in engine.agents if item.id == target_id), None)
            if not target:
                reasons.append("对话对象不存在")
            elif target.state.current_location != location:
                reasons.append("面对面对话要求双方在同一地点")
            else:
                effects.append({"type": "social_interaction", "target": target_id})
        elif kind in {"take", "put", "transfer"}:
            resource = self.resources.get(str(action.get("resource_id", "")))
            quantity = max(0.01, float(action.get("quantity", 1)))
            if not resource:
                reasons.append("资源不存在")
            elif resource.availability != "available" or resource.quantity < quantity:
                reasons.append("资源当前不可用或数量不足")
            elif kind == "take" and resource.location != location:
                reasons.append("只能拿取当前位置的资源")
            elif kind == "take" and not resource.portable:
                reasons.append("该实体不可携带")
            elif kind == "put" and resource.location != inventory:
                reasons.append("只能放下自己携带的资源")
            else:
                destination = inventory if kind == "take" else location if kind == "put" else str(action.get("destination", ""))
                container_id = str(action.get("container_id") or action.get("destination_container_id") or "")
                if kind == "transfer" and destination in self.scene.containers:
                    container_id, destination = destination, self.scene.containers[destination].location_id
                if kind == "transfer" and not destination:
                    target_id = str(action.get("target", ""))
                    destination = f"agent:{target_id}" if target_id else location
                container = self.scene.containers.get(container_id) if container_id else None
                if container_id and not container:
                    reasons.append("目标容器不存在")
                elif container and container.location_id != location:
                    reasons.append("目标容器不在当前位置")
                elif container and container.accepts_kinds and resource.kind not in container.accepts_kinds:
                    reasons.append("目标容器不接收该资源类型")
                elif container and container_id != resource.container_id and self.resources.container_usage(container_id) + quantity > container.capacity:
                    reasons.append("目标容器容量不足")
                if resource.location not in {location, inventory}:
                    reasons.append("资源不在角色可操作范围")
                elif not reasons:
                    action.update(quantity=quantity, destination=destination, destination_container_id=container_id)
                    effects.append({
                        "type": "resource_move", "resource_id": resource.id, "quantity": quantity,
                        "destination": destination, "destination_container_id": container_id,
                    })
        elif kind == "operate":
            process = self.resources.processes.get(str(action.get("process_id", "")))
            if not process:
                reasons.append("过程不存在")
            else:
                device = self._resolve_device(action, location, process)
                required_anchor_capability = str(process.get("required_anchor_capability", ""))
                process_anchor = self._resolve_anchor(action, location, required_anchor_capability)
                if not device:
                    reasons.append("当前位置没有满足过程能力的可用设备")
                elif required_anchor_capability and not process_anchor:
                    reasons.append("当前位置没有满足过程锚点能力的可用交互点")
                else:
                    state_requirements = dict(process.get("state_requirements", {}))
                    for key, expected in state_requirements.items():
                        if device.state.get(key) != expected:
                            reasons.append(f"设备状态不满足条件: {key}")
                    input_effects, input_reasons = self._resolve_process_inputs(process, location, agent.id)
                    reasons.extend(input_reasons)
                    if not reasons:
                        duration = max(5, int(process.get("duration_minutes", duration)))
                        action.update(device_id=device.id, process_id=str(process["id"]))
                        if process_anchor:
                            action["anchor_id"] = process_anchor.id
                        effects.extend(input_effects)
                        for output in process.get("outputs", []):
                            effects.append({
                                "type": "resource_add",
                                "resource_id": output.get("resource_id"),
                                "kind": str(output.get("kind", "object")),
                                "name": str(output.get("name", output.get("kind", "object"))),
                                "quantity": float(output.get("quantity", 1)),
                                "destination": inventory if output.get("destination") == "inventory" else location,
                                "properties": dict(output.get("properties", {})),
                            })
                        effects.append({"type": "entity_used", "entity_id": device.id, "process_id": process["id"]})
                        if process_anchor:
                            effects.append({"type": "anchor_used", "anchor_id": process_anchor.id})
        elif kind == "use_resource":
            resource = self.resources.get(str(action.get("resource_id", "")))
            required = str(action.get("capability", ""))
            if not resource:
                reasons.append("资源不存在")
            elif resource.location not in {location, inventory}:
                reasons.append("资源不在角色可操作范围")
            elif resource.availability != "available" or resource.quantity <= 0:
                reasons.append("资源当前不可用")
            elif required and required not in resource.capabilities:
                reasons.append("实体不具备所需能力")
            else:
                effects.append({"type": "entity_used", "entity_id": resource.id, "capability": required})
        elif kind == "produce":
            product_kind = str(action.get("product_kind", "")).strip()
            if not product_kind:
                reasons.append("没有声明产物类型")
            else:
                effects.append({
                    "type": "resource_add", "kind": product_kind,
                    "name": str(action.get("product_name", product_kind)),
                    "quantity": max(0.01, float(action.get("quantity", 1))),
                    "destination": location, "properties": dict(action.get("properties", {})),
                })
        elif kind == "work_on_goal":
            goal_id = str(action.get("goal_id", ""))
            known_goals = {
                goal.id for goal in agent.mental_state.goals
                if goal.status in {"proposed", "active", "blocked", "suspended"}
            }
            if not goal_id:
                reasons.append("推进目标必须引用当前心智状态中的goal_id")
            elif goal_id not in known_goals:
                reasons.append("目标不存在、已完成或不再可推进")
            try:
                progress_delta = float(action["progress_delta"])
            except (KeyError, TypeError, ValueError):
                progress_delta = 0.0
                reasons.append("推进目标必须声明progress_delta")
            if not 0 < progress_delta <= 1:
                reasons.append("progress_delta必须大于0且不超过1")
            if not reasons:
                effects.append({"type": "goal_progress", "goal_id": goal_id, "delta": progress_delta})
        elif kind == "inspect":
            target_id = str(action.get("target_id", ""))
            snapshot = self.observation_snapshot(agent, engine, target_id)
            if target_id and not snapshot["entities"]:
                reasons.append("观察目标不在当前位置或对当前角色不可见")
            if self.inspect_is_redundant(agent, engine, target_id, location):
                reasons.append("近期已经观察过同一目标，环境没有变化")
            action.update(target_id=target_id, observation_signature=snapshot["signature"])
            effects.append({"type": "observation", "snapshot": snapshot})
        elif kind == "wait":
            effects.append({"type": "time_elapsed", "location": location})

        return FeasibilityResult(
            option_id=option_id, feasible=not reasons, reasons=reasons,
            estimated_minutes=duration, expected_effects=effects, resolved_action=action,
        )

    # Which capabilities the town's utilities provide. A failed service stops
    # the activities that genuinely depend on it, indoors: a blackout does not
    # close an open-air stall, and a broken water main does not stop a chat.
    SERVICE_CAPABILITIES = {
        "power": {"service"},
        "water": {"prepare", "process"},
    }
    # Being somewhere is not the same as using its equipment: a blackout does
    # not stop people sitting in the cafe, only being served there.
    SERVICE_DEPENDENT_ACTIONS = {"request_service", "use_resource", "operate", "produce"}

    def _infrastructure_reasons(self, agent, kind: str, action: dict, engine) -> list[str]:
        """Report world state that makes the action impossible right now."""
        infra = getattr(engine, "infrastructure", {}) or {}
        if not isinstance(infra, dict) or not infra:
            return []
        target = str(action.get("location", "")) or agent.state.current_location
        if infra.get("road_main") == "closed" and target == "road_main":
            return [f"{location_name('road_main')}当前封闭"]
        if kind not in self.SERVICE_DEPENDENT_ACTIONS:
            return []
        blocked: set[str] = set()
        labels: list[str] = []
        for service, capabilities in self.SERVICE_CAPABILITIES.items():
            if infra.get(service, "normal") == "normal":
                continue
            blocked |= capabilities
            labels.append(service)
        if not blocked:
            return []
        for anchor in self.scene.anchors.values():
            if anchor.location_id != target:
                continue
            if not (blocked & set(anchor.capabilities)):
                continue
            if is_outdoor(target):
                continue
            return [f"服务中断（{'、'.join(labels)}），这里暂时无法进行该活动"]
        return []

    def apply_effects(self, agent, action: dict, effects: list[dict], sim_timestamp: int | None = None) -> list[dict]:
        applied = []
        for effect in effects:
            effect = dict(effect)
            effect_type = effect.get("type")
            if effect_type == "need_change":
                need = str(effect.get("need", ""))
                before = float(agent.state.needs.get(need, 0))
                after = max(0.0, min(100.0, before + float(effect.get("delta", 0))))
                agent.state.needs[need] = after
                effect.update(before=before, after=after)
            elif effect_type == "resource_change":
                resource = self.resources.get(str(effect.get("resource_id", "")))
                if resource:
                    before = resource.quantity
                    resource.quantity = max(0.0, before + float(effect.get("delta", 0)))
                    resource.availability = "depleted" if resource.quantity <= 0 else "available"
                    effect.update(before=before, after=resource.quantity)
            elif effect_type == "resource_move":
                moved = self._move_quantity(
                    str(effect.get("resource_id", "")), float(effect.get("quantity", 0)),
                    str(effect.get("destination", "")),
                    str(effect.get("destination_container_id", "")),
                )
                effect.update(moved_resource_id=moved.id if moved else "")
            elif effect_type == "resource_add":
                resource = self._add_quantity(
                    resource_id=effect.get("resource_id"), kind=str(effect.get("kind", "object")),
                    name=str(effect.get("name", effect.get("kind", "object"))),
                    quantity=float(effect.get("quantity", 0)), destination=str(effect.get("destination", "")),
                    properties=dict(effect.get("properties", {})),
                )
                effect.update(resource_id=resource.id, after=resource.quantity)
            elif effect_type == "anchor_used":
                anchor = self.scene.anchors.get(str(effect.get("anchor_id", "")))
                if anchor:
                    effect["location"] = anchor.location_id
                    effect["capabilities"] = list(anchor.capabilities)
            elif effect_type == "observation":
                snapshot = dict(effect.get("snapshot", {}))
                if snapshot:
                    observed_at = int(sim_timestamp if sim_timestamp is not None else effect.get("observed_at", 0))
                    self._store_observation(agent, snapshot, observed_at)
                    effect["observed_at"] = observed_at
            elif effect_type == "social_interaction":
                # Talking to someone is what satisfies the social need; without
                # this branch the effect was produced and silently dropped, so
                # the need could only ever decay.
                delta = float(effect.get("delta", config.SOCIAL_INTERACTION_RECOVERY))
                before = float(agent.state.needs.get("social", 0))
                after = max(0.0, min(100.0, before + delta))
                agent.state.needs["social"] = after
                effect.update(before=before, after=after)
            elif effect_type == "goal_progress":
                goal_id = str(effect.get("goal_id", ""))
                for goal in agent.mental_state.goals:
                    if goal.id == goal_id:
                        before = goal.progress
                        goal.progress = max(0.0, min(1.0, before + float(effect.get("delta", 0))))
                        if goal.progress >= 1:
                            goal.status = "completed"
                        effect.update(
                            before=before,
                            after=goal.progress,
                            completed=goal.status == "completed",
                        )
            applied.append(effect)
        return applied

    def _service_facility_id(self, resource: Resource) -> str:
        return str(resource.properties.get("service_facility_id", ""))

    def _available_service_staff(self, facility_id: str, location: str, engine) -> list:
        staff_ids = {
            duty.agent_id for duty in self.scene.duties
            if duty.facility_id == facility_id and "service" in duty.capabilities
        }
        return [
            person for person in engine.agents
            if person.id in staff_ids and person.state.current_location == location
            and person.state.status != "MOVING"
        ]

    def _is_service_staff(self, agent_id: str, facility_id: str, location: str, engine) -> bool:
        return any(
            person.id == agent_id
            for person in self._available_service_staff(facility_id, location, engine)
        )

    def inspect_is_redundant(self, agent, engine, target_id: str, location: str) -> bool:
        """True when the agent already looked at this, here, and nothing changed."""
        previous = getattr(agent, "_last_observation", None)
        if not isinstance(previous, dict):
            return False
        if previous.get("location") != location or previous.get("target_id", "") != target_id:
            return False
        if engine.get_sim_timestamp() - int(previous.get("observed_at", 0)) >= 30:
            return False
        current = self.observation_snapshot(agent, engine, target_id)
        return previous.get("signature") == current.get("signature")

    def _store_observation(self, agent, snapshot: dict, sim_timestamp: int) -> dict:
        record = {**snapshot, "observed_at": int(sim_timestamp)}
        agent._last_observation = record
        return record

    def record_observation(self, agent, engine, target_id: str = "") -> dict:
        """Write down what this agent can see, here and now.

        Seeing the room it just walked into is bookkeeping, not a decision, so
        the engine does it. ``inspect`` remains the deliberate act of looking at
        something in particular.
        """
        return self._store_observation(
            agent, self.observation_snapshot(agent, engine, target_id),
            engine.get_sim_timestamp(),
        )

    def service_needs_a_look(self, agent, location: str, resource, engine) -> bool:
        """True when ordering this item would be refused for lack of a fresh look.

        Ordering at a counter requires having observed its staff within the last
        30 sim-minutes, and only an explicit ``inspect`` records an observation —
        walking in does not. Asked at planning time so a step that can only be
        refused is not written into a plan.
        """
        facility_id = self._service_facility_id(resource)
        if not facility_id:
            return False        # not counter goods: other reasons decide this step
        return not self._has_observed_staff(agent, location, facility_id, engine)

    def _has_observed_staff(self, agent, location: str, facility_id: str, engine) -> bool:
        observation = getattr(agent, "_last_observation", None)
        if not isinstance(observation, dict) or observation.get("location") != location:
            return False
        observed_at = int(observation.get("observed_at", -10**9))
        if engine.get_sim_timestamp() - observed_at > 30:
            return False
        observed_ids = {
            str(item.get("id", "")) for item in observation.get("nearby_agents", [])
        }
        return any(
            person.id in observed_ids
            for person in self._available_service_staff(facility_id, location, engine)
        )

    def _resolve_accessible_resource(self, action: dict, location: str, agent_id: str, kind: str) -> Resource | None:
        resource_id = str(action.get("resource_id", ""))
        if resource_id:
            resource = self.resources.get(resource_id)
            return resource if resource and resource.location in {location, f"agent:{agent_id}"} else None
        resources = self.resources.accessible_to(location, agent_id, kind=kind)
        return resources[0] if resources else None

    def _resolve_anchor(self, action: dict, location: str, required_capability: str = ""):
        requested_id = str(action.get("anchor_id") or action.get("target_anchor_id") or "")
        if requested_id:
            anchor = self.scene.anchors.get(requested_id)
            if anchor and anchor.location_id == location and (not required_capability or required_capability in anchor.capabilities):
                return anchor
            return None
        candidates = self.scene.anchors_at(location, required_capability)
        return candidates[0] if candidates else None

    def _resolve_device(self, action: dict, location: str, process: dict) -> Resource | None:
        device_id = str(action.get("device_id", ""))
        required = str(process.get("required_capability", ""))
        candidates = [self.resources.get(device_id)] if device_id else self.resources.at_location(location)
        return next((
            item for item in candidates if item and item.location == location
            and item.availability == "available"
            and (not required or required in item.capabilities)
        ), None)

    def _resolve_process_inputs(self, process: dict, location: str, agent_id: str) -> tuple[list[dict], list[str]]:
        effects: list[dict] = []
        reasons: list[str] = []
        reserved: dict[str, float] = {}
        for requirement in process.get("inputs", []):
            kind = str(requirement.get("kind", ""))
            needed = max(0.01, float(requirement.get("quantity", 1)))
            candidates = self.resources.accessible_to(location, agent_id, kind=kind)
            remaining = needed
            for resource in candidates:
                available = resource.quantity - reserved.get(resource.id, 0)
                amount = min(remaining, max(0.0, available))
                if amount > 0:
                    reserved[resource.id] = reserved.get(resource.id, 0) + amount
                    effects.append({"type": "resource_change", "resource_id": resource.id, "delta": -amount})
                    remaining -= amount
                if remaining <= 0:
                    break
            if remaining > 0:
                reasons.append(f"缺少过程输入: {kind} {round(remaining, 2)}")
        return effects, reasons

    def _move_quantity(self, resource_id: str, quantity: float, destination: str,
                       destination_container_id: str = "") -> Resource | None:
        resource = self.resources.get(resource_id)
        if not resource or quantity <= 0 or resource.quantity < quantity or not destination:
            return None
        if resource.quantity == quantity:
            resource.location = destination
            resource.container_id = destination_container_id
            return resource
        resource.quantity -= quantity
        return self._add_quantity(
            None, resource.kind, resource.name, quantity, destination, resource.properties,
            container_id=destination_container_id,
        )

    def _add_quantity(self, resource_id, kind: str, name: str, quantity: float,
                      destination: str, properties: dict, container_id: str = "") -> Resource:
        existing = self.resources.get(str(resource_id)) if resource_id else None
        if not existing:
            existing = next((
                item for item in self.resources.resources.values()
                if item.location == destination and item.container_id == container_id
                and item.kind == kind and item.properties == properties
                and item.entity_type == "resource"
            ), None)
        if existing:
            existing.quantity += quantity
            existing.availability = "available"
            return existing
        new_id = str(resource_id or f"runtime_{kind}_{uuid.uuid4().hex[:10]}")
        return self.resources.add(Resource(
            id=new_id, kind=kind, name=name, location=destination,
            quantity=quantity, properties=properties, portable=True,
            container_id=container_id,
        ))
