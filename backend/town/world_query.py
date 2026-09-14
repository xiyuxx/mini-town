"""Read-only, task-scoped queries over the current world state."""

from .world import LOCATION_MAP, bfs_path, can_enter


class WorldQuery:
    """Resolve an intention into a bounded view of the shared world."""

    def __init__(self, world_pack, resources):
        self.world_pack = world_pack
        self.resources = resources

    def _facility_view(self, location_id: str) -> dict:
        """Ids a plan may cite for a place it intends to go to.

        A plan is written where the agent stands but carried out elsewhere, and
        every other part of the context describes only the current location. No
        destination ids means the model can only invent device, anchor, and
        process ids — and the step dies on arrival.
        """
        return {
            "entities": [
                {"id": item.id, "kind": item.kind, "name": item.name,
                 "capabilities": sorted(item.capabilities)}
                for item in self.resources.at_location(location_id)
            ],
            "processes": [
                {"id": str(process.get("id", "")), "name": str(process.get("name", "")),
                 "required_capability": str(process.get("required_capability", ""))}
                for process in self.resources.processes_at(location_id)
            ],
            "anchors": [
                {"id": anchor.id, "name": anchor.name,
                 "capabilities": sorted(anchor.capabilities)}
                for anchor in self.resources.scene.anchors_at(location_id)
            ],
        }

    def locations_for_intention(self, agent, engine, intention: dict | None = None,
                                limit: int = 6, with_facilities: bool = False) -> list[dict]:
        """Return accessible, reachable locations relevant to an intention."""
        intention = intention if isinstance(intention, dict) else {}
        interaction_type = str(intention.get("interaction_type", ""))
        target_location = str(intention.get("target_location", ""))
        target_agent_id = str(intention.get("target_agent_id", ""))
        resource_kind = str(intention.get("resource_kind", ""))
        target_agent = next(
            (item for item in engine.agents if item.id == target_agent_id), None
        )
        ranked: list[tuple[float, dict]] = []
        for location in LOCATION_MAP.values():
            if not can_enter(agent.id, location.id):
                continue
            is_current = location.id == agent.state.current_location
            path = [] if is_current else bfs_path(
                agent.state.x, agent.state.y, *location.center,
                agent_id=agent.id, target_location_id=location.id,
            )
            if not is_current and not path:
                continue

            distance = abs(agent.state.x - location.center[0]) + abs(
                agent.state.y - location.center[1]
            )
            score = 0.25 if is_current else 0.0
            reasons: list[str] = []
            if is_current:
                reasons.append("当前地点")
            if location.id == target_location:
                score += 3.0
                reasons.append("意图指定地点")
            if target_agent and target_agent.state.current_location == location.id:
                score += 2.0
                reasons.append("目标角色在此处")
            if resource_kind:
                matching = self.resources.at_location(location.id, kind=resource_kind)
                if matching:
                    score += 1.2
                    reasons.append("有相关可用资源")
            if interaction_type in {"operate", "produce", "use_resource"}:
                processes = self.resources.processes_at(location.id)
                if processes:
                    score += 0.8
                    reasons.append("有可用过程")
            if interaction_type in {"communicate", "request_service"}:
                nearby = sum(
                    1 for other in engine.agents
                    if other.id != agent.id and other.state.current_location == location.id
                )
                if nearby:
                    score += min(0.8, nearby * 0.2)
                    reasons.append(f"有{nearby}位其他角色")
            score -= min(0.8, distance / 80)
            ranked.append((score, {
                "id": location.id,
                "name": location.name,
                "type": location.type,
                "accessible": True,
                "reachable": True,
                "distance": distance,
                "estimated_travel_minutes": len(path) * 5,
                "relevance": round(max(0.0, min(1.0, score / 4)), 3),
                "reasons": reasons,
            }))

        ranked.sort(key=lambda item: (-item[0], item[1]["id"]))
        selected = [item for _, item in ranked[:max(1, limit)]]
        if target_location and target_location in LOCATION_MAP and can_enter(agent.id, target_location):
            target = next((item for item in selected if item["id"] == target_location), None)
            if target is None:
                target = self._location_view(agent, engine, target_location)
                if target["reachable"]:
                    selected[-1] = target
        if with_facilities:
            for entry in selected:
                entry["facilities"] = self._facility_view(entry["id"])
        return selected

    def visible_environment(self, agent, engine, target_id: str = "") -> dict:
        """Return the existing observation snapshot as a world interface."""
        return engine.interactions.observation_snapshot(agent, engine, target_id)

    def current_constraints(self, agent, engine) -> list[dict]:
        """Return code-owned constraints needed before choosing an action."""
        constraints = []
        commitment = agent.get_current_commitment(engine.day, engine.hour, engine.minute)
        if commitment and commitment.get("status") not in {"completed", "skipped"}:
            constraints.append({"type": "commitment", "value": commitment})
        if not can_enter(agent.id, agent.state.current_location):
            constraints.append({"type": "location_access", "location": agent.state.current_location})
        weather = getattr(engine, "weather", {})
        if isinstance(weather, dict) and float(weather.get("travel_cost", 0.0)) > 0:
            constraints.append({
                "type": "travel_cost",
                "value": float(weather.get("travel_cost", 0.0)),
                "condition": weather.get("condition", ""),
            })
        return constraints

    def _location_view(self, agent, engine, location_id: str) -> dict:
        location = LOCATION_MAP[location_id]
        is_current = location.id == agent.state.current_location
        path = [] if is_current else bfs_path(
            agent.state.x, agent.state.y, *location.center,
            agent_id=agent.id, target_location_id=location.id,
        )
        return {
            "id": location.id,
            "name": location.name,
            "type": location.type,
            "accessible": can_enter(agent.id, location.id),
            "reachable": is_current or bool(path),
            "distance": len(path),
            "estimated_travel_minutes": len(path) * 5,
            "relevance": 1.0,
            "reasons": ["意图指定地点"],
        }
