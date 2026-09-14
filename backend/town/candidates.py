"""Deterministic candidate generation for flexible routine blocks."""

from dataclasses import dataclass

from .world import LOCATION_MAP, can_enter


@dataclass(frozen=True)
class ActionCandidate:
    id: str
    action: dict
    score: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "action": dict(self.action),
            "score": round(self.score, 3),
            "reasons": list(self.reasons),
        }


def _distance(agent, location_id: str) -> int:
    location = LOCATION_MAP.get(location_id)
    if not location:
        return 999
    return abs(agent.state.x - location.center[0]) + abs(agent.state.y - location.center[1])


def build_routine_candidates(agent, engine, limit: int = 6) -> list[ActionCandidate]:
    """Create explainable, feasible-location candidates for the active block.

    Scores are deliberately only a deterministic baseline. Future habit,
    relationship, and memory features add terms without allowing the LLM to
    invent inaccessible actions.
    """
    blocks = agent.get_routine_blocks_for_time(engine.hour, engine.minute)
    candidates: list[ActionCandidate] = []
    for block in blocks:
        locations = list(block.candidate_locations) or [agent.state.current_location]
        activities = list(block.candidate_activities) or [block.intent]
        for location_id in locations:
            if location_id not in LOCATION_MAP or not can_enter(agent.id, location_id):
                continue
            distance = _distance(agent, location_id)
            people = [
                other for other in engine.agents
                if other.id != agent.id and other.state.current_location == location_id
            ]
            nearby = len(people)
            ties = [engine.relationship_store.rapport(agent.id, other.id) for other in people]
            strongest = max((tie.familiarity for tie in ties), default=0.0)
            liked = max((tie.affinity for tie in ties), default=0.0)
            strangers = sum(1 for tie in ties if tie.familiarity <= 0.0)
            for activity_index, activity in enumerate(activities):
                travel_cost = min(0.35, distance / 60)
                social_need = agent.social_urgency()
                social_value = min(0.12, nearby * 0.04) * (0.5 + social_need)
                # Who is already there decides where an agent goes: a known and
                # liked face pulls, and when starved for company an unknown one
                # becomes a reason to go somewhere new.
                rapport_value = min(0.14, strongest / 20 * 0.08 + max(0.0, liked) / 10 * 0.06)
                stranger_pull = 0.06 if social_need > 0.5 and strangers else 0.0
                habit_strength = engine.habits.strength_for(
                    agent.id, block.id, location_id, activity,
                )
                score = max(0.0, min(
                    1.0, block.priority + social_value + rapport_value + stranger_pull
                    + habit_strength * 0.18 - travel_cost,
                ))
                action = {
                    "interaction_type": "wait",
                    "content": activity,
                    "location": location_id,
                    "duration_minutes": 15,
                    "source": "routine_block",
                    "routine_block_id": block.id,
                }
                reasons = [f"当前处于{block.intent}时间区块"]
                if location_id == agent.state.current_location:
                    reasons.append("无需移动")
                else:
                    reasons.append(f"预计移动距离{distance}")
                if nearby:
                    reasons.append(f"该地点有{nearby}位其他角色")
                if strongest > 0:
                    reasons.append(f"那里有熟人（熟悉度{strongest:.1f}）")
                elif strangers and social_need > 0.5:
                    reasons.append("那里有还没打过交道的人")
                if habit_strength >= 0.08:
                    reasons.append(f"这是最近反复选择的活动（惯性{habit_strength:.2f}）")
                candidates.append(ActionCandidate(
                    id=f"{block.id}:{location_id}:{activity_index}",
                    action=action,
                    score=score,
                    reasons=tuple(reasons),
                ))
    candidates.sort(key=lambda candidate: (-candidate.score, candidate.id))
    return candidates[:limit]
