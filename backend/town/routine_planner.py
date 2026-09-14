"""Deterministic task selection for the simulation hot path."""

from .candidates import build_routine_candidates
from .tasks import Task, TaskStore


class RoutinePlanner:
    """Choose visible actions from generic task and world state.

    This planner intentionally knows no world-specific building, role, or item
    names. Scenario data supplies locations, task payloads, and resource kinds.
    """

    def __init__(self, tasks: TaskStore):
        self.tasks = tasks

    def next_action(self, agent, engine) -> dict | None:
        now = engine.get_sim_timestamp()
        self.tasks.project_schedule(agent, engine.day, now)

        need_task = self._urgent_need_task(agent, engine, now)
        if need_task:
            return self._task_action(need_task, agent, engine)

        current_slot = agent._get_schedule_item(engine.hour, engine.minute)
        if current_slot and current_slot.is_flexible_slot:
            flexible_candidates = build_routine_candidates(agent, engine)
            if flexible_candidates:
                return dict(flexible_candidates[0].action)

        candidates = self.tasks.available_for(agent.id, now)
        if not candidates:
            return None
        task = max(candidates, key=lambda item: self._score(item, agent, engine, now))
        return self._task_action(task, agent, engine)

    def _urgent_need_task(self, agent, engine, now: int) -> Task | None:
        needs = agent.state.needs
        location = agent.state.current_location
        if needs.get("hunger", 0) >= 80:
            food = engine.resources.accessible_to(location, agent.id, kind="food")
            if food:
                return self.tasks.create_need_task(
                    agent.id, "补充食物", now, location, "consume", 0.97,
                    {"resource_id": food[0].id, "quantity": 1},
                )
        if needs.get("energy", 100) <= 20:
            return self.tasks.create_need_task(
                agent.id, "恢复精力", now, location, "rest", 0.92,
                {"quality": 0.5, "duration_minutes": 30},
            )
        return None

    @staticmethod
    def _score(task: Task, agent, engine, now: int) -> float:
        score = task.priority
        if task.deadline_at is not None:
            window = max(1, task.deadline_at - task.earliest_at)
            remaining = task.deadline_at - now
            pressure = max(0.0, min(1.0, 1.0 - remaining / window))
            score += pressure * 0.55
        if task.location_id and task.location_id != agent.state.current_location:
            score += 0.08
        if task.status == "active":
            score += 0.12
        return score

    @staticmethod
    def _task_action(task: Task, agent, engine) -> dict:
        payload = dict(task.payload)
        commitment = agent.get_current_commitment(engine.day, engine.hour, engine.minute)
        action = {
            "interaction_type": task.interaction_type,
            "content": task.title,
            "location": task.location_id or agent.state.current_location,
            "duration_minutes": task.duration_minutes,
            "task_id": task.id,
            "source": "routine",
            **payload,
        }
        if task.source == "schedule" and commitment:
            action["commitment"] = commitment
        return action
