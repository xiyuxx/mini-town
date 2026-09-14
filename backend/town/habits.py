"""Bounded, explainable behavior inertia for voluntary routine choices."""

from dataclasses import dataclass


@dataclass
class Habit:
    key: str
    agent_id: str
    block_id: str
    location_id: str
    activity: str
    strength: float = 0.0
    voluntary_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_used_at: int = 0

    def to_dict(self) -> dict:
        return {
            "key": self.key, "agentId": self.agent_id, "blockId": self.block_id,
            "locationId": self.location_id, "activity": self.activity,
            "strength": round(self.strength, 3), "voluntaryCount": self.voluntary_count,
            "successCount": self.success_count, "failureCount": self.failure_count,
            "lastUsedAt": self.last_used_at,
        }


class HabitStore:
    def __init__(self, max_per_agent: int = 8):
        self.max_per_agent = max_per_agent
        self._habits: dict[str, Habit] = {}

    def clear(self) -> None:
        self._habits.clear()

    def restore(self, habit: Habit) -> Habit:
        """Restore a durable habit snapshot without changing its history."""
        self._habits[habit.key] = habit
        self._trim(habit.agent_id)
        return habit

    @staticmethod
    def key_for(agent_id: str, block_id: str, location_id: str, activity: str) -> str:
        return f"{agent_id}:{block_id}:{location_id}:{activity.strip()}"

    def strength_for(self, agent_id: str, block_id: str, location_id: str, activity: str) -> float:
        key = self.key_for(agent_id, block_id, location_id, activity)
        return self._habits.get(key, Habit(key, agent_id, block_id, location_id, activity)).strength

    def record(self, agent_id: str, action: dict, now: int, success: bool) -> Habit | None:
        if action.get("source") != "routine_block":
            return None
        block_id = str(action.get("routine_block_id", ""))
        location_id = str(action.get("location", ""))
        activity = str(action.get("content", "")).strip()
        if not block_id or not location_id or not activity:
            return None
        key = self.key_for(agent_id, block_id, location_id, activity)
        habit = self._habits.get(key)
        if habit is None:
            habit = Habit(key, agent_id, block_id, location_id, activity)
            self._habits[key] = habit
        habit.voluntary_count += 1
        habit.last_used_at = now
        if success:
            habit.success_count += 1
            habit.strength = min(1.0, habit.strength * 0.92 + 0.14)
        else:
            habit.failure_count += 1
            habit.strength = max(0.0, habit.strength * 0.7 - 0.08)
        self._trim(agent_id)
        return habit

    def decay(self, agent_id: str, now: int) -> None:
        for habit in list(self._habits.values()):
            if habit.agent_id != agent_id:
                continue
            age_days = max(0.0, (now - habit.last_used_at) / 1440)
            if age_days > 0:
                habit.strength *= 0.97 ** age_days
            if habit.strength < 0.03 and age_days > 7:
                self._habits.pop(habit.key, None)

    def for_agent(self, agent_id: str) -> list[Habit]:
        return sorted(
            (habit for habit in self._habits.values() if habit.agent_id == agent_id),
            key=lambda habit: (-habit.strength, -habit.last_used_at, habit.key),
        )

    def _trim(self, agent_id: str) -> None:
        habits = self.for_agent(agent_id)
        for habit in habits[self.max_per_agent:]:
            self._habits.pop(habit.key, None)
