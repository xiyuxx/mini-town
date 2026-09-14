"""Scenario-neutral task state shared by schedules, duties, and world events."""

from dataclasses import asdict, dataclass, field


OPEN_TASK_STATES = {"planned", "active"}
TERMINAL_TASK_STATES = {"completed", "expired", "cancelled"}


@dataclass
class Task:
    id: str
    title: str
    assignee_id: str
    source: str
    location_id: str = ""
    interaction_type: str = "wait"
    duration_minutes: int = 15
    earliest_at: int = 0
    deadline_at: int | None = None
    priority: float = 0.5
    required_capabilities: list[str] = field(default_factory=list)
    payload: dict = field(default_factory=dict)
    status: str = "planned"
    created_at: int = 0
    started_at: int | None = None
    completed_at: int | None = None
    blocking_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class TaskStore:
    """In-memory task board with deterministic schedule projection.

    A WorldPack or runtime event creates data records; the scheduler only sees
    these records and never needs a theme-specific location or occupation name.
    """

    def __init__(self):
        self.tasks: dict[str, Task] = {}

    def clear(self) -> None:
        self.tasks.clear()

    def upsert(self, task: Task) -> Task:
        existing = self.tasks.get(task.id)
        if existing is None:
            self.tasks[task.id] = task
            return task
        # Static task metadata may be refreshed from scenario data without
        # overwriting runtime progress or a terminal result.
        if existing.status in TERMINAL_TASK_STATES:
            return existing
        existing.title = task.title
        existing.location_id = task.location_id
        existing.interaction_type = task.interaction_type
        existing.duration_minutes = task.duration_minutes
        existing.earliest_at = task.earliest_at
        existing.deadline_at = task.deadline_at
        existing.priority = task.priority
        existing.required_capabilities = list(task.required_capabilities)
        existing.payload = dict(task.payload)
        return existing

    def project_schedule(self, agent, day: int, now: int) -> None:
        """Turn declared schedule slots into ordinary tasks for the given day."""
        day_start = (day - 1) * 1440
        for index, item in enumerate(agent.schedule):
            start_at = day_start + item.start_minutes
            next_item = agent.schedule[index + 1] if index + 1 < len(agent.schedule) else None
            deadline_at = day_start + next_item.start_minutes if next_item else day_start + 24 * 60
            task = Task(
                id=f"schedule:{day}:{agent.id}:{item.hour:02d}{item.minute:02d}",
                title=item.activity or item.label,
                assignee_id=agent.id,
                source="schedule",
                location_id=item.location,
                interaction_type=item.interaction_type,
                duration_minutes=max(5, item.expected_duration),
                earliest_at=start_at,
                deadline_at=deadline_at,
                priority=round(0.35 + item.responsibility * 0.45 + (1.0 - item.flexibility) * 0.2, 3),
                payload={
                    "schedule_label": item.label,
                    "schedule_kind": item.kind,
                    "commitment_id": f"{day}:{item.hour:02d}{item.minute:02d}",
                    "responsibility": item.responsibility,
                    "flexibility": item.flexibility,
                },
                created_at=now,
            )
            self.upsert(task)
        self.expire_overdue(agent.id, now)

    def create_need_task(self, agent_id: str, title: str, now: int, location_id: str,
                         interaction_type: str, priority: float, payload: dict | None = None) -> Task:
        task_id = f"need:{agent_id}:{interaction_type}"
        existing = self.tasks.get(task_id)
        if existing and existing.status in TERMINAL_TASK_STATES:
            task_id = f"{task_id}:{now}"
        task = Task(
            id=task_id, title=title, assignee_id=agent_id, source="need",
            location_id=location_id, interaction_type=interaction_type,
            duration_minutes=15, earliest_at=now, priority=priority,
            payload=dict(payload or {}), created_at=now,
        )
        return self.upsert(task)

    def expire_overdue(self, assignee_id: str, now: int) -> None:
        for task in self.tasks.values():
            if (task.assignee_id == assignee_id and task.status in OPEN_TASK_STATES
                    and task.deadline_at is not None and now >= task.deadline_at):
                task.status = "expired"
                task.blocking_reason = "任务窗口已结束"

    def available_for(self, assignee_id: str, now: int) -> list[Task]:
        self.expire_overdue(assignee_id, now)
        return [
            task for task in self.tasks.values()
            if task.assignee_id == assignee_id
            and task.status in OPEN_TASK_STATES
            and task.earliest_at <= now
        ]

    def start(self, task_id: str, now: int) -> Task | None:
        task = self.tasks.get(task_id)
        if task and task.status in OPEN_TASK_STATES:
            task.status = "active"
            task.started_at = task.started_at if task.started_at is not None else now
        return task

    def complete(self, task_id: str, now: int) -> Task | None:
        task = self.tasks.get(task_id)
        if task and task.status not in TERMINAL_TASK_STATES:
            task.status = "completed"
            task.completed_at = now
            task.blocking_reason = ""
        return task

    def block(self, task_id: str, reason: str) -> Task | None:
        task = self.tasks.get(task_id)
        if task and task.status not in TERMINAL_TASK_STATES:
            task.status = "blocked"
            task.blocking_reason = reason[:240]
        return task

    def to_dict(self, limit: int = 120) -> list[dict]:
        ordered = sorted(
            self.tasks.values(),
            key=lambda task: (task.status in TERMINAL_TASK_STATES, task.deadline_at or 10**12, -task.priority, task.id),
        )
        return [task.to_dict() for task in ordered[:limit]]
