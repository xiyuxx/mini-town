"""Structured schedule compilation without natural-language inference."""

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class TimeWindow:
    start_minutes: int
    end_minutes: int

    def __post_init__(self):
        if not 0 <= self.start_minutes < 24 * 60:
            raise ValueError("time window start must be within a day")
        if not 0 < self.end_minutes <= 24 * 60:
            raise ValueError("time window end must be within a day")
        if self.end_minutes <= self.start_minutes:
            raise ValueError("time window must end after it starts")

    def contains(self, minute: int) -> bool:
        return self.start_minutes <= minute < self.end_minutes


@dataclass(frozen=True)
class RoutineBlock:
    id: str
    window: TimeWindow
    intent: str
    candidate_locations: tuple[str, ...] = ()
    candidate_activities: tuple[str, ...] = ()
    priority: float = 0.5
    flexibility: float = 0.7

    def contains(self, hour: int, minute: int) -> bool:
        return self.window.contains(hour * 60 + minute)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "window": {"start": self.window.start_minutes, "end": self.window.end_minutes},
            "intent": self.intent,
            "candidateLocations": list(self.candidate_locations),
            "candidateActivities": list(self.candidate_activities),
            "priority": self.priority,
            "flexibility": self.flexibility,
        }


@dataclass(frozen=True)
class ScheduleItem:
    hour: int
    minute: int
    label: str
    location: str
    kind: str
    activity: str
    expected_duration: int
    flexibility: float
    responsibility: float
    affected_people: tuple[str, ...]
    consequence_of_delay: str
    interaction_type: str = "work_on_goal"
    effects: tuple[dict, ...] = ()
    is_flexible_slot: bool = False

    @property
    def start_minutes(self) -> int:
        return self.hour * 60 + self.minute


def _parse_clock(value: Any) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", text)
    if not match:
        raise ValueError(f"invalid clock value: {value}")
    hour, minute = int(match.group(1)), int(match.group(2))
    if not 0 <= hour < 24 or not 0 <= minute < 60:
        raise ValueError(f"invalid clock value: {value}")
    return hour * 60 + minute


def _parse_window(raw: Any) -> TimeWindow:
    if isinstance(raw, dict):
        start, end = raw.get("start"), raw.get("end")
    elif isinstance(raw, (list, tuple)) and len(raw) == 2:
        start, end = raw
    else:
        raise ValueError("routine block window must contain start and end")
    return TimeWindow(_parse_clock(start), _parse_clock(end))


def _clamp(value: Any, low: float, high: float, default: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default


def compile_routine_blocks(raw_blocks: list[dict] | None) -> list[RoutineBlock]:
    """Compile flexible time blocks without turning them into hard tasks."""
    blocks: list[RoutineBlock] = []
    for index, raw in enumerate(raw_blocks or []):
        if not isinstance(raw, dict):
            raise ValueError("routine block must be an object")
        intent = str(raw.get("intent", "")).strip()
        if not intent:
            raise ValueError("routine block intent cannot be empty")
        blocks.append(RoutineBlock(
            id=str(raw.get("id", f"routine_{index + 1}")),
            window=_parse_window(raw.get("window")),
            intent=intent,
            candidate_locations=tuple(str(item) for item in raw.get("candidate_locations", raw.get("candidateLocations", []))),
            candidate_activities=tuple(str(item) for item in raw.get("candidate_activities", raw.get("candidateActivities", []))),
            priority=_clamp(raw.get("priority"), 0.0, 1.0, 0.5),
            flexibility=_clamp(raw.get("flexibility"), 0.0, 1.0, 0.7),
        ))
    ids = [block.id for block in blocks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate routine block id")
    return sorted(blocks, key=lambda block: block.window.start_minutes)


def compile_schedule(raw_schedule: list[tuple | dict]) -> list[ScheduleItem]:
    """Compile explicit schedule data.

    Tuple entries remain supported for the existing town, but receive neutral
    defaults. Domain semantics must be declared in dict entries instead of
    being inferred from words in the display label.
    """
    items: list[ScheduleItem] = []
    for raw in raw_schedule:
        if isinstance(raw, dict):
            hour = int(raw.get("hour", 0))
            minute = int(raw.get("minute", 0))
            label = str(raw.get("label", raw.get("activity", "安排"))).strip()
            location = str(raw.get("location", ""))
            kind = str(raw.get("kind", "activity"))
            activity = str(raw.get("activity", label)).strip()
            duration = max(0, int(raw.get("expected_duration", raw.get("duration", 30))))
            flexibility = _clamp(raw.get("flexibility"), 0.0, 1.0, 0.6)
            responsibility = _clamp(raw.get("responsibility"), 0.0, 1.0, 0.4)
            affected = tuple(str(item) for item in raw.get("affected_people", []))
            consequence = str(raw.get("consequence_of_delay", "延后会占用后续时间"))
            interaction_type = str(raw.get("interaction_type", "work_on_goal"))
            effects = tuple(dict(item) for item in raw.get("effects", []) if isinstance(item, dict))
            is_flexible_slot = bool(raw.get("is_flexible_slot", raw.get("isFlexibleSlot", False)))
        else:
            hour, minute, label, location = raw
            label = str(label).strip()
            location = str(location)
            kind = "activity"
            activity = label
            duration = 30
            flexibility = 0.6
            responsibility = 0.4
            affected = ()
            consequence = "延后会占用后续时间；具体影响需结合当时事实判断"
            # Legacy tuple schedules only describe a visible calendar slot.
            # Keep them as ordinary activities; domain actions such as eating
            # must be declared explicitly in the world pack.
            interaction_type = "wait"
            effects = ()
            is_flexible_slot = True
        items.append(ScheduleItem(
            hour=int(hour), minute=int(minute), label=label, location=location,
            kind=kind, activity=activity, expected_duration=duration,
            flexibility=flexibility, responsibility=responsibility,
            affected_people=affected, consequence_of_delay=consequence,
            interaction_type=interaction_type, effects=effects,
            is_flexible_slot=is_flexible_slot,
        ))
    return sorted(items, key=lambda item: item.start_minutes)
