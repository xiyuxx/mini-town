"""Coarse-grained daily plans that sit above executable tasks."""

from dataclasses import dataclass, field
from typing import Any

from .schedule import TimeWindow


BLOCK_STATES = {"planned", "active", "completed", "skipped"}


@dataclass
class DailyPlanBlock:
    id: str
    window: TimeWindow
    intent: str
    candidates: list[dict] = field(default_factory=list)
    priority: float = 0.5
    flexibility: float = 0.7
    status: str = "planned"

    def contains(self, minute: int) -> bool:
        return self.window.contains(minute)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "window": {
                "start": self.window.start_minutes,
                "end": self.window.end_minutes,
            },
            "intent": self.intent,
            "candidates": list(self.candidates),
            "priority": self.priority,
            "flexibility": self.flexibility,
            "status": self.status,
        }


@dataclass
class DailyPlan:
    agent_id: str
    day: int
    focus: str
    blocks: list[DailyPlanBlock]
    created_at: int
    review_at: int
    replan_reason: str = ""

    def current_block(self, minute: int) -> DailyPlanBlock | None:
        for block in self.blocks:
            if block.status in {"completed", "skipped"}:
                continue
            if block.contains(minute):
                block.status = "active"
                return block
        return None

    def mark_completed(self, block_id: str) -> bool:
        for block in self.blocks:
            if block.id == block_id and block.status in {"planned", "active"}:
                block.status = "completed"
                return True
        return False

    def mark_skipped(self, block_id: str) -> bool:
        for block in self.blocks:
            if block.id == block_id and block.status in {"planned", "active"}:
                block.status = "skipped"
                return True
        return False

    def to_dict(self) -> dict:
        return {
            "agentId": self.agent_id,
            "day": self.day,
            "focus": self.focus,
            "blocks": [block.to_dict() for block in self.blocks],
            "createdAt": self.created_at,
            "reviewAt": self.review_at,
            "replanReason": self.replan_reason,
        }


def _number(value: Any, low: float, high: float, default: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default


def parse_daily_plan(raw: dict, agent_id: str, day: int, now: int) -> DailyPlan:
    if not isinstance(raw, dict):
        raise ValueError("daily plan must be an object")
    blocks: list[DailyPlanBlock] = []
    for index, item in enumerate(raw.get("blocks", [])):
        if not isinstance(item, dict):
            continue
        intent = str(item.get("intent", "")).strip()
        window = item.get("window")
        if not intent or not isinstance(window, (list, tuple)) or len(window) != 2:
            continue
        try:
            start, end = int(window[0]), int(window[1])
            parsed_window = TimeWindow(start, end)
        except (TypeError, ValueError):
            continue
        status = str(item.get("status", "planned"))
        if status not in BLOCK_STATES:
            status = "planned"
        candidates = [candidate for candidate in item.get("candidates", []) if isinstance(candidate, dict)]
        blocks.append(DailyPlanBlock(
            id=str(item.get("id", f"block_{index + 1}")),
            window=parsed_window,
            intent=intent[:180],
            candidates=candidates[:8],
            priority=_number(item.get("priority"), 0.0, 1.0, 0.5),
            flexibility=_number(item.get("flexibility"), 0.0, 1.0, 0.7),
            status=status,
        ))
    if not blocks:
        raise ValueError("daily plan contains no valid blocks")
    blocks.sort(key=lambda block: block.window.start_minutes)
    for previous, current in zip(blocks, blocks[1:]):
        if current.window.start_minutes < previous.window.end_minutes:
            raise ValueError("daily plan blocks overlap")
    ids = [block.id for block in blocks]
    if len(ids) != len(set(ids)):
        raise ValueError("daily plan contains duplicate block ids")
    return DailyPlan(
        agent_id=agent_id,
        day=day,
        focus=str(raw.get("focus", blocks[0].intent))[:180],
        blocks=blocks,
        created_at=now,
        review_at=now + max(15, min(720, int(raw.get("review_after_minutes", 180)))),
        replan_reason=str(raw.get("replan_reason", ""))[:240],
    )
