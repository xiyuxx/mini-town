"""Structured simulation time helpers."""

from dataclasses import dataclass
import re

_SIM_TIME_RE = re.compile(r"^第(?P<day>\d+)天\s+(?P<hour>\d{1,2}):(?P<minute>\d{2})$")


@dataclass(frozen=True)
class SimTime:
    day: int
    hour: int
    minute: int

    @property
    def timestamp(self) -> int:
        return (self.day - 1) * 1440 + self.hour * 60 + self.minute

    def __str__(self) -> str:
        return f"第{self.day}天 {self.hour:02d}:{self.minute:02d}"


def parse_sim_time(value: str | None) -> SimTime | None:
    if not value:
        return None
    match = _SIM_TIME_RE.match(value.strip())
    if not match:
        return None
    return SimTime(
        day=int(match.group("day")),
        hour=int(match.group("hour")),
        minute=int(match.group("minute")),
    )


def sim_timestamp(value: str | None) -> int | None:
    parsed = parse_sim_time(value)
    return parsed.timestamp if parsed else None
