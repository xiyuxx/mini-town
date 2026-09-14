"""How eventful the town is, in one place.

These are the knobs worth reaching for while watching the town run, so they live
together instead of being scattered between config.py and the world pack. Every
value can also be overridden from the environment for a single run.

The probabilities are per tick, and a simulated day is 288 of them, which is why
they look small: the numbers below are the rates the town should actually feel.
"""

from dataclasses import dataclass, replace
import os

from ..config import config

TICKS_PER_SIM_DAY = max(1, 1440 // max(1, config.TICK_INTERVAL_MINUTES))


def _float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class TownDynamics:
    # Weather changes about three times a simulated day. It used to be 0.08 per
    # tick, which is twenty-three changes a day — a town where the sky cannot
    # make up its mind, and every change is announced to everyone.
    weather_change_probability: float = 0.01

    # One failure every couple of days. It used to be 0.005 per tick, about 1.4
    # a day, and the comment beside it claimed 0.7% per day.
    infrastructure_failure_probability: float = 0.0015
    infrastructure_failure_ticks: tuple[int, int] = (6, 36)

    # How often an agent re-reads its surroundings while staying put. Kept under
    # the 30-minute window that ordering at a counter depends on.
    observation_refresh_minutes: int = 10

    @classmethod
    def from_env(cls) -> "TownDynamics":
        base = cls()
        ticks = (
            _int("INFRASTRUCTURE_FAILURE_TICKS_MIN", base.infrastructure_failure_ticks[0]),
            _int("INFRASTRUCTURE_FAILURE_TICKS_MAX", base.infrastructure_failure_ticks[1]),
        )
        return replace(
            base,
            weather_change_probability=_float(
                "WEATHER_CHANGE_PROBABILITY", base.weather_change_probability,
            ),
            infrastructure_failure_probability=_float(
                "INFRASTRUCTURE_FAILURE_PROBABILITY",
                base.infrastructure_failure_probability,
            ),
            infrastructure_failure_ticks=(
                max(1, ticks[0]), max(max(1, ticks[0]), ticks[1]),
            ),
            observation_refresh_minutes=max(
                1, _int("OBSERVATION_REFRESH_MINUTES", base.observation_refresh_minutes),
            ),
        )

def per_sim_day(probability: float) -> float:
    """What a per-tick chance means over a day, for reasoning about rates."""
    return probability * TICKS_PER_SIM_DAY


dynamics = TownDynamics.from_env()
