"""Town environment agent: weather, infrastructure, festivals.

The town agent reports the state of the world each tick.
It does NOT make decisions for people.
"""

import random
from dataclasses import dataclass
from ..config import config
from .world import LOCATION_MAP
from .world_pack import DEFAULT_WORLD, WorldPack


def _weighted_choice(choices: dict[str, float]) -> str:
    """Pick a key from a weighted dict."""
    total = sum(choices.values())
    r = random.random() * total
    cumulative = 0.0
    for key, weight in choices.items():
        cumulative += weight
        if r <= cumulative:
            return key
    return next(iter(choices))


@dataclass
class TownState:
    weather: dict         # {condition, temperature, wind, description}
    infrastructure: dict  # {power, water, road_main}
    events: list[dict]    # [{type, location, description}]
    season: str           # spring, summer, autumn, winter
    festivals: list[str]  # active festival names this tick


class WeatherEngine:
    """Markov weather simulation driven by declarative world data."""

    def __init__(self, world_pack: WorldPack):
        self.weather = world_pack.weather
        self.states = dict(self.weather.get("states", {}))
        self._condition = str(self.weather.get("initial", next(iter(self.states), "default")))

    def tick(self, season: str = "summer") -> dict:
        if random.random() < config.WEATHER_CHANGE_PROBABILITY:
            transitions = dict(self.states.get(self._condition, {}).get("transitions", {}))
            if transitions:
                self._condition = _weighted_choice(transitions)
        state = dict(self.states.get(self._condition, {}))
        modifier = state.get("temperature_modifier", [-2, 3])
        base_temperatures = self.weather.get("season_base_temperature", {})
        base_temp = float(base_temperatures.get(season, 20))
        temperature = round(base_temp + random.uniform(-5, 5) + random.uniform(*modifier), 1)
        winds = [str(item) for item in state.get("winds", [""])]
        return {
            "condition": self._condition,
            "temperature": temperature,
            "wind": random.choice(winds) if winds else "",
            "description": str(state.get("description", self._condition)),
            "travel_cost": float(state.get("travel_cost", 0.0)),
            "sensory_salience": float(state.get("sensory_salience", 0.0)),
        }

    @property
    def condition(self) -> str:
        return self._condition


class InfrastructureState:
    """Tracks declarative infrastructure services and temporary failures."""

    def __init__(self, world_pack: WorldPack):
        raw = world_pack.infrastructure
        self.services = {str(key): dict(value) for key, value in raw.get("services", {}).items()}
        duration = raw.get("failure_duration_ticks", [1, 4])
        self.duration_range = (max(1, int(duration[0])), max(1, int(duration[1])))
        self.values = {key: str(spec.get("normal", "normal")) for key, spec in self.services.items()}
        self._pending: dict[str, int] = {}

    def tick(self) -> dict:
        for service in [key for key, remaining in self._pending.items() if remaining <= 0]:
            self.values[service] = str(self.services[service].get("normal", "normal"))
            del self._pending[service]
        for service in list(self._pending):
            self._pending[service] -= 1
        if self.services and random.random() < config.INFRASTRUCTURE_EVENT_PROBABILITY:
            available = [key for key in self.services if key not in self._pending]
            if available:
                service = random.choice(available)
                self._pending[service] = random.randint(*self.duration_range)
                self.values[service] = str(self.services[service].get("failed", "failed"))
        return self.to_dict()

    def to_dict(self) -> dict:
        return dict(self.values)


class TownAgent:

    def __init__(self, world_pack: WorldPack = DEFAULT_WORLD):
        self.world_pack = world_pack
        weather = world_pack.weather
        self.seasons = [str(item) for item in weather.get("seasons", ["default"])]
        self.season_length_days = max(1, int(weather.get("season_length_days", 30)))
        self.festivals = world_pack.festivals
        self.services_spec = dict(world_pack.infrastructure.get("services", {}))
        self.need_deltas = dict(world_pack.simulation.get("need_deltas_per_tick", {}))
        self.weather_engine = WeatherEngine(world_pack)
        self.infrastructure = InfrastructureState(world_pack)
        self._last_state: TownState | None = None
        self._last_condition = ""
        self._last_services: dict[str, str] = {}

    def estimate_activity_duration(self, agent, description: str,
                                   location_id: str,
                                   hour: int,
                                   minute: int,
                                   requested_minutes: int | None = None) -> int:
        """Estimate activity duration as a town-level world rule.

        The LLM may suggest a duration, but the town clamps and normalizes it
        so visible actions always consume at least one tick and stay plausible.
        """
        tick = config.TICK_INTERVAL_MINUTES
        base = requested_minutes if requested_minutes is not None else 15
        base = max(tick, min(120, int(base)))
        return ((base + tick - 1) // tick) * tick

    def advance_agent_needs(self, agent) -> None:
        """Advance deterministic needs once per tick.

        Needs are deliberately rule-owned so an LLM cannot make a character
        ignore hunger or exhaustion indefinitely.
        """
        needs = agent.state.needs
        needs["hunger"] = max(0, min(100, needs.get("hunger", 25) + float(self.need_deltas.get("hunger", 0))))
        needs["social"] = max(0, min(100, needs.get("social", 55) + float(self.need_deltas.get("social", 0))))
        if agent.state.status != "ACTING":
            needs["energy"] = max(0, min(100, needs.get("energy", 75) + float(self.need_deltas.get("energy_when_not_acting", 0))))

    def apply_activity_outcome(self, agent, description: str) -> dict[str, float]:
        """Compatibility shim; structured InteractionEngine owns effects."""
        return agent.state.needs.copy()

    def choose_rule_based_action(self, agent, day: int, hour: int, minute: int) -> dict | None:
        """Provide deterministic offline fallback without choosing for a live LLM."""
        commitment = agent.get_current_commitment(day, hour, minute)
        # In normal mode schedule items are motives with flexibility and
        # responsibility metadata. They remain in the LLM context instead of
        # bypassing deliberation as mandatory actions.
        if (agent.llm.fallback and commitment
                and commitment["status"] not in ("active", "completed", "skipped")):
            kind = commitment["kind"]
            at_destination = agent.state.current_location == commitment["location"]

            if kind == "travel":
                if at_destination:
                    agent.complete_travel_commitment(commitment["id"])
                    return None
                return {
                    "action": "move",
                    "location": commitment["location"],
                    "content": commitment["label"],
                    "commitment": commitment,
                    "source": "commitment",
                }

            if not at_destination:
                return {
                    "action": "move",
                    "location": commitment["location"],
                    "content": commitment["label"],
                    "commitment": commitment,
                    "source": "commitment",
                }

            activity = (commitment.get("activity") or commitment.get("label") or "").strip()
            if not activity:
                agent.complete_travel_commitment(commitment["id"])
                return None
            return {
                "action": "act",
                "content": activity,
                "location": commitment["location"],
                "duration_minutes": commitment["expected_duration"],
                "commitment": commitment,
                "source": "commitment",
            }

        # Needs are world facts, not predetermined actions. In normal mode the
        # LLM receives them with weather and place context, then chooses how to
        # satisfy them. Fallback mode stays deterministic for offline runs.
        if not agent.llm.fallback:
            return None
        needs = agent.state.needs
        if needs.get("hunger", 0) >= 80:
            return {
                "action": "act", "interaction_type": "consume",
                "content": "使用当前位置可用的食物资源补充体力",
                "location": agent.state.current_location,
                "source": "need:hunger:fallback",
            }
        if needs.get("energy", 100) <= 20:
            return {
                "action": "act", "interaction_type": "rest",
                "content": "短暂休息恢复精力", "quality": 0.5,
                "duration_minutes": 30,
                "location": agent.state.current_location,
                "source": "need:energy:fallback",
            }
        return None

    def tick(self, sim_hour: int, day: int = 1, agent_locations: dict[str, str] | None = None) -> TownState:
        """Advance one simulation tick and return the new TownState.

        Args:
            sim_hour: current simulation hour (0-23)
            day: current simulation day (1-indexed)
            agent_locations: dict of agent_id -> location_id
        """
        if agent_locations is None:
            agent_locations = {}

        season_idx = ((day - 1) // self.season_length_days) % len(self.seasons)
        season = self.seasons[season_idx]

        weather = self.weather_engine.tick(season)
        infra = self.infrastructure.tick()
        events = self._weather_events(weather)
        events.extend(self._infrastructure_events(infra))

        # Festivals
        festivals = self._check_festivals(day, sim_hour)
        for festival in festivals:
            events.append({
                "type": "festival",
                "location": str(festival.get("location", "")),
                "description": str(festival.get("description", festival.get("name", ""))),
            })


        self._last_state = TownState(
            weather=weather,
            infrastructure=infra,
            events=events,
            season=season,
            festivals=[str(item.get("name", item.get("id", ""))) for item in festivals],
        )
        return self._last_state

    def _weather_events(self, weather: dict) -> list[dict]:
        """Report a change in the sky.

        Weather is the one state that genuinely comes from outside the town, so
        it is announced rather than discovered: rain needs people to notice it
        even when they were not looking out of the window.
        """
        condition = str(weather.get("condition", ""))
        previous = self._last_condition
        self._last_condition = condition
        if not previous or previous == condition:
            return []
        return [{
            "type": "weather_change",
            "location": "",
            "description": f"天气变成{condition}——{weather.get('description', '')}",
        }]

    def _infrastructure_events(self, values: dict) -> list[dict]:
        """Report services that broke or came back, using the pack's own labels."""
        events = []
        for name, status in values.items():
            previous = self._last_services.get(name, status)
            if previous == status:
                continue
            spec = self.services_spec.get(name, {})
            label = str(spec.get("labels", {}).get(str(status), status))
            events.append({
                "type": "infrastructure",
                "service": name,
                "status": str(status),
                "location": name if name in LOCATION_MAP else "",
                "description": label,
            })
        self._last_services = dict(values)
        return events

    def _check_festivals(self, day: int, sim_hour: int) -> list[dict]:
        active = []
        for festival in self.festivals:
            exact_day = festival.get("day")
            repeat = festival.get("repeat_every_days")
            if exact_day is not None and day == int(exact_day):
                active.append(festival)
            elif repeat is not None and day % max(1, int(repeat)) == 0:
                active.append(festival)
        return active

    def get_state(self) -> dict:
        if self._last_state is None:
            return {
                "weather": self.world_pack.initial_weather(),
                "infrastructure": self.infrastructure.to_dict(),
                "events": [],
                "season": self.seasons[0],
                "festivals": [],
            }
        return {
            "weather": self._last_state.weather,
            "infrastructure": self._last_state.infrastructure,
            "events": self._last_state.events,
            "season": self._last_state.season,
            "festivals": self._last_state.festivals,
        }
