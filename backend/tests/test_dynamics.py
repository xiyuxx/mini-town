"""The town's rhythms live in one file, at rates someone can reason about."""

import random

from backend.town.dynamics import TICKS_PER_SIM_DAY, TownDynamics, dynamics, per_sim_day
from backend.town.town_agent import InfrastructureState, WeatherEngine
from backend.town.world_pack import DEFAULT_WORLD


def test_the_rates_are_what_the_town_should_feel():
    """Per-tick probabilities are unreadable; the day rate is the contract."""
    weather = per_sim_day(dynamics.weather_change_probability)
    failures = per_sim_day(dynamics.infrastructure_failure_probability)

    assert 1 <= weather <= 6, f"weather should change a few times a day, not {weather:.1f}"
    assert 0.2 <= failures <= 0.7, f"a failure should be rare, not {failures:.2f} a day"


def test_a_power_cut_every_couple_of_days_not_every_day():
    """The rate this used to run at is what made failures feel constant."""
    old_rate = per_sim_day(0.005)

    assert old_rate > 1.0, "the old value was about 1.4 failures a day"
    assert per_sim_day(dynamics.infrastructure_failure_probability) < old_rate / 2


def test_the_outage_duration_comes_from_the_tuning_file():
    """It used to come from the world pack, which split the knob in two."""
    assert InfrastructureState(DEFAULT_WORLD).duration_range == dynamics.infrastructure_failure_ticks
    assert "failure_duration_ticks" not in DEFAULT_WORLD.infrastructure


def test_weather_changes_only_at_the_declared_rate(monkeypatch):
    """The declared rate is the gate: a roll above it must change nothing."""
    engine = WeatherEngine(DEFAULT_WORLD)
    engine.states = {"晴": {"transitions": {"雨": 1}}, "雨": {"transitions": {"晴": 1}}}
    engine._condition = "晴"
    monkeypatch.setattr(random, "random", lambda: 0.5)

    monkeypatch.setattr(dynamics, "weather_change_probability", 0.4)
    engine.tick("summer")
    assert engine.condition == "晴"

    monkeypatch.setattr(dynamics, "weather_change_probability", 0.6)
    engine.tick("summer")
    assert engine.condition == "雨"


def test_an_environment_override_replaces_the_default(monkeypatch):
    monkeypatch.setenv("WEATHER_CHANGE_PROBABILITY", "0.42")
    monkeypatch.setenv("INFRASTRUCTURE_FAILURE_TICKS_MIN", "3")
    monkeypatch.setenv("INFRASTRUCTURE_FAILURE_TICKS_MAX", "9")
    monkeypatch.setenv("OBSERVATION_REFRESH_MINUTES", "5")

    tuned = TownDynamics.from_env()

    assert tuned.weather_change_probability == 0.42
    assert tuned.infrastructure_failure_ticks == (3, 9)
    assert tuned.observation_refresh_minutes == 5
    assert tuned.infrastructure_failure_probability == dynamics.infrastructure_failure_probability


def test_a_nonsense_override_does_not_take_the_town_down(monkeypatch):
    monkeypatch.setenv("WEATHER_CHANGE_PROBABILITY", "very often")
    monkeypatch.setenv("OBSERVATION_REFRESH_MINUTES", "")

    tuned = TownDynamics.from_env()

    assert tuned.weather_change_probability == TownDynamics().weather_change_probability
    assert tuned.observation_refresh_minutes == TownDynamics().observation_refresh_minutes
    assert TICKS_PER_SIM_DAY == 288
