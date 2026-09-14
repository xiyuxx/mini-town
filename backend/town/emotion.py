"""Deterministic, personality-sensitive emotion dynamics."""

from dataclasses import dataclass


@dataclass
class EmotionProfile:
    baseline_valence: float = 0.0
    baseline_arousal: float = 0.18
    social_sensitivity: float = 0.5
    responsibility_sensitivity: float = 0.5
    weather_sensitivity: float = 0.35
    novelty_sensitivity: float = 0.4


def profile_from_data(raw: dict | None) -> EmotionProfile:
    raw = raw or {}
    def value(name, default):
        try:
            return max(0.0, min(1.0, float(raw.get(name, default))))
        except (TypeError, ValueError):
            return default
    return EmotionProfile(
        baseline_valence=max(-1.0, min(1.0, float(raw.get("baseline_valence", 0.0)))),
        baseline_arousal=value("baseline_arousal", 0.18),
        social_sensitivity=value("social_sensitivity", 0.5),
        responsibility_sensitivity=value("responsibility_sensitivity", 0.5),
        weather_sensitivity=value("weather_sensitivity", 0.35),
        novelty_sensitivity=value("novelty_sensitivity", 0.4),
    )


def apply_event(agent, event) -> str:
    """Update lasting emotion and return a memory label only when salient."""
    profile = getattr(agent, "emotion_profile", EmotionProfile())
    social = min(1.0, event.social_relevance + (0.55 if len(event.actors) > 1 else 0.0))
    pressure = 1.0 if event.type in {"action_blocked", "schedule_deviation"} else 0.0
    weather = min(1.0, event.sensory_salience)
    scale = 0.45 + social * profile.social_sensitivity * 0.35 + pressure * profile.responsibility_sensitivity * 0.35 + weather * profile.weather_sensitivity * 0.2
    delta_valence = event.emotional_valence * scale
    delta_arousal = max(event.emotional_intensity, abs(delta_valence)) * scale
    state = agent.mental_state.emotion
    state.valence = max(-1.0, min(1.0, state.valence * 0.82 + delta_valence))
    state.arousal = max(0.0, min(1.0, state.arousal * 0.8 + delta_arousal))
    if state.valence >= 0.22:
        state.label = "愉快" if state.arousal < 0.45 else "兴奋"
    elif state.valence <= -0.22:
        state.label = "烦躁" if state.arousal >= 0.4 else "低落"
    elif state.arousal >= 0.42:
        state.label = "惦记"
    else:
        state.label = "平静"
    agent.state.mood = state.label
    return "" if abs(state.valence) < 0.16 and state.arousal < 0.3 else state.label


def decay(agent) -> None:
    profile = getattr(agent, "emotion_profile", EmotionProfile())
    state = agent.mental_state.emotion
    state.valence += (profile.baseline_valence - state.valence) * 0.12
    state.arousal += (profile.baseline_arousal - state.arousal) * 0.16
    if abs(state.valence) < 0.16 and state.arousal < 0.3:
        state.label = "平静"
    agent.state.mood = state.label
