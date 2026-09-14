from types import SimpleNamespace

from backend.town.emotion import EmotionProfile, apply_event, decay
from backend.town.experience import ExperienceEvent


def agent(profile):
    state = SimpleNamespace(label="平静", valence=0.0, arousal=0.1)
    return SimpleNamespace(
        emotion_profile=profile,
        mental_state=SimpleNamespace(emotion=state),
        state=SimpleNamespace(mood="平静"),
    )


def test_neutral_event_does_not_create_memory_emotion_label():
    person = agent(EmotionProfile())
    label = apply_event(person, ExperienceEvent("arrival", "park", {}, emotional_intensity=0.05))
    assert label == ""
    assert person.state.mood == "平静"


def test_same_event_has_personality_sensitive_emotion():
    social = agent(EmotionProfile(social_sensitivity=1.0))
    reserved = agent(EmotionProfile(social_sensitivity=0.1))
    event = ExperienceEvent("dialogue", "cafe", {}, actors=["a", "b"], emotional_valence=0.5, emotional_intensity=0.5, social_relevance=0.8)
    apply_event(social, event)
    apply_event(reserved, event)
    assert social.mental_state.emotion.valence > reserved.mental_state.emotion.valence


def test_emotion_decays_toward_personal_baseline():
    person = agent(EmotionProfile(baseline_valence=-0.1, baseline_arousal=0.2))
    person.mental_state.emotion.valence = 0.8
    person.mental_state.emotion.arousal = 0.8
    decay(person)
    assert person.mental_state.emotion.valence < 0.8
    assert person.mental_state.emotion.arousal < 0.8
