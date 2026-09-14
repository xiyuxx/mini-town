from backend.town.dialogue import DialogueSession
from backend.town.engine import SimulationEngine
from backend.town.request_state import (
    capture_dialogue_request, capture_request_state, dialogue_stale_reason,
    stale_reason,
)


def test_deferred_request_becomes_stale_when_agent_location_changes():
    engine = SimulationEngine()
    engine._init_agents()
    agent = engine.agents[0]
    request = capture_request_state(engine, agent, "life_plan")

    agent.state.current_location = "in_transit"

    assert stale_reason(engine, agent, request) == "agent location changed"


def test_deferred_request_survives_time_advance_when_context_remains_valid():
    engine = SimulationEngine()
    engine._init_agents()
    agent = engine.agents[0]
    request = capture_request_state(engine, agent, "life_plan")
    engine._advance_time()

    assert stale_reason(engine, agent, request) == ""


def test_dialogue_request_becomes_stale_when_session_participant_moves():
    engine = SimulationEngine()
    engine._init_agents()
    first, second = engine.agents[:2]
    second.state.current_location = first.state.current_location
    session = DialogueSession(
        "dlg", [first, second], [first, second], first.state.current_location,
    )
    engine.dialogue._active_dialogues[session.location_id] = session
    request = capture_dialogue_request(engine, session)
    second.state.current_location = "in_transit"

    assert dialogue_stale_reason(engine, request) == "dialogue participant location changed"


def test_reset_generation_invalidates_old_request():
    engine = SimulationEngine()
    engine._init_agents()
    agent = engine.agents[0]
    request = capture_request_state(engine, agent, "life_plan")
    engine._cancel_deferred_tasks()

    assert stale_reason(engine, agent, request) == "simulation generation changed"
