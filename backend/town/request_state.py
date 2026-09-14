"""Request metadata and stale-result checks for deferred LLM work."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RequestState:
    agent_id: str
    task_type: str
    generation: int
    sim_timestamp: int
    state_version: int
    day: int
    status: str
    location: str
    commitment_id: str = ""
    plan_id: str = ""
    plan_step_id: str = ""
    intention_id: str = ""


@dataclass(frozen=True)
class DialogueRequestState:
    dialogue_id: str
    location_id: str
    participant_ids: tuple[str, ...]
    generation: int
    sim_timestamp: int
    state_version: int


def capture_request_state(engine, agent, task_type: str) -> RequestState:
    commitment = agent.get_current_commitment(engine.day, engine.hour, engine.minute) or {}
    plan = agent.mental_state.life_plan
    step = plan.current_step if plan and plan.status == "active" else None
    intention = agent.mental_state.relevant_intention({
        "interaction_type": step.action.get("interaction_type", "") if step else "",
        "location": step.action.get("location", "") if step else "",
    }) if step else None
    return RequestState(
        agent_id=agent.id,
        task_type=task_type,
        generation=engine._task_generation,
        sim_timestamp=engine.get_sim_timestamp(),
        state_version=engine.state_version,
        day=engine.day,
        status=agent.state.status,
        location=agent.state.current_location,
        commitment_id=str(commitment.get("id", "")),
        plan_id=plan.id if step else "",
        plan_step_id=step.id if step else "",
        intention_id=intention.id if intention else "",
    )


def capture_dialogue_request(engine, session) -> DialogueRequestState:
    return DialogueRequestState(
        dialogue_id=session.id,
        location_id=session.location_id,
        participant_ids=tuple(sorted(item.id for item in session.participants)),
        generation=engine._task_generation,
        sim_timestamp=engine.get_sim_timestamp(),
        state_version=engine.state_version,
    )


def dialogue_stale_reason(engine, request: DialogueRequestState) -> str:
    if request.generation != engine._task_generation:
        return "simulation generation changed"
    session = engine.dialogue._active_dialogues.get(request.location_id)
    if not session or session.id != request.dialogue_id:
        return "dialogue session changed"
    participant_ids = tuple(sorted(item.id for item in session.participants))
    if participant_ids != request.participant_ids:
        return "dialogue participants changed"
    if any(item.state.current_location != request.location_id for item in session.participants):
        return "dialogue participant location changed"
    return ""


def stale_reason(engine, agent, request: RequestState) -> str:
    """Return a reason when a deferred result can no longer be committed."""
    if request.generation != engine._task_generation:
        return "simulation generation changed"
    if engine.day != request.day:
        return "simulation day changed"
    if request.task_type not in {"daily_plan", "reflection"} and agent.state.status != "IDLE":
        return f"agent is no longer idle ({agent.state.status})"
    if agent.state.current_location != request.location:
        return "agent location changed"
    if request.commitment_id:
        current = agent.get_current_commitment(engine.day, engine.hour, engine.minute) or {}
        if current.get("id") != request.commitment_id:
            return "current commitment changed"
    if request.plan_id:
        plan = agent.mental_state.life_plan
        step = plan.current_step if plan and plan.status == "active" else None
        if not plan or plan.id != request.plan_id or not step or step.id != request.plan_step_id:
            return "current plan step changed"
    if request.intention_id:
        intention = next(
            (item for item in agent.mental_state.intentions if item.id == request.intention_id), None,
        )
        if not intention or intention.status not in {"open", "executing", "suspended", "blocked"}:
            return "intention is no longer active"
    return ""
