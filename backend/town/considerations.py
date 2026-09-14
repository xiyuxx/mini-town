"""Code-owned considerations that shape the next Agent decision."""

from .cognition import ActiveConsideration


def _social_target(agent, engine):
    """Who this agent would most want to see right now.

    The strongest tie wins; someone with no ties at all is pointed at the
    nearest other agent, so the drive to socialise has a direction even before
    any relationship exists.
    """
    others = [other for other in engine.agents if other.id != agent.id]
    if not others:
        return None
    ranked = []
    for other in others:
        tie = engine.relationship_store.rapport(agent.id, other.id)
        ranked.append((tie.familiarity / 20 + max(0.0, tie.affinity) / 10, other))
    best_score, best = max(ranked, key=lambda item: item[0])
    if best_score > 0:
        return best
    return min(
        others,
        key=lambda other: abs(other.state.x - agent.state.x) + abs(other.state.y - agent.state.y),
    )


def collect_active_considerations(agent, engine) -> list[ActiveConsideration]:
    """Collect current motives and constraints without asking an LLM."""
    now = engine.get_sim_timestamp()
    considerations: list[ActiveConsideration] = []

    commitment = agent.get_current_commitment(engine.day, engine.hour, engine.minute)
    if commitment and commitment.get("status") not in {"completed", "skipped"}:
        deadline = commitment.get("deadline_minutes")
        current_minutes = engine.hour * 60 + engine.minute
        urgency = 1.0 if deadline is not None and current_minutes >= deadline else 0.85
        considerations.append(ActiveConsideration(
            kind="commitment",
            source_id=str(commitment.get("id", "")),
            description=str(commitment.get("label", "当前日程")),
            priority=0.98,
            urgency=urgency,
            hard=True,
            target_location=str(commitment.get("location", "")),
            affected_agents=list(commitment.get("affected_people", [])),
            details={
                "status": commitment.get("status", "pending"),
                "activity": commitment.get("activity", ""),
                "kind": commitment.get("kind", "activity"),
                "deadline_minutes": deadline,
            },
        ))

    needs = agent.state.needs
    hunger_urgency = max(0.0, min(1.0, (needs.get("hunger", 0) - 55) / 45))
    if hunger_urgency > 0:
        considerations.append(ActiveConsideration(
            kind="urgent_need", source_id=f"need:{agent.id}:hunger",
            description="需要补充食物", priority=0.94, urgency=hunger_urgency,
            details={"need": "hunger", "value": needs.get("hunger", 0)},
        ))

    energy_urgency = max(0.0, min(1.0, (25 - needs.get("energy", 0)) / 25))
    if energy_urgency > 0:
        considerations.append(ActiveConsideration(
            kind="urgent_need", source_id=f"need:{agent.id}:energy",
            description="需要恢复精力", priority=0.94, urgency=energy_urgency,
            details={"need": "energy", "value": needs.get("energy", 0)},
        ))

    social_urgency = agent.social_urgency()
    if social_urgency > 0:
        confidant = _social_target(agent, engine)
        description = "需要与人交流或接触"
        details = {"need": "social", "value": needs.get("social", 0)}
        if confidant is not None:
            tie = engine.relationship_store.rapport(agent.id, confidant.id)
            description = f"需要与人交流，想去找{confidant.name}"
            details.update(
                target_agent_id=confidant.id,
                target_location=confidant.state.current_location,
                familiarity=round(tie.familiarity, 1),
            )
        considerations.append(ActiveConsideration(
            kind="urgent_need", source_id=f"need:{agent.id}:social",
            description=description, priority=0.7, urgency=social_urgency,
            details=details,
        ))

    plan = agent.mental_state.life_plan
    step = plan.current_step if plan and plan.status == "active" else None
    if step:
        considerations.append(ActiveConsideration(
            kind="plan_step", source_id=step.id, description=step.description,
            priority=0.82, urgency=0.55, target_location=str(step.action.get("location", "")),
            details={"plan_id": plan.id, "status": step.status},
        ))

    if plan and plan.status == "blocked":
        considerations.append(ActiveConsideration(
            kind="blocking_fact", source_id=plan.id,
            description=plan.replan_reason or "当前生活计划被阻塞",
            priority=0.91, urgency=0.8, hard=True,
            details={"plan_id": plan.id},
        ))

    for goal in agent.mental_state.goals:
        if goal.status == "blocked":
            considerations.append(ActiveConsideration(
                kind="blocking_fact", source_id=goal.id,
                description=goal.description, priority=0.9, urgency=0.7,
                hard=True, details={"goal_id": goal.id},
            ))

    for task in engine.tasks.tasks.values():
        if (
            task.assignee_id == agent.id
            and task.status in {"planned", "active"}
            and task.earliest_at <= now
        ):
            considerations.append(ActiveConsideration(
                kind="task", source_id=task.id, description=task.title,
                priority=task.priority, urgency=0.65 if task.status == "active" else 0.35,
                target_location=task.location_id,
                details={"status": task.status, "interaction_type": task.interaction_type},
            ))

    for intention in agent.mental_state.open_intentions(now):
        urgency = 0.35
        if intention.expected_until is not None:
            remaining = intention.expected_until - now
            urgency = max(0.0, min(1.0, 1.0 - remaining / 180))
        considerations.append(ActiveConsideration(
            kind="intention", source_id=intention.id, description=intention.description,
            priority=0.72, urgency=urgency,
            target_location=intention.target_location,
            details={
                "status": intention.status,
                "interaction_type": intention.interaction_type,
                "goal_id": intention.goal_id,
                "confidence": intention.confidence,
            },
        ))

    block = next(iter(agent.get_routine_blocks_for_time(engine.hour, engine.minute)), None)
    if block:
        considerations.append(ActiveConsideration(
            kind="routine", source_id=block.id, description=block.intent,
            priority=block.priority, urgency=0.2,
            target_location=block.candidate_locations[0] if block.candidate_locations else "",
            details={"flexibility": block.flexibility},
        ))

    considerations.sort(
        key=lambda item: (item.hard, item.priority, item.urgency), reverse=True,
    )
    return considerations
