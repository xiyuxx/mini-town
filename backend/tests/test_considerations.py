from backend.town.considerations import collect_active_considerations
from backend.town.cognition import Intention, IntentionProposal, LifePlan, PlanStep
from backend.town.engine import SimulationEngine


def test_fixed_commitment_precedes_urgent_need_and_open_intention():
    engine = SimulationEngine()
    engine._init_agents()
    engine.day, engine.hour, engine.minute = 1, 6, 0
    agent = engine.agents[0]
    agent.state.needs["hunger"] = 95
    agent.state.needs["social"] = 10
    agent.mental_state.add_intention(Intention(
        description="去公园散步", source="test", created_at=0,
        target_location="park", interaction_type="wait",
    ))

    considerations = collect_active_considerations(agent, engine)

    assert considerations[0].kind == "commitment"
    assert considerations[0].hard is True
    assert any(item.kind == "urgent_need" and item.details["need"] == "hunger" for item in considerations)
    assert any(item.kind == "intention" and item.source_id for item in considerations)


def test_plan_step_is_exposed_as_a_consideration():
    engine = SimulationEngine()
    engine._init_agents()
    agent = engine.agents[0]
    step = PlanStep(
        description="去公园观察天气", action={"interaction_type": "inspect", "location": "park"},
    )
    agent.mental_state.life_plan = LifePlan(
        focus="完成观察", motive="测试", created_at=0, review_at=100, steps=[step],
    )

    considerations = collect_active_considerations(agent, engine)

    plan_item = next(item for item in considerations if item.kind == "plan_step")
    assert plan_item.source_id == step.id
    assert plan_item.target_location == "park"


def test_intention_proposal_has_no_location_and_normalizes_invalid_values():
    proposal = IntentionProposal.from_dict({
        "description": "寻找一个可以安静阅读的地方",
        "interaction_type": "invented_action",
        "target": "mei",
        "goal_ids": ["goal_1", "goal_2"],
        "urgency": "not-a-number",
    })

    assert proposal.purpose == "寻找一个可以安静阅读的地方"
    assert proposal.interaction_type == "wait"
    assert proposal.target_agent_id == "mei"
    assert proposal.supports_goal_ids == ["goal_1", "goal_2"]
    assert proposal.urgency == 0.0
    assert "location" not in proposal.to_dict()
