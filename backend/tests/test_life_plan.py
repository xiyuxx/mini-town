"""Life plan steps: what counts as done, and what counts as a failure."""

import asyncio

from backend.town.cognition import LifePlan, LifePlanExhausted, PlanStep, PlanStepBlocked
from backend.town.engine import SimulationEngine
from backend.town.world import LOCATION_MAP


def run(coro):
    return asyncio.run(coro)


async def make_engine(tmp_path):
    engine = SimulationEngine()
    engine.llm.fallback = True
    engine.embedding_provider.fallback = True
    db_path = str(tmp_path / "town.db")
    engine.memory.db_path = db_path
    engine.relationship_store.db_path = db_path
    engine.trace.db_path = db_path
    await engine.init()
    return engine


def plan_agent(engine, steps):
    agent = next(item for item in engine.agents if item.id == "wang")
    agent.state.current_location = "home_wang"
    agent.state.x, agent.state.y = LOCATION_MAP["home_wang"].center
    agent.state.status = "IDLE"
    agent.mental_state.life_plan = LifePlan(
        focus="测试计划", motive="", created_at=0, review_at=10 ** 9,
        steps=[
            PlanStep(id=f"s{index}", description=text, action=action)
            for index, (text, action) in enumerate(steps, start=1)
        ],
    )
    return agent, agent.mental_state.life_plan


MOVE_HOME = ("走回家", {"interaction_type": "move", "content": "走回家", "location": "home_wang"})
REST = ("在客厅歇一会儿", {"interaction_type": "rest", "content": "在客厅歇一会儿", "duration_minutes": 10})


def test_a_step_the_agent_already_stands_at_is_done(tmp_path):
    """Plans open with travel. Walking home while at home is finished, not broken."""
    async def scenario():
        engine = await make_engine(tmp_path)
        agent, plan = plan_agent(engine, [MOVE_HOME, REST])

        action = engine.planner.next_plan_action(agent, engine)

        assert action["plan_step_id"] == "s2"
        assert plan.steps[0].status == "skipped"
        assert plan.steps[0].completed_at is None
        assert plan.status == "active"

    run(scenario())


def test_a_plan_left_with_nothing_to_do_is_completed_not_blocked(tmp_path):
    """Callers block a plan whenever a step raises; a done plan must survive that."""
    async def scenario():
        engine = await make_engine(tmp_path)
        agent, plan = plan_agent(engine, [MOVE_HOME])

        try:
            engine.planner.next_plan_action(agent, engine)
            raise AssertionError("an exhausted plan must not hand out an action")
        except LifePlanExhausted:
            pass

        plan.block_current_step("", "life plan is complete")
        assert plan.status == "completed"

    run(scenario())


def test_a_look_that_repeats_a_fresh_look_is_skipped(tmp_path):
    """The engine records what an agent walks into, so a second look is done."""
    async def scenario():
        engine = await make_engine(tmp_path)
        agent, plan = plan_agent(engine, [
            ("看看四周", {"interaction_type": "inspect", "content": "看看四周",
                          "target_id": ""}),
            REST,
        ])
        engine.interactions.store_observation(
            agent, engine.interactions.observation_snapshot(agent, engine),
            engine.get_sim_timestamp(),
        )

        action = engine.planner.next_plan_action(agent, engine)

        assert action["plan_step_id"] == "s2"
        assert plan.steps[0].status == "skipped"

    run(scenario())


def test_a_real_journey_is_not_skipped(tmp_path):
    """The skip is for steps already satisfied — not for steps still to come."""
    async def scenario():
        engine = await make_engine(tmp_path)
        agent, plan = plan_agent(engine, [
            ("去咖啡馆", {"interaction_type": "move", "content": "去咖啡馆", "location": "cafe"}),
            REST,
        ])

        action = engine.planner.next_plan_action(agent, engine)

        assert action["plan_step_id"] == "s1"
        assert plan.steps[0].status == "pending"

    run(scenario())


def test_an_impossible_step_is_reported_as_a_plan_problem(tmp_path):
    """The engine tells a stale plan apart from a broken model by exception type."""
    async def scenario():
        engine = await make_engine(tmp_path)
        agent, _ = plan_agent(engine, [
            ("去不存在的地方", {"interaction_type": "move", "content": "去不存在的地方",
                                "location": "nowhere_at_all"}),
        ])

        try:
            engine.planner.next_plan_action(agent, engine)
            raise AssertionError("an unreachable step must not be accepted")
        except PlanStepBlocked:
            pass

    run(scenario())
