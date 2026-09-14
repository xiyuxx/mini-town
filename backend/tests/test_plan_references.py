"""A plan may only cite ids that exist, and that includes places it has not reached."""

import asyncio

from backend.town.engine import SimulationEngine
from backend.town.world import LOCATION_MAP


def run(coro):
    return asyncio.run(coro)


async def make_engine(tmp_path):
    engine = SimulationEngine()
    engine.llm.fallback = False      # planning is stubbed below, never called out
    engine.embedding_provider.fallback = True
    db_path = str(tmp_path / "town.db")
    engine.memory.db_path = db_path
    engine.relationship_store.db_path = db_path
    engine.trace.db_path = db_path
    await engine.init()
    return engine


def stub_planner(engine, plan_payloads):
    """Answer the intention and plan calls with canned JSON, recording payloads."""
    calls: list[dict] = []
    answered = 0

    async def fake_json_call(system, payload, mode="chat", task="json_call"):
        nonlocal answered
        calls.append({"task": task, "payload": payload})
        if task == "intention_proposal":
            return {"purpose": "去咖啡馆坐坐", "interaction_type": "wait"}
        answer = plan_payloads[min(answered, len(plan_payloads) - 1)]
        answered += 1
        return answer

    engine.planner._json_call = fake_json_call
    return calls


def location_with_process(engine):
    """A real place that offers a real process, with the id to cite."""
    for location_id in LOCATION_MAP:
        processes = engine.resources.processes_at(location_id)
        if processes:
            return location_id, processes[0]["id"]
    raise AssertionError("the world has no usable process to cite")


def counter_item(engine, location):
    """A real item sold over a counter: one this agent has not looked at yet."""
    agent = next(item for item in engine.agents if item.id == "wang")
    for resource in engine.resources.at_location(location):
        if engine.interactions.service_needs_a_look(agent, location, resource, engine):
            return resource.id
    raise AssertionError("no counter goods left to order at " + location)


def step(description, action):
    return {"description": description, "action": action, "expected_outcome": []}


def plan_payload(steps):
    return {"focus": "去咖啡馆坐一会儿", "motive": "想喝点热的", "review_after_minutes": 60,
            "goal_ids": [], "steps": steps}


def test_the_plan_context_carries_the_ids_of_places_it_may_go_to(tmp_path):
    """Otherwise a step for another location can only cite invented devices."""
    async def scenario():
        engine = await make_engine(tmp_path)
        agent = next(item for item in engine.agents if item.id == "wang")
        elsewhere, real_process = location_with_process(engine)
        agent.state.current_location = "home_wang"

        context = await engine.context.build(
            agent, "life_plan", engine,
            intention={"interaction_type": "operate", "target_location": elsewhere},
        )

        entries = {item["id"]: item for item in context["known_locations"]}
        assert elsewhere in entries, "a place the agent plans to work in must be visible"
        assert real_process in {
            item["id"] for item in entries[elsewhere]["facilities"]["processes"]
        }

    run(scenario())


def test_a_step_citing_a_device_that_does_not_exist_is_asked_again(tmp_path):
    """The ids are in the context; naming the bad one lets the model use them."""
    async def scenario():
        engine = await make_engine(tmp_path)
        agent = next(item for item in engine.agents if item.id == "wang")
        elsewhere, real_process = location_with_process(engine)
        calls = stub_planner(engine, [
            plan_payload([
                step("走过去", {"interaction_type": "move", "content": "走过去",
                                "location": elsewhere}),
                step("用设备做一件事", {"interaction_type": "operate", "content": "用设备做一件事",
                                       "process_id": "machine_that_does_not_exist"}),
            ]),
            plan_payload([
                step("走过去", {"interaction_type": "move", "content": "走过去",
                                "location": elsewhere}),
                step("用设备做一件事", {"interaction_type": "operate", "content": "用设备做一件事",
                                       "process_id": real_process}),
            ]),
        ])

        plan = await engine.planner.create_life_plan(agent, engine, {"sim_time": "第1天 09:00"})

        plan_calls = [call for call in calls if call["task"] == "life_plan"]
        assert len(plan_calls) == 2, "one rewrite should be requested"
        feedback = plan_calls[1]["payload"]["rejected_steps"]
        assert "machine_that_does_not_exist" in str(feedback)
        assert [item.action.get("process_id") for item in plan.steps] == [None, real_process]
        assert plan.invalid_steps == []

    run(scenario())


def test_a_step_that_stays_impossible_is_left_out_of_the_plan(tmp_path):
    """A committed plan must not contain a step that can only fail on arrival."""
    async def scenario():
        engine = await make_engine(tmp_path)
        agent = next(item for item in engine.agents if item.id == "wang")
        elsewhere, _real = location_with_process(engine)
        impossible = step("用设备做一件事", {"interaction_type": "operate",
                                             "content": "用设备做一件事",
                                             "process_id": "machine_that_does_not_exist"})
        calls = stub_planner(engine, [
            plan_payload([
                step("走过去", {"interaction_type": "move", "content": "走过去",
                                "location": elsewhere}),
                impossible,
            ]),
        ])

        plan = await engine.planner.create_life_plan(agent, engine, {"sim_time": "第1天 09:00"})

        assert len([call for call in calls if call["task"] == "life_plan"]) == 2
        assert [item.description for item in plan.steps] == ["走过去"]
        assert plan.invalid_steps and "过程不存在" in plan.invalid_steps[0]

    run(scenario())


def test_a_step_that_never_names_its_process_is_caught_before_it_is_committed(tmp_path):
    """The measured cause of death: 'operate' with no process_id at all."""
    async def scenario():
        engine = await make_engine(tmp_path)
        agent = next(item for item in engine.agents if item.id == "wang")
        elsewhere, real_process = location_with_process(engine)
        calls = stub_planner(engine, [
            plan_payload([
                step("走过去", {"interaction_type": "move", "content": "走过去",
                                "location": elsewhere}),
                step("打开店门", {"interaction_type": "operate", "content": "打开店门",
                                 "duration_minutes": 15}),
            ]),
            plan_payload([
                step("走过去", {"interaction_type": "move", "content": "走过去",
                                "location": elsewhere}),
                step("打开店门", {"interaction_type": "operate", "content": "打开店门",
                                 "process_id": real_process}),
            ]),
        ])

        plan = await engine.planner.create_life_plan(agent, engine, {"sim_time": "第1天 09:00"})

        plan_calls = [call for call in calls if call["task"] == "life_plan"]
        assert len(plan_calls) == 2
        assert plan_calls[1]["payload"]["rejected_steps"][0]["problems"] == [
            {"field": "process_id", "value": "", "reason": "过程不存在"},
        ]
        assert [item.action.get("process_id") for item in plan.steps] == [None, real_process]

    run(scenario())


def test_ordering_where_the_plan_never_goes_is_not_committed(tmp_path):
    """Arriving is seeing, so the plan must actually arrive before it orders."""
    async def scenario():
        engine = await make_engine(tmp_path)
        agent = next(item for item in engine.agents if item.id == "wang")
        item = counter_item(engine, "cafe")

        calls = stub_planner(engine, [
            plan_payload([
                step("点一份午餐", {"interaction_type": "request_service",
                                    "content": "点一份午餐", "location": "cafe",
                                    "resource_id": item}),
            ]),
            plan_payload([
                step("走到咖啡馆", {"interaction_type": "move", "content": "走到咖啡馆",
                                    "location": "cafe"}),
                step("点一份午餐", {"interaction_type": "request_service",
                                    "content": "点一份午餐", "location": "cafe",
                                    "resource_id": item}),
            ]),
        ])

        plan = await engine.planner.create_life_plan(agent, engine, {"sim_time": "第1天 09:00"})

        plan_calls = [call for call in calls if call["task"] == "life_plan"]
        assert len(plan_calls) == 2, "one rewrite should be requested"
        problems = plan_calls[1]["payload"]["rejected_steps"][0]["problems"]
        assert [item["reason"] for item in problems] == ["尚未观察到可服务工作人员；请先查看店内情况"]
        assert [step_.action.get("interaction_type") for step_ in plan.steps] == [
            "move", "request_service",
        ]

    run(scenario())
