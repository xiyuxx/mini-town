import asyncio
import json
from collections import Counter

from backend.config import config
from backend.town.cognition import DecisionProposal, LifePlan, PlanStep
from backend.town.engine import SimulationEngine
from backend.town.experience import ExperienceEvent, MemoryCandidate


def run(coro):
    return asyncio.run(coro)


async def settle_realtime(engine):
    """Let deferred realtime work finish, as the 6-second tick budget would."""
    for _ in range(6):
        await asyncio.sleep(0)
    pending = [task for _, task in engine._dialogue_turn_tasks.values() if not task.done()]
    pending += [task for task in engine._background_tasks if not task.done()]
    if pending:
        await asyncio.wait(pending, timeout=5)


async def park_idle_agents(engine, count):
    """Put the first `count` agents in the park, idle, on separate cells."""
    chosen = engine.agents[:count]
    engine.agents = list(chosen)
    engine.routine_planner.next_action = lambda _agent, _engine: None
    for index, agent in enumerate(chosen):
        agent.state.current_location = "park"
        agent.state.x, agent.state.y = 9 + index, 8
        agent.state.status = "IDLE"
    return chosen


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


def install_schedule_plan_stub(engine):
    calls = {}

    async def create_plan(agent, _engine, _context, replan_reason=""):
        calls.setdefault(agent.id, []).append(_engine.get_sim_timestamp())
        commitment = agent.get_current_commitment(
            _engine.day, _engine.hour, _engine.minute
        )
        steps = []
        if commitment:
            location = str(commitment.get("location", ""))
            if location and location != agent.state.current_location:
                steps.append(PlanStep(
                    description=f"前往{location}",
                    action={
                        "interaction_type": "move", "location": location,
                        "content": f"前往{location}",
                    },
                ))
            activity = str(commitment.get("activity") or commitment.get("label") or "").strip()
            if activity:
                steps.append(PlanStep(
                    description=activity,
                    action={
                        "interaction_type": "wait", "content": activity,
                        "duration_minutes": int(commitment.get("expected_duration") or 15),
                    },
                ))
        if not steps:
            steps.append(PlanStep(
                description="保持当前安排",
                action={
                    "interaction_type": "wait", "content": "保持当前安排",
                    "duration_minutes": 30,
                },
            ))
        return LifePlan(
            focus=steps[-1].description, motive="测试中的确定性计划",
            created_at=_engine.get_sim_timestamp(),
            review_at=_engine.get_sim_timestamp() + 120,
            steps=steps, replan_reason=replan_reason,
        )

    engine.planner.create_life_plan = create_plan
    return calls


def test_student_arrival_completes_travel_without_duplicate_activity(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        calls = install_schedule_plan_stub(engine)
        events = []
        student = next(agent for agent in engine.agents if agent.id == "ming")
        for _ in range(60):
            await engine.tick()
            events.extend(engine._last_events[-10:])
            if student.state.current_location == "school" and student.state.current_action == "上课":
                break
        assert student.state.current_location == "school"
        assert student.state.current_action == "上课"
        call_times = calls[student.id]
        assert all(
            sum(start <= timestamp < start + 60 for timestamp in call_times) <= 3
            for start in call_times
        )
        unique_events = {event.get("id"): event for event in events if event.get("id")}
        contents = [event.get("content", "") for event in unique_events.values()]
        assert sum("开始去学校" in content for content in contents) <= 1

    run(scenario())


def test_routine_events_do_not_pollute_long_term_memory(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        install_schedule_plan_stub(engine)
        for _ in range(75):
            await engine.tick()
        for agent_id in ("hua", "wang", "li", "ming"):
            memories = await engine.memory.get_recent(agent_id, limit=100)
            assert not any(memory.content.startswith(("开始了：", "完成了：", "出发前往", "到达")) for memory in memories)

    run(scenario())


def test_doctor_morning_run_has_bounded_duration(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        install_schedule_plan_stub(engine)
        doctor = next(agent for agent in engine.agents if agent.id == "li")
        timeline = []
        for _ in range(45):
            await engine.tick()
            timeline.append((engine.hour, engine.minute, doctor.state.current_action, doctor.state.status))
        running_times = [hour * 60 + minute for hour, minute, action, _ in timeline if action == "起床晨跑"]
        assert running_times
        # Someone greeting the doctor in the park pauses the run and the run
        # resumes, so the span covers a conversation. A run that never finished
        # would be an order of magnitude longer, which is what this bound is for.
        assert max(running_times) - min(running_times) <= 60
        assert timeline[-1][2] != "起床晨跑", "the run must end, not carry on forever"
        assert not any(not action.strip() for _, _, action, status in timeline if status == "ACTING")

    run(scenario())


def test_interrupted_activity_resumes_if_commitment_is_still_current(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        agent = engine.agents[0]
        commitment = agent.get_current_commitment(engine.day, engine.hour, engine.minute)
        await agent.start_activity("整理房间", 30, engine, commitment_id=commitment["id"])
        agent._activity_ticks = 4
        description = agent.interrupt_activity()
        assert description == "整理房间"
        assert agent.resume_interrupted_activity(commitment["id"])
        assert agent.state.status == "ACTING"
        assert agent._activity_ticks == 4

    run(scenario())


def test_severe_weather_and_hunger_remain_an_llm_choice(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        artist = next(agent for agent in engine.agents if agent.id == "hua")
        artist.llm.fallback = False
        artist.state.current_location = "cafe"
        artist.state.needs["hunger"] = 92
        engine.weather = {"condition": "大雨", "temperature": 17, "wind": "大风"}
        decision = engine.town_agent.choose_rule_based_action(
            artist, engine.day, engine.hour, engine.minute
        )
        assert decision is None

    run(scenario())


def test_memory_encoder_rejects_wrong_time_periods(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        owner = next(agent for agent in engine.agents if agent.id == "liu")
        engine.hour, engine.minute = 11, 15
        engine.llm.fallback = False

        async def hallucinated_memory(*args, **kwargs):
            return "下午关店后，我坐在窗边读《陶庵梦忆》。"

        engine.llm.encode_experience_memory = hallucinated_memory
        candidate = MemoryCandidate(
            event=ExperienceEvent(
                type="activity_outcome", location="bookstore",
                facts={"activity": "读《陶庵梦忆》"}, actors=[owner.id],
            ),
            score=0.4, tier="working", importance=3,
            fact_summary="在大刘书店读《陶庵梦忆》",
            interpretation_hint="读书让我安静下来", emotion="满足",
        )
        encoded = await engine.memory_formation._encode(owner, candidate, engine)
        assert "下午" not in encoded
        assert "关店后" not in encoded
        assert "陶庵梦忆" in encoded
        assert not engine.memory_formation._time_consistent("这样的午后很安静", "读书", 11)
        assert engine.memory_formation._time_period(11) == "上午"

    run(scenario())


def test_decision_proposal_normalizes_real_model_shape():
    proposal = DecisionProposal.from_dict({
        "assessment": "当前信息有限",
        "options": [{
            "id": "prepare",
            "action": {"interaction_type": "wait", "content": "整理晨间安排"},
            "expected_outcome": "为接下来的日程做好准备",
            "risks": "可能占用少量时间",
            "assumptions": "当前没有紧急事项",
        }],
        "preferred_option_id": "prepare",
        "mental_update": {},
    })
    assert proposal.assessment.uncertainties == [{"question": "当前信息有限"}]
    assert proposal.options[0].expected_effects == [
        {"description": "为接下来的日程做好准备"}
    ]
    assert proposal.options[0].risks == [{"description": "可能占用少量时间"}]
    assert proposal.options[0].assumptions == ["当前没有紧急事项"]


def test_planner_json_call_retries_truncation_with_larger_budget(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        budgets = []

        async def adaptive(*args, **kwargs):
            budgets.append(kwargs["max_tokens"])
            if kwargs["max_tokens"] == 4096:
                return {"choices": [{"finish_reason": "length", "message": {"content": ""}}]}
            return {"choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}]}

        engine.planner.llm._chat_raw = adaptive
        result = await engine.planner._json_call("输出JSON", {"value": 1})
        assert result == {"ok": True}
        assert budgets == [4096, 8192]

    run(scenario())


def test_decision_failure_defers_agent_without_stopping_world(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        engine.llm.fallback = False
        started_at = engine.get_sim_timestamp()

        async def fail_plan(*args, **kwargs):
            raise RuntimeError("planner returned empty content")

        engine.planner.create_life_plan = fail_plan
        engine.running = True
        await engine.tick()

        assert engine.get_sim_timestamp() == started_at + 5
        assert engine.running
        assert engine.get_state()["health"]["simulation"] == "degraded"
        assert all(agent.state.status == "IDLE" for agent in engine.agents)
        assert any(event.get("type") == "decision_deferred" for event in engine._last_events)
        assert not any(event.get("type") == "activity_started" for event in engine._last_events)

    run(scenario())


def test_work_on_goal_requires_real_goal_and_explicit_progress(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        agent = next(item for item in engine.agents if item.id == "wang")
        agent.sync_schedule_goals(1, 6, 0)
        goal = agent.mental_state.active_goal_object
        assert goal is not None

        missing = engine.interactions.validate(
            agent,
            {"interaction_type": "work_on_goal", "content": "继续准备"},
            engine,
        )
        assert not missing.feasible
        assert any("goal_id" in reason for reason in missing.reasons)

        valid = engine.interactions.validate(
            agent,
            {
                "interaction_type": "work_on_goal",
                "content": "洗漱完毕",
                "goal_id": goal.id,
                "progress_delta": 1.0,
            },
            engine,
        )
        assert valid.feasible
        effects = engine.interactions.apply_effects(
            agent, valid.resolved_action, valid.expected_effects, engine.get_sim_timestamp()
        )
        assert effects[0]["completed"] is True
        agent.complete_goal_commitment(goal.id, 1)
        assert agent.get_current_commitment(1, 6, 0)["status"] == "completed"

    run(scenario())


def test_move_target_is_resolved_from_supported_goal(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        agent = next(item for item in engine.agents if item.id == "wang")
        engine.hour, engine.minute = 6, 35
        agent.sync_schedule_goals(1, 6, 35)
        goal = agent.mental_state.active_goal_object
        assert goal is not None

        result = engine.interactions.validate(
            agent,
            {
                "interaction_type": "move",
                "content": "出门晨练",
                "supports_goal_ids": [goal.id],
            },
            engine,
        )
        assert result.feasible
        assert result.resolved_action["location"] == "park"

    run(scenario())


def test_public_action_text_hides_ids_and_assumptions(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        doctor = next(item for item in engine.agents if item.id == "li")
        text = engine._public_action_text(
            doctor,
            "吃家里剩余的食物（food_home_li），提前完成早餐（假设厨房可用）",
        )
        assert "food_home_li" not in text
        assert "假设" not in text
        assert "食物" in text

    run(scenario())


def test_life_plan_steps_continue_without_replanning_each_tick(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        agent = next(item for item in engine.agents if item.id == "wang")
        engine.agents = [agent]
        calls = {"count": 0}

        async def create_plan(_agent, _engine, _context, replan_reason=""):
            calls["count"] += 1
            return LifePlan(
                focus="完成两段连续活动", motive="保持行动连续",
                created_at=_engine.get_sim_timestamp(),
                review_at=_engine.get_sim_timestamp() + 90,
                steps=[
                    PlanStep(description="整理桌面", action={
                        "interaction_type": "wait", "content": "整理桌面",
                        "duration_minutes": 10,
                    }),
                    PlanStep(description="写下安排", action={
                        "interaction_type": "wait", "content": "写下安排",
                        "duration_minutes": 10,
                    }),
                ],
            )

        engine.planner.create_life_plan = create_plan
        actions = []
        for _ in range(4):
            await engine.tick()
            actions.append(agent.state.current_action)

        assert calls["count"] == 1
        assert "整理桌面" in actions
        assert "写下安排" in actions
        assert agent.mental_state.life_plan.steps[0].status == "completed"
        assert agent.mental_state.life_plan.steps[1].status == "executing"

    run(scenario())


def test_blocked_plan_replans_with_reason(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        agent = next(item for item in engine.agents if item.id == "wang")
        engine.agents = [agent]
        reasons = []

        async def create_plan(_agent, _engine, _context, replan_reason=""):
            reasons.append(replan_reason)
            action = (
                {"interaction_type": "move", "content": "去不存在的地方", "location": "missing"}
                if len(reasons) == 1 else
                {"interaction_type": "wait", "content": "重新核对现有安排", "duration_minutes": 10}
            )
            return LifePlan(
                focus="处理受阻安排", motive="根据结果调整",
                created_at=_engine.get_sim_timestamp(),
                review_at=_engine.get_sim_timestamp() + 60,
                steps=[PlanStep(description=action["content"], action=action)],
                replan_reason=replan_reason,
            )

        engine.planner.create_life_plan = create_plan
        await engine.tick()

        assert len(reasons) == 2
        assert "目标地点不存在" in reasons[1]
        assert agent.mental_state.life_plan.status == "active"
        assert agent.state.current_action == "重新核对现有安排"

    run(scenario())


def test_invalid_action_is_blocked_without_fake_rest(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        agent = engine.agents[0]
        plan = LifePlan(
            focus="尝试未知动作", motive="测试失败路径",
            created_at=engine.get_sim_timestamp(),
            review_at=engine.get_sim_timestamp() + 30,
            steps=[PlanStep(description="尝试未知动作", action={
                "interaction_type": "unknown_action", "content": "尝试未知动作",
            })],
        )
        agent.mental_state.life_plan = plan
        decision = dict(plan.current_step.action, plan_step_id=plan.current_step.id)
        events = await engine._execute_decision(agent, decision, engine.get_sim_time_str())

        assert events[0]["type"] == "action_blocked"
        assert plan.status == "blocked"
        assert agent.state.status == "IDLE"
        assert agent.state.current_action not in {"稍作休息", "休息中"}

    run(scenario())


def test_fallback_mode_defers_without_inventing_activity(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        started_at = engine.get_sim_timestamp()
        await engine.tick()

        assert engine.get_sim_timestamp() == started_at + 5
        assert all(agent.state.status == "IDLE" for agent in engine.agents)
        assert any(event["type"] == "decision_deferred" for event in engine._last_events)
        assert not any(event["type"] in {"activity_started", "move_start"} for event in engine._last_events)

    run(scenario())


def test_low_stakes_lateness_does_not_create_urgency(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        retiree = next(agent for agent in engine.agents if agent.id == "wang")
        engine.hour, engine.minute = 7, 50
        event = ExperienceEvent(
            type="arrival", location="home_wang", actors=[retiree.id],
            source="commitment", expected=False,
            facts={"purpose": "吃早餐", "lateness": 20, "weather": "晴"},
        )
        summary, interpretation, emotion, intention = engine.memory_formation._summarize(
            retiree, event, engine
        )
        assert "晚了20分钟" in summary
        assert emotion == ""
        assert intention == ""
        assert "紧迫" in interpretation

    run(scenario())


def test_realtime_tick_starts_a_routine_task_without_llm(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        agent = engine.agents[0]
        engine.agents = [agent]
        engine._realtime_llm = True

        async def unavailable_plan(*_args, **_kwargs):
            raise AssertionError("routine scheduling must not request an LLM plan")

        engine.planner.create_life_plan = unavailable_plan
        started_at = engine.get_sim_timestamp()
        await asyncio.wait_for(engine.tick(), timeout=0.5)

        assert engine.get_sim_timestamp() == started_at + 5
        assert engine._planning_tasks == {}
        assert agent.state.status == "ACTING"
        task = next(item for item in engine.tasks.to_dict() if item["assignee_id"] == agent.id)
        assert task["source"] == "schedule"
        assert task["status"] == "active"

    run(scenario())


def test_declared_meal_schedule_uses_consume_interaction(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        retiree = next(agent for agent in engine.agents if agent.id == "wang")
        engine.agents = [retiree]
        engine.hour, engine.minute = 7, 30

        action = engine.routine_planner.next_action(retiree, engine)

        assert action is not None
        assert action["interaction_type"] == "consume"
        assert action["location"] == "home_wang"
        result = engine.interactions.validate(retiree, action, engine)
        assert result.feasible
        assert result.resolved_action["resource_id"] == "food_home_wang"

    run(scenario())


def test_world_staggers_meals_and_has_no_bookstore_grocery_trip(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        lunches = {}
        for agent in engine.agents:
            meal = next(
                (item for item in agent.schedule
                 if item.kind == "meal" and 11 * 60 <= item.start_minutes < 14 * 60),
                None,
            )
            assert meal is not None
            lunches[agent.id] = (meal.start_minutes, meal.location)
            assert not (agent.id == "wang" and meal.location == "bookstore")

        assert len(set(lunches.values())) >= 4
        wang_before_lunch = next(
            item for item in engine.agents[0].schedule
            if item.hour == 11 and item.minute == 30
        )
        assert wang_before_lunch.location == "fresh_market"
        assert "买菜" in wang_before_lunch.label

    run(scenario())


def test_world_initial_needs_are_individualized(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        hunger_values = {agent.state.needs["hunger"] for agent in engine.agents}

        assert len(hunger_values) > 1
        assert max(hunger_values) < 30

    run(scenario())


def test_new_idle_encounter_starts_a_bounded_dialogue(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        first, second = engine.agents[:2]
        engine.agents = [first, second]
        first.state.current_location = second.state.current_location = "park"
        first.state.x = second.state.x = 9
        first.state.y = second.state.y = 8
        first.state.status = second.state.status = "IDLE"

        sim_time = engine.get_sim_time_str()
        events = await engine._detect_encounters(sim_time)
        await engine._update_colocation(sim_time)
        await engine._queue_colocated_dialogues()
        dialogue_events = await engine._run_group_dialogues(sim_time)

        assert any(event["type"] == "encounter" for event in events)
        assert any(event["type"] == "dialogue_start" for event in dialogue_events)
        assert engine.dialogue.participant_ids() == {first.id, second.id}

    run(scenario())


def test_realtime_shared_place_starts_a_dialogue_without_sharing_a_cell(tmp_path):
    """An idle pair in the same place must not need the exact same cell.

    Only the same-cell trigger used to run in realtime, so a shared place never
    turned into a conversation (0 dialogues in two simulated days).
    """
    async def scenario():
        engine = await make_engine(tmp_path)
        engine._realtime_llm = True
        first, second = await park_idle_agents(engine, 2)
        second.state.x, second.state.y = 10, 9

        await engine.tick()

        assert engine.dialogue.participant_ids() == {first.id, second.id}

    run(scenario())


def test_dialogue_pacing_avoids_the_pair_that_just_talked(tmp_path):
    """Selection must reach agents who have not talked, not the same pair."""
    async def scenario():
        engine = await make_engine(tmp_path)
        engine._realtime_llm = True
        first, second, third = await park_idle_agents(engine, 3)
        engine._pair_last_talk[tuple(sorted((first.id, second.id)))] = engine.get_sim_timestamp()

        await engine.tick()

        talking = engine.dialogue.participant_ids()
        assert third.id in talking
        assert {first.id, second.id} != talking

    run(scenario())


def test_dialogue_cooldown_stops_an_immediate_repeat(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        engine._realtime_llm = True
        first, second = await park_idle_agents(engine, 2)
        await engine.tick()
        assert engine.dialogue.participant_ids() == {first.id, second.id}

        for _ in range(30):
            await engine.tick()
            await settle_realtime(engine)
            if not engine.dialogue._active_dialogues:
                break
        assert not engine.dialogue._active_dialogues
        sessions = len(await engine.dialogue_store.list_recent(limit=50))

        for _ in range(5):
            await engine.tick()
            await settle_realtime(engine)

        assert len(await engine.dialogue_store.list_recent(limit=50)) == sessions

    run(scenario())


def test_agent_daily_dialogue_budget_caps_the_day(tmp_path, monkeypatch):
    async def scenario():
        monkeypatch.setattr(config, "DIALOGUE_PAIR_COOLDOWN_MINUTES", 0)
        monkeypatch.setattr(config, "DIALOGUE_MAX_PER_AGENT_PER_DAY", 1)
        engine = await make_engine(tmp_path)
        engine._realtime_llm = True
        first, second = await park_idle_agents(engine, 2)

        for _ in range(60):
            await engine.tick()
            await settle_realtime(engine)

        started = Counter()
        for session in await engine.dialogue_store.list_recent(limit=100):
            members = session["participants"]
            members = json.loads(members) if isinstance(members, str) else members
            started.update(members)
        assert started[first.id] == 1
        assert started[second.id] == 1

    run(scenario())


def test_travelling_together_is_greeted_once(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        first, second = engine.agents[:2]
        engine.agents = [first, second]
        for agent in (first, second):
            agent.state.current_location = "in_transit"
            agent.state.status = "MOVING"
            agent.state.x, agent.state.y = 8, 7
            agent._movement_path = [(9, 8), (10, 9), (11, 9), (12, 9), (13, 9), (14, 9)]
        sim_time = engine.get_sim_time_str()

        await engine._process_moving(first, sim_time)
        events = await engine._process_moving(second, sim_time)
        assert [event["type"] for event in events] == ["encounter"]

        await engine._process_moving(first, sim_time)
        assert await engine._process_moving(second, sim_time) == []

    run(scenario())


def test_planning_context_survives_company_nearby(tmp_path):
    """Planning must not fail just because another agent is in the same place.

    The summary omitted recent_interactions for every task type except
    action_decision/dialogue, and the shared context builder subscripts that key
    over nearby agents — so live planning raised KeyError exactly when company
    was present, which is when social plans would be made.
    """
    async def scenario():
        engine = await make_engine(tmp_path)
        first, second = engine.agents[:2]
        engine.agents = [first, second]
        for agent in (first, second):
            agent.state.current_location = "park"
            agent.state.x, agent.state.y = 9, 8
            agent.state.status = "IDLE"

        for task_type in ("life_plan", "intention_proposal", "routine_selection", "reflection"):
            context = await first.build_planning_context(
                engine.get_sim_time_str(), engine, task_type=task_type,
            )
            assert "recent_interactions" in context["mental_state"]

    run(scenario())


def test_finished_realtime_dialogue_releases_both_speakers(tmp_path):
    """A conversation that ends must return both speakers to the world.

    Closing events arrive after the session has already left _active_dialogues,
    so the stale-result guard used to discard them and leave both participants
    in SPEAKING/LISTENING forever — after which they never act or talk again.
    """
    async def scenario():
        engine = await make_engine(tmp_path)
        engine._realtime_llm = True
        first, second = await park_idle_agents(engine, 2)

        session_id = ""
        settled_after_end = 0
        for _ in range(30):
            await engine.tick()
            await settle_realtime(engine)
            if engine.dialogue._active_dialogues:
                session_id = next(iter(engine.dialogue._active_dialogues.values())).id
                settled_after_end = 0
            elif session_id:
                # The engine commits the closing turn on the tick after it finished.
                settled_after_end += 1
                if settled_after_end >= 2:
                    break

        assert session_id
        assert not engine.dialogue._active_dialogues
        assert first.state.status not in {"SPEAKING", "LISTENING"}
        assert second.state.status not in {"SPEAKING", "LISTENING"}
        assert engine._interactions[session_id].status == "completed"

    run(scenario())


def test_realtime_mode_replans_after_a_blocked_plan_step(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        artist = next(agent for agent in engine.agents if agent.id == "hua")
        engine.agents = [artist]
        engine.llm.fallback = False
        engine._realtime_llm = True
        artist.mental_state.life_plan = LifePlan(
            focus="取得咖啡", motive="解决饥饿",
            created_at=engine.get_sim_timestamp(), review_at=engine.get_sim_timestamp() + 30,
            status="blocked", replan_reason="观察到服务台当前无人值守",
        )
        reasons = []

        async def create_plan(_agent, _engine, _context, replan_reason=""):
            reasons.append(replan_reason)
            return LifePlan(
                focus="改去其他地点", motive="服务不可用后调整",
                created_at=_engine.get_sim_timestamp(), review_at=_engine.get_sim_timestamp() + 30,
                steps=[PlanStep(description="查看替代安排", action={
                    "interaction_type": "wait", "content": "查看替代安排", "duration_minutes": 5,
                })],
            )

        engine.planner.create_life_plan = create_plan
        await engine.tick()
        await engine._planning_tasks[artist.id][1]
        await engine.tick()

        assert reasons == ["观察到服务台当前无人值守"]
        assert artist.state.status == "ACTING"
        assert artist.state.current_action == "查看替代安排"

    run(scenario())
