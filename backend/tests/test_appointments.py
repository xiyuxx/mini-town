"""Arrangements made in conversation: kept, or broken with consequences."""

import asyncio

from backend.config import config
from backend.town.engine import SimulationEngine
from backend.town.schedule import ScheduleItem
from backend.town.tasks import Task


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


async def settle(engine):
    for _ in range(6):
        await asyncio.sleep(0)
    pending = [task for _, task in engine._dialogue_turn_tasks.values() if not task.done()]
    pending += [task for task in engine._background_tasks if not task.done()]
    if pending:
        await asyncio.wait(pending, timeout=5)


def proposal(**overrides):
    base = {"with_name": "小美", "day_offset": 0, "hour": 7, "minute": 0,
            "location": "park", "activity": "下棋"}
    base.update(overrides)
    return base


def test_an_arrangement_becomes_a_task_for_both_sides(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        speaker = next(agent for agent in engine.agents if agent.id == "wang")
        partner = next(agent for agent in engine.agents if agent.id == "mei")
        for agent in (speaker, partner):
            agent.schedule = []

        accepted, note = await engine.create_appointment(speaker, [partner], proposal())

        assert accepted and "公园" in note
        appointment = next(iter(engine._appointments.values()))
        for agent in (speaker, partner):
            task = engine.tasks.tasks[f"appointment:{appointment['id']}:{agent.id}"]
            assert task.location_id == "park"
            assert task.deadline_at == appointment["meeting_at"] + config.APPOINTMENT_GRACE_MINUTES
        fact = next(item for item in engine.fact_ledger.known_for(speaker.id, limit=20)
                    if item.type == "appointment_made")
        assert set(fact.known_by) == {speaker.id, partner.id}
        assert fact.details["partner_id"] == partner.id

    run(scenario())


def test_invalid_arrangements_are_refused(tmp_path, monkeypatch):
    async def scenario():
        engine = await make_engine(tmp_path)
        speaker = next(agent for agent in engine.agents if agent.id == "wang")
        partner = next(agent for agent in engine.agents if agent.id == "mei")
        for agent in (speaker, partner):
            agent.schedule = []

        # Somebody who is not in the conversation.
        refused, note = await engine.create_appointment(
            speaker, [partner], proposal(with_name="大刘"),
        )
        assert not refused and "要约的人" in note

        # A place that does not exist.
        refused, note = await engine.create_appointment(
            speaker, [partner], proposal(location="moon_base"),
        )
        assert not refused and "没有这个地方" in note

        # Too soon to be a real plan.
        now = engine.get_sim_timestamp()
        refused, _ = await engine.create_appointment(
            speaker, [partner],
            proposal(hour=(now + 1) // 60, minute=(now + 1) % 60),
        )
        assert not refused
        assert engine._appointments == {}

        # One promise is the limit here, and a second slot clashes with it.
        monkeypatch.setattr(config, "APPOINTMENT_MAX_PENDING_PER_AGENT", 1)
        accepted, _ = await engine.create_appointment(speaker, [partner], proposal())
        assert accepted
        refused, note = await engine.create_appointment(
            speaker, [partner], proposal(hour=8),
        )
        assert not refused and "约定" in note

    run(scenario())


def test_keeping_the_appointment_brings_them_together(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        engine._realtime_llm = True
        speaker = next(agent for agent in engine.agents if agent.id == "wang")
        partner = next(agent for agent in engine.agents if agent.id == "mei")
        engine.agents = [speaker, partner]
        for agent in (speaker, partner):
            agent.schedule = []
            agent.routine_blocks = []
            agent.state.status = "IDLE"
        speaker.state.current_location = "park"
        speaker.state.x, speaker.state.y = 9, 8
        partner.state.current_location = "cafe"
        partner.state.x, partner.state.y = 6, 4

        accepted, _ = await engine.create_appointment(
            speaker, [partner],
            proposal(hour=engine.hour, minute=engine.minute + 30),
        )
        assert accepted
        appointment = next(iter(engine._appointments.values()))

        for _ in range(10):
            await engine.tick()
            await settle(engine)
            if engine.dialogue.participant_ids() == {speaker.id, partner.id}:
                break

        assert speaker.state.current_location == "park"
        assert partner.state.current_location == "park"
        assert appointment["status"] == "pending"
        assert engine.dialogue.participant_ids() == {speaker.id, partner.id}

    run(scenario())


def test_fixed_duties_block_promises_that_would_be_broken(tmp_path):
    """Nobody should promise a time they are already committed to."""
    async def scenario():
        engine = await make_engine(tmp_path)
        speaker = next(agent for agent in engine.agents if agent.id == "wang")
        partner = next(agent for agent in engine.agents if agent.id == "mei")
        busy = next(
            (item for item in speaker.schedule
             if not item.is_flexible_slot and item.hour >= config.SIM_START_HOUR + 1),
            None,
        )
        assert busy is not None, "expected a fixed duty in the schedule"

        refused, note = await engine.create_appointment(
            speaker, [partner],
            proposal(hour=busy.hour, minute=busy.minute, day_offset=0),
        )

        assert not refused and "固定安排" in note
        assert engine._appointments == {}

    run(scenario())


def test_a_promise_pulls_an_agent_out_of_what_it_was_doing(tmp_path):
    """An uncommitted activity must not survive a promise it would miss."""
    async def scenario():
        engine = await make_engine(tmp_path)
        engine._realtime_llm = True
        speaker = next(agent for agent in engine.agents if agent.id == "wang")
        partner = next(agent for agent in engine.agents if agent.id == "mei")
        engine.agents = [speaker, partner]
        for agent in (speaker, partner):
            agent.schedule = []
            agent.routine_blocks = []
        speaker.state.current_location = "park"
        speaker.state.x, speaker.state.y = 9, 8
        speaker.state.status = "IDLE"
        partner.state.current_location = "restaurant"
        partner.state.x, partner.state.y = 16, 4
        partner.state.status = "ACTING"
        partner._activity_ticks = 200
        partner._activity_desc = "在餐厅消磨时间"
        partner._activity_commitment_id = None

        accepted, _ = await engine.create_appointment(
            speaker, [partner],
            proposal(hour=engine.hour, minute=engine.minute + 30),
        )
        assert accepted
        appointment = next(iter(engine._appointments.values()))

        for _ in range(12):
            await engine.tick()
            await settle(engine)
            if appointment["status"] != "pending":
                break

        assert appointment["status"] == "honored"
        assert partner.state.current_location == "park"

    run(scenario())


def test_a_promise_made_in_free_time_outranks_the_schedule_board(tmp_path):
    """Agreed in free time, so ordinary duties must not eat the slot later."""
    async def scenario():
        engine = await make_engine(tmp_path)
        engine._realtime_llm = True
        speaker = next(agent for agent in engine.agents if agent.id == "wang")
        partner = next(agent for agent in engine.agents if agent.id == "mei")
        engine.agents = [speaker, partner]
        for agent in (speaker, partner):
            agent.schedule = []
            agent.routine_blocks = []
        speaker.state.current_location = "home_wang"
        speaker.state.x, speaker.state.y = 1, 1
        speaker.state.status = "IDLE"
        partner.state.current_location = "cafe"
        partner.state.x, partner.state.y = 6, 4
        partner.state.status = "IDLE"

        accepted, _ = await engine.create_appointment(
            speaker, [partner],
            proposal(hour=engine.hour, minute=engine.minute + 40),
        )
        assert accepted
        appointment = next(iter(engine._appointments.values()))

        # A duty for the same slot appears after the promise was made.
        now = engine.get_sim_timestamp()
        engine.tasks.upsert(Task(
            id="duty:clinic", title="去诊所帮忙", assignee_id=speaker.id,
            source="schedule", location_id="clinic", interaction_type="wait",
            duration_minutes=60, earliest_at=now,
            deadline_at=appointment["meeting_at"] + 60, priority=1.0,
            payload={"commitment_id": "duty:clinic"}, created_at=now,
        ))

        for _ in range(14):
            await engine.tick()
            await settle(engine)
            if appointment["status"] != "pending":
                break

        assert appointment["status"] == "honored"
        assert speaker.state.current_location == "park"

    run(scenario())


def test_an_emergency_keeps_an_agent_away(tmp_path):
    """The one thing allowed to break a promise is a more urgent need."""
    async def scenario():
        engine = await make_engine(tmp_path)
        engine._realtime_llm = True
        waiter = next(agent for agent in engine.agents if agent.id == "mei")
        busy = next(agent for agent in engine.agents if agent.id == "wang")
        engine.agents = [waiter, busy]
        for agent in (waiter, busy):
            agent.schedule = []
            agent.routine_blocks = []
        waiter.state.current_location = "park"
        waiter.state.x, waiter.state.y = 9, 8
        waiter.state.status = "IDLE"
        busy.state.current_location = "restaurant"
        busy.state.x, busy.state.y = 16, 4
        busy.state.status = "IDLE"

        accepted, _ = await engine.create_appointment(
            waiter, [busy],
            proposal(with_name="老王", hour=engine.hour, minute=engine.minute + 30),
        )
        assert accepted
        appointment = next(iter(engine._appointments.values()))

        for _ in range(14):
            busy.state.needs["hunger"] = 99      # a standing emergency
            await engine.tick()
            await settle(engine)
            if appointment["status"] != "pending":
                break

        assert appointment["status"] == "broken"
        assert busy.state.current_location != "park"
        assert "失信" in engine.relationship_store.rapport(waiter.id, busy.id).tags

    run(scenario())


def test_a_meeting_that_would_run_into_a_shift_is_refused(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        speaker = next(agent for agent in engine.agents if agent.id == "wang")
        partner = next(agent for agent in engine.agents if agent.id == "mei")
        free_all_day = ScheduleItem(
            hour=6, minute=0, label="自由时间", location="park", kind="flexible",
            activity="自由时间", expected_duration=600, flexibility=1.0,
            responsibility=0.1, affected_people=(), consequence_of_delay="",
            is_flexible_slot=True,
        )
        speaker.schedule = [
            free_all_day,
            ScheduleItem(
                hour=12, minute=15, label="午班", location="cafe", kind="work",
                activity="看店", expected_duration=120, flexibility=0.0,
                responsibility=0.8, affected_people=(), consequence_of_delay="",
                is_flexible_slot=False,
            ),
        ]
        partner.schedule = [free_all_day]

        # 11:00 finishes before the shift starts, so it is a real arrangement.
        accepted, _ = await engine.create_appointment(
            speaker, [partner], proposal(hour=11, minute=0),
        )
        assert accepted
        engine._appointments.clear()
        engine.tasks.clear()

        # 12:00 would run into the 12:15 shift.
        refused, note = await engine.create_appointment(
            speaker, [partner], proposal(hour=12, minute=0),
        )
        assert not refused and "固定安排" in note

    run(scenario())


def test_standing_someone_up_costs_trust(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        engine._realtime_llm = True
        waiter = next(agent for agent in engine.agents if agent.id == "wang")
        no_show = next(agent for agent in engine.agents if agent.id == "mei")
        engine.agents = [waiter, no_show]
        for agent in (waiter, no_show):
            agent.schedule = []
            agent.routine_blocks = []
        waiter.state.current_location = "park"
        waiter.state.x, waiter.state.y = 9, 8
        waiter.state.status = "IDLE"
        # The other one is stuck in a long activity, so it never turns up.
        no_show.state.current_location = "home_mei"
        no_show.state.x, no_show.state.y = 5, 1
        no_show.state.status = "ACTING"
        no_show._activity_ticks = 400
        no_show._activity_desc = "在家午睡"
        no_show._activity_commitment_id = "duty:home_mei"

        accepted, _ = await engine.create_appointment(
            waiter, [no_show],
            proposal(hour=engine.hour, minute=engine.minute + 20),
        )
        assert accepted
        appointment = next(iter(engine._appointments.values()))

        events = []
        for _ in range(12):
            await engine.tick()
            await settle(engine)
            events.extend(engine._last_events[-8:])
            if appointment["status"] != "pending":
                break

        assert appointment["status"] == "broken"
        result = next(event for event in events if event["type"] == "appointment_result")
        assert "没有来" in result["content"]
        # Trust floors at zero, so a first betrayal shows as affinity and a tag.
        after = engine.relationship_store.rapport(waiter.id, no_show.id)
        assert after.affinity < 0 and "失信" in after.tags
        assert after.interaction_count == 1
        memories = await engine.memory.get_all_for_agent(waiter.id)
        assert any(memory.event_type == "appointment_broken" for memory in memories)
        own = await engine.memory.get_all_for_agent(no_show.id)
        assert any(memory.event_type == "appointment_missed" for memory in own)

    run(scenario())
