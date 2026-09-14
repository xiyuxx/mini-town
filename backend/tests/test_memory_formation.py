import asyncio

from backend.town.engine import SimulationEngine
from backend.town.experience import ExperienceEvent


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


def test_autonomous_goal_related_experience_forms_working_memory(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        artist = next(agent for agent in engine.agents if agent.id == "hua")
        memory = await engine.memory_formation.process(artist, ExperienceEvent(
            type="activity_outcome",
            location="home_hua",
            actors=[artist.id],
            source="llm",
            expected=False,
            emotional_intensity=0.2,
            facts={"activity": "给豆豆添猫粮，观察它今天的状态", "reason": "自己决定"},
        ), engine)
        assert memory is not None
        assert memory.tier in ("working", "episodic")
        assert "豆豆" in memory.content

    run(scenario())


def test_routine_commitment_is_not_remembered(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        artist = next(agent for agent in engine.agents if agent.id == "hua")
        memory = await engine.memory_formation.process(artist, ExperienceEvent(
            type="activity_outcome",
            location="home_hua",
            actors=[artist.id],
            source="commitment",
            expected=True,
            facts={"activity": "起床洗漱", "reason": "日常安排"},
        ), engine)
        assert memory is None

    run(scenario())


def test_routine_activity_becomes_part_of_continuous_life_episode(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        student = next(agent for agent in engine.agents if agent.id == "ming")
        activity = ExperienceEvent(
            type="activity_outcome", location="home_ming",
            actors=[student.id], source="commitment", expected=True,
            facts={"activity": "起床准备上学", "reason": "日常安排"},
        )
        assert await engine.memory_formation.process(student, activity, engine) is None
        arrival = ExperienceEvent(
            type="arrival", location="school", actors=[student.id],
            source="commitment", expected=True,
            facts={"purpose": "上课", "lateness": 0, "weather": "阴"},
        )
        memory = await engine.memory_formation.process(student, arrival, engine)
        assert memory is not None
        assert memory.event_type == "life_episode"
        assert "起床准备上学" in memory.content
        assert "镇中学" in memory.content

    run(scenario())


def test_adverse_weather_arrival_is_memorable_and_deduplicated(tmp_path):
    async def scenario():
        engine = await make_engine(tmp_path)
        doctor = next(agent for agent in engine.agents if agent.id == "li")
        event = ExperienceEvent(
            type="arrival",
            location="park",
            actors=[doctor.id],
            source="commitment",
            expected=False,
            emotional_valence=-0.1,
            emotional_intensity=0.2,
            sensory_salience=0.8,
            facts={"purpose": "晨跑", "lateness": 0, "weather": "小雨"},
        )
        first = await engine.memory_formation.process(doctor, event, engine)
        second = await engine.memory_formation.process(doctor, event, engine)
        assert first is not None
        assert second is None
        memories = await engine.memory.get_recent(doctor.id, limit=10)
        assert len(memories) == 1

    run(scenario())
