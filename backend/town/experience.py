"""Selective, subjective memory formation from structured experiences."""

from dataclasses import dataclass, field
import uuid

from .embedding import similarity_gates
from .memory import Memory
from .world import location_name
from .cognition import Intention


@dataclass
class ExperienceEvent:
    type: str
    location: str
    facts: dict
    actors: list[str] = field(default_factory=list)
    source: str = "world"
    expected: bool = True
    emotional_valence: float = 0.0
    emotional_intensity: float = 0.0
    social_relevance: float = 0.0
    goal_relevance: float = 0.0
    unresolvedness: float = 0.0
    sensory_salience: float = 0.0
    id: str = field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:10]}")


@dataclass
class MemoryCandidate:
    event: ExperienceEvent
    score: float
    tier: str
    importance: int
    fact_summary: str
    interpretation_hint: str
    emotion: str
    future_intention: str = ""



class MemoryFormation:
    def __init__(self, memory_store, llm, trace=None):
        self.memory = memory_store
        self.llm = llm
        self.trace = trace
        self._episode_buffers: dict[str, list[ExperienceEvent]] = {}

    async def process(self, agent, event: ExperienceEvent, engine) -> Memory | None:
        if event.type == "activity_outcome":
            self._remember_in_episode(agent.id, event)
        elif event.type == "arrival":
            episode_memory = await self._close_episode(agent, event, engine)
            if episode_memory:
                return episode_memory

        candidate = await self._evaluate(agent, event, engine)
        if candidate is None:
            return None
        if await self._is_redundant(agent.id, candidate, engine.get_sim_timestamp()):
            await self._trace(engine, agent.id, "memory_suppressed", candidate.fact_summary)
            return None
        await self._enrich_high_value_candidate(agent, candidate, engine)

        content = await self._encode(agent, candidate, engine)
        memory = await self.memory.add(
            agent.id,
            event.location,
            content,
            "observation" if event.type != "dialogue" else "dialogue",
            candidate.importance,
            sim_time=engine.get_sim_time_str(),
            participants=[actor for actor in event.actors if actor != agent.id],
            emotion=candidate.emotion,
            event_type=event.type,
            tier=candidate.tier,
            fact_summary=candidate.fact_summary,
            interpretation=candidate.interpretation_hint,
            unresolved=event.unresolvedness >= 0.5,
            future_intention=candidate.future_intention,
            confidence=0.85,
            formation_score=candidate.score,
            source_ids=[str(item) for item in event.facts.get("fact_ids", [])],
        )
        if memory and candidate.future_intention:
            agent.mental_state.add_intention(Intention(
                description=candidate.future_intention, source="memory",
                created_at=engine.get_sim_timestamp(), target_location=event.location,
                confidence=memory.confidence, source_ids=[memory.id],
            ))
        if memory:
            await self._trace(engine, agent.id, "memory_formed", f"{candidate.tier}:{candidate.score:.2f}:{content}")
        return memory

    async def _enrich_high_value_candidate(self, agent, candidate: MemoryCandidate, engine) -> None:
        if self.llm.fallback or candidate.score < 0.65:
            return
        result = await self.llm.reason_about_memory(
            agent.persona_text(), candidate.fact_summary, agent.mental_state.active_goal,
            candidate.emotion, [str(item) for item in candidate.event.facts.get("fact_ids", [])],
        )
        interpretation = str(result.get("interpretation", "")).strip()
        if interpretation:
            candidate.interpretation_hint = interpretation[:240]
        intention = str(result.get("future_intention", "")).strip()
        if intention:
            candidate.future_intention = intention[:180]
            candidate.event.unresolvedness = max(candidate.event.unresolvedness, 0.6)
        emotion = str(result.get("emotion", "")).strip()
        if emotion:
            candidate.emotion = emotion[:20]

    def _remember_in_episode(self, agent_id: str, event: ExperienceEvent):
        buffer = self._episode_buffers.setdefault(agent_id, [])
        buffer.append(event)
        if len(buffer) > 4:
            del buffer[:-4]

    async def _close_episode(self, agent, arrival: ExperienceEvent, engine) -> Memory | None:
        buffered = self._episode_buffers.pop(agent.id, [])
        activities = [str(item.facts.get("activity", "")).strip() for item in buffered]
        activities = [activity for activity in activities if activity]
        if not activities:
            return None
        destination = location_name(arrival.location)
        purpose = str(arrival.facts.get("purpose", "接下来的安排"))
        weather = str(arrival.facts.get("weather", ""))
        lateness = int(arrival.facts.get("lateness", 0) or 0)
        fact_summary = f"先{'，接着'.join(activities[-3:])}，之后来到{destination}准备{purpose}"
        if weather:
            fact_summary += f"；当时天气是{weather}"
        if lateness > 0:
            fact_summary += f"，比计划晚了{lateness}分钟"
        episode_event = ExperienceEvent(
            type="life_episode", location=arrival.location,
            facts={"summary": fact_summary}, actors=[agent.id],
        )
        schedule = agent._get_schedule_item(engine.hour, engine.minute)
        responsibility = schedule.responsibility if schedule else 0.0
        flexibility = schedule.flexibility if schedule else 1.0
        subjective_pressure = max(
            responsibility * (1.0 - flexibility),
            agent.mental_state.emotion.arousal
            if agent.mental_state.emotion.cause_fact_ids else 0.0,
        )
        pressured = lateness > 0 and subjective_pressure >= 0.45
        candidate = MemoryCandidate(
            event=episode_event,
            score=0.38 if lateness <= 0 else 0.48 + subjective_pressure * 0.12,
            tier="episodic" if pressured else "working",
            importance=6 if pressured else 3,
            fact_summary=fact_summary,
            interpretation_hint=(
                "这次偏差可能影响了他人的安排" if pressured
                else "时间有些偏差，但没有明确的紧迫后果"
            ),
            emotion=agent.mental_state.emotion.label if pressured else "",
            future_intention="重新安排时间，尽量减少对他人的影响" if pressured else "",
        )
        if await self._is_redundant(agent.id, candidate, engine.get_sim_timestamp()):
            await self._trace(engine, agent.id, "episode_suppressed", fact_summary)
            return None
        content = await self._encode(agent, candidate, engine)
        memory = await self.memory.add(
            agent.id, arrival.location, content, "observation", candidate.importance,
            sim_time=engine.get_sim_time_str(), emotion=candidate.emotion,
            event_type="life_episode", tier=candidate.tier,
            fact_summary=fact_summary, interpretation=candidate.interpretation_hint,
            future_intention=candidate.future_intention,
            unresolved=bool(candidate.future_intention), formation_score=candidate.score,
        )
        if memory:
            await self._trace(engine, agent.id, "episode_formed", content)
        return memory

    async def _evaluate(self, agent, event: ExperienceEvent, engine) -> MemoryCandidate | None:
        facts = event.facts
        effects = [item for item in facts.get("effects", []) if isinstance(item, dict)]
        active_goal = agent.mental_state.active_goal_object
        goal_ids = {
            str(item.get("goal_id", "")) for item in effects
            if item.get("type") == "goal_progress"
        }
        goal_impact = event.goal_relevance
        if active_goal and (active_goal.id in goal_ids or facts.get("goal_id") == active_goal.id):
            goal_impact = max(goal_impact, 0.8)
        if any(item.get("type") == "goal_progress" for item in effects):
            goal_impact = max(goal_impact, 0.65)

        belief_updates = facts.get("belief_updates", [])
        belief_impact = min(1.0, 0.35 * len(belief_updates)) if isinstance(belief_updates, list) else 0.0
        action_id = str(facts.get("action_id", ""))
        prediction_error = 0.0
        for outcome in reversed(agent.mental_state.outcomes):
            if action_id and outcome.action_id == action_id:
                prediction_error = outcome.prediction_error
                break
        if event.type == "action_blocked":
            prediction_error = max(prediction_error, 1.0)

        relationship_impact = max(
            event.social_relevance,
            0.55 if len([actor for actor in event.actors if actor != agent.id]) > 0 else 0.0,
        )
        unexpectedness = 0.8 if not event.expected else 0.05
        emotional = max(event.emotional_intensity, abs(event.emotional_valence))
        unresolved = max(event.unresolvedness, 0.65 if facts.get("blocking_fact_ids") else 0.0)
        state_change = 0.55 if effects else 0.0
        autonomous_choice = 1.0 if event.source == "llm" and not event.expected else 0.0

        score = min(1.0, (
            goal_impact * 0.20
            + belief_impact * 0.16
            + prediction_error * 0.18
            + relationship_impact * 0.14
            + unresolved * 0.10
            + unexpectedness * 0.18
            + emotional * 0.10
            + event.sensory_salience * 0.12
            + autonomous_choice * 0.12
            + state_change * 0.02
        ))
        if event.type == "dialogue":
            score = max(score, 0.48)
        if score < 0.25:
            return None
        tier = "working" if score < 0.5 else "episodic" if score < 0.75 else "core"
        if event.type == "dialogue" and tier == "core":
            tier = "episodic"
        importance = 3 if tier == "working" else 6 if tier == "episodic" else 9
        summary, hint, emotion, intention = self._summarize(agent, event, engine)
        return MemoryCandidate(event, score, tier, importance, summary, hint, emotion, intention)

    def _summarize(self, agent, event: ExperienceEvent, engine) -> tuple[str, str, str, str]:
        facts = event.facts
        place = location_name(event.location)
        if event.type == "schedule_deviation":
            planned = facts.get("planned", "原来的安排")
            actual = facts.get("actual", "临时改变了计划")
            reason = facts.get("reason", "情况有变")
            return (
                f"原本打算{planned}，后来{actual}，因为{reason}",
                "计划没有照原样进行，需要决定是否补上",
                "在意",
                f"找机会补上{planned}",
            )
        if event.type == "arrival":
            purpose = facts.get("purpose", "办事")
            lateness = int(facts.get("lateness", 0))
            weather = facts.get("weather", "")
            if lateness > 0:
                item = agent._get_schedule_item(engine.hour, engine.minute)
                responsibility = item.responsibility if item else 0.0
                flexibility = item.flexibility if item else 1.0
                pressure = responsibility * (1.0 - flexibility)
                if pressure >= 0.45:
                    return (
                        f"为了{purpose}来到{place}，比计划晚了{lateness}分钟",
                        "这次偏差可能让依赖这项安排的人等待",
                        agent.mental_state.emotion.label if agent.mental_state.emotion.arousal >= 0.4 else "在意",
                        "重新安排时间，尽量减少对他人的影响",
                    )
                return (
                    f"为了{purpose}来到{place}，比计划晚了{lateness}分钟",
                    "时间有些偏差，但没有明确的紧迫后果",
                    str(facts.get("emotion_label", "")),
                    "",
                )
            return (f"在{weather}中来到{place}准备{purpose}", "这段路程和平时有些不同", str(facts.get("emotion_label", "")), "")
        if event.type == "activity_outcome":
            activity = facts.get("activity", "做事")
            reason = facts.get("reason", "")
            outcome = facts.get("outcome", "")
            return (
                f"在{place}{activity}" + (f"，{outcome}" if outcome else ""),
                f"这件事和{reason}有关" if reason else "这段经历值得留意",
                str(facts.get("emotion_label", "")) or ("满足" if event.emotional_valence > 0.18 else "疲惫" if event.emotional_valence < -0.18 else ""),
                "",
            )
        summary = str(facts.get("summary", facts))
        return (summary, "这件事对我有些影响", str(facts.get("emotion_label", "")), str(facts.get("future_intention", "")))

    async def _is_redundant(self, agent_id: str, candidate: MemoryCandidate,
                            current_timestamp: int) -> bool:
        """Whether an equivalent memory of the same kind already exists.

        Read-only and fail-open: a lookup failure reports "not redundant" so a
        provider outage cannot silently discard the experience.
        """
        try:
            scored = await self.memory.semantic_search_scored(
                agent_id, candidate.fact_summary, limit=3,
                current_sim_timestamp=current_timestamp,
            )
        except Exception:
            return False
        gate = similarity_gates(self.memory.embedding_provider)["redundancy"]
        return any(
            similarity > gate and memory.event_type == candidate.event.type
            for memory, similarity in scored
        )

    async def _encode(self, agent, candidate: MemoryCandidate, engine) -> str:
        fallback = self._fallback_content(candidate)
        if self.llm.fallback:
            return fallback
        sim_time = engine.get_sim_time_str()
        period = self._time_period(engine.hour)
        try:
            content = await self.llm.encode_experience_memory(
                agent.persona_text(), candidate.fact_summary,
                candidate.interpretation_hint, candidate.emotion,
                candidate.future_intention, sim_time=sim_time, time_period=period,
            )
            return content if self._time_consistent(
                content, candidate.fact_summary, engine.hour
            ) else fallback
        except Exception:
            return fallback

    @staticmethod
    def _time_period(hour: int) -> str:
        if hour < 6:
            return "凌晨"
        if hour < 12:
            return "上午"
        if hour < 13:
            return "中午"
        if hour < 18:
            return "下午"
        return "晚上"

    @staticmethod
    def _time_consistent(content: str, facts: str, hour: int) -> bool:
        allowed = MemoryFormation._time_period(hour)
        period_words = ("凌晨", "清晨", "早晨", "上午", "中午", "午后", "下午", "傍晚", "晚上", "夜里")
        aliases = {
            "凌晨": {"凌晨", "夜里"},
            "上午": {"清晨", "早晨", "上午"},
            "中午": {"中午"},
            "下午": {"午后", "下午", "傍晚"},
            "晚上": {"晚上", "夜里"},
        }
        for word in period_words:
            if word in content and word not in facts and word not in aliases.get(allowed, set()):
                return False
        for relation in ("开店前", "开店后", "关店前", "关店后", "下班后"):
            if relation in content and relation not in facts:
                return False
        return True

    @staticmethod
    def _fallback_content(candidate: MemoryCandidate) -> str:
        content = candidate.fact_summary
        if candidate.interpretation_hint:
            content += f"。{candidate.interpretation_hint}"
        if candidate.future_intention:
            content += f"，{candidate.future_intention}。"
        return content

    async def _trace(self, engine, agent_id: str, event: str, detail: str):
        if self.trace:
            await self.trace.log(engine.get_sim_time_str(), agent_id, "memory", event, detail[:300])
