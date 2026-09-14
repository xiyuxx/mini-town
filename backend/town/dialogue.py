"""Persistent cross-tick dialogue state machine."""

import random
from dataclasses import dataclass, field

from .memory import memories_to_text
from .sim_time import sim_timestamp
from .cognition import InteractionContext
from .world import LOCATIONS
from .world_pack import DEFAULT_WORLD

FALLBACK_TALK_REPLIES = [str(item) for item in DEFAULT_WORLD.data.get("fallback_dialogue", ["嗯。"])]

DIALOGUE_SYSTEM_PROMPT = """你是当前世界中的角色，正在和{others}对话。
当前模拟时间是{sim_time}。说话、行动和记忆共享同一份持续心智状态。
可以自然换话题或再次和刚聊过的人开口，但必须知道近期会话中已经确认的事实。
对方还不知道的消息可以主动讲出来，讲的时候要引用它的事实ID。
只把“当前可观察事实、当前对话、近期会话”中的内容当作确定事实。
背景资料仅决定说话风格；推测必须用“好像、看起来、也许”等不确定表达。
不补写不存在的动作、过去、年份、年龄、病情或共同经历。
调用 dialogue_reply：content 为1-3句口语；action 为 continue 或 end；同时提交mental_update和referenced_fact_ids。
如果聊到要一起做某件事，可以填 appointment：约好的人名、0或1（0=今天，1=明天）、小时、分钟、地点、做什么。地点只能从"可约的地点"里选，时间要留出准备时间。不确定就不要填。"""

DIALOGUE_REPLY_TOOL_DEF = {
    "function": {
        "description": "提交这一轮对话回复",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "action": {"type": "string", "enum": ["continue", "end"]},
                "emoji": {"type": "string"},
                "mental_update": {"type": "object", "additionalProperties": True},
                "referenced_fact_ids": {"type": "array", "items": {"type": "string"}},
                "appointment": {
                    "type": "object",
                    "description": "聊到要一起做什么时填；其余情况留空",
                    "properties": {
                        "with_name": {"type": "string"},
                        "day_offset": {"type": "integer", "enum": [0, 1]},
                        "hour": {"type": "integer"},
                        "minute": {"type": "integer"},
                        "location": {"type": "string"},
                        "activity": {"type": "string"},
                    },
                    "required": ["with_name", "day_offset", "hour", "minute", "location", "activity"],
                },
            },
            "required": ["content", "action", "emoji", "mental_update", "referenced_fact_ids"],
        },
    }
}


@dataclass
class DialogueSession:
    id: str
    participants: list
    original_participants: list
    location_id: str
    conversation: list[str] = field(default_factory=list)
    current_speaker_index: int = 0
    turn_count: int = 1
    max_turns: int = 12


class DialogueManager:
    def __init__(self, llm, memory, relationship_store, tool_registry_factory, store):
        self.llm = llm
        self.memory = memory
        self.relationship_store = relationship_store
        self.tool_registry_factory = tool_registry_factory
        self.store = store
        self.fact_ledger = None
        self.fact_store = None
        self.appointment_handler = None
        self._active_dialogues: dict[str, DialogueSession] = {}

    def participant_ids(self) -> set[str]:
        return {
            participant.id
            for session in self._active_dialogues.values()
            for participant in session.participants
        }

    async def start_dialogue(self, participants: list, opening_line: str,
                             location_id: str, trace, sim_time: str) -> list[dict]:
        existing = self._active_dialogues.get(location_id)
        if existing:
            return []
        initiator = participants[0]
        dialogue_id = await self.store.create(
            sim_time, sim_timestamp(sim_time) or 0, location_id,
            [participant.id for participant in participants],
        )
        session = DialogueSession(
            id=dialogue_id,
            participants=list(participants),
            original_participants=list(participants),
            location_id=location_id,
        )
        line = f"{initiator.name}: {opening_line}"
        session.conversation.append(line)
        session.current_speaker_index = 1 % len(participants)
        self._active_dialogues[location_id] = session
        await self.store.add_message(dialogue_id, 1, sim_time, initiator.id,
                                     initiator.name, opening_line)
        await trace.log(sim_time, initiator.id, "dialogue", "dialogue_start",
                        f"dialogue_id={dialogue_id}, participants=" +
                        "、".join(p.name for p in participants))
        ids = [p.id for p in participants]
        return [
            {"type": "dialogue_start", "dialogueId": dialogue_id,
             "interactionId": dialogue_id,
             "agentIds": ids, "location": location_id,
             "content": "群聊开始：" + "、".join(p.name for p in participants)},
            {"type": "dialogue_line", "dialogueId": dialogue_id,
             "agentIds": ids, "location": location_id,
             "content": line, "speaker": initiator.name},
        ]

    async def advance_all(self, trace, sim_time: str) -> list[dict]:
        events: list[dict] = []
        for location_id in list(self._active_dialogues):
            session = self._active_dialogues.get(location_id)
            if session:
                events.extend(await self.advance_one_turn(session, trace, sim_time))
        return events

    async def advance_one_turn(self, session: DialogueSession, trace,
                               sim_time: str) -> list[dict]:
        if len(session.participants) <= 1 or session.turn_count >= session.max_turns:
            return await self._finish(session, trace, sim_time)
        speaker = session.participants[session.current_speaker_index]
        others = [p for p in session.participants if p.id != speaker.id]
        reply, action, mental_update, referenced_fact_ids = await self._generate_reply(
            session, speaker, others, trace, sim_time
        )
        if not reply.strip():
            return await self._finish(session, trace, sim_time)

        now = sim_timestamp(sim_time) or 0
        if self.fact_ledger is not None:
            valid_refs, invalid_refs = self.fact_ledger.validate_references(
                referenced_fact_ids, speaker.id, now
            )
            referenced_fact_ids = valid_refs
            facts = [
                {"id": fact.id, "type": fact.type, "time": fact.sim_time, "details": fact.details}
                for fact in self.fact_ledger.many(valid_refs)
            ]
            support = await self.llm.validate_claim_support(reply, facts)
            if invalid_refs or not support.get("supported", False):
                reply = str(support.get("rewrite") or f"我不太确定，不过{reply}")[:300]
                mental_update = dict(mental_update or {})
                updates = list(mental_update.get("belief_updates", []))
                updates.append({
                    "proposition": reply, "status": "uncertain",
                    "confidence": 0.35,
                    "source_fact_ids": valid_refs,
                })
                mental_update["belief_updates"] = updates
            # Saying something out loud is how the listener comes to know it.
            for fact in self.fact_ledger.share(referenced_fact_ids, [p.id for p in others]):
                await trace.log(
                    sim_time, speaker.id, "dialogue", "fact_shared",
                    f"dialogue_id={session.id}, fact_id={fact.id}, "
                    f"listeners={','.join(item for item in fact.known_by if item != speaker.id)}",
                )
                if self.fact_store is not None:
                    await self.fact_store.mark_known(fact)

        session.turn_count += 1
        speaker.mental_state.apply_update(
            mental_update, sim_timestamp(sim_time) or 0,
            source_ids=[session.id, *referenced_fact_ids],
            valid_fact_ids=set(referenced_fact_ids),
        )
        line = f"{speaker.name}: {reply}"
        session.conversation.append(line)
        await self.store.add_message(session.id, session.turn_count, sim_time,
                                     speaker.id, speaker.name, reply, action)
        await trace.log(sim_time, speaker.id, "dialogue", "dialogue_turn_end",
                        f"dialogue_id={session.id}, action={action}, reply_len={len(reply)}")
        event = {
            "type": "dialogue_line", "dialogueId": session.id,
            "interactionId": session.id,
            "agentIds": [p.id for p in session.original_participants],
            "location": session.location_id, "content": line,
            "speaker": speaker.name,
        }
        if action == "end":
            return [event, *(await self._finish(session, trace, sim_time))]
        session.current_speaker_index = (
            session.current_speaker_index + 1
        ) % len(session.participants)
        if session.turn_count >= session.max_turns:
            return [event, *(await self._finish(session, trace, sim_time))]
        return [event]

    async def _generate_reply(self, session, speaker, others, trace, sim_time):
        if self.llm.fallback:
            return (
                random.choice(FALLBACK_TALK_REPLIES),
                random.choice(["continue", "end"]), {}, [],
            )
        memories = await self.memory.retrieve(
            speaker.id, session.location_id,
            nearby_agents=[p.name for p in others],
            current_sim_timestamp=sim_timestamp(sim_time),
        )
        recent_dialogues = {}
        for other in others:
            recent = await self.store.get_recent_between(
                speaker.id, other.id, before_timestamp=sim_timestamp(sim_time),
            )
            if recent:
                recent_dialogues[other.name] = {
                    "endedAt": recent["endedAt"],
                    "summary": recent["summary"],
                    "lastMessages": [message["content"] for message in recent.get("messages", [])[-4:]],
                }
        meetup_places = [
            {"id": loc.id, "name": loc.name}
            for loc in LOCATIONS if loc.access.get("mode", "public") == "public"
        ]
        now_minutes = (sim_timestamp(sim_time) or 0) % 1440
        my_day = [
            {"time": f"{item.hour:02d}:{item.minute:02d}", "what": item.activity or item.label}
            for item in speaker.schedule
            if item.start_minutes >= now_minutes
        ][:6]
        known_facts = []
        shareable_facts = []
        if self.fact_ledger is not None:
            known_facts = [
                {"id": fact.id, "type": fact.type, "time": fact.sim_time, "details": fact.details}
                for fact in self.fact_ledger.known_for(speaker.id, limit=20)
            ]
            shareable_facts = [
                {"id": fact.id, "type": fact.type, "time": fact.sim_time, "details": fact.details}
                for fact in self.fact_ledger.unknown_to(
                    speaker.id, [p.id for p in others], limit=8,
                )
            ]
        relationships = {}
        for other in others:
            toward_other, from_other = await self.relationship_store.get_bidirectional(
                speaker.id, other.id,
            )
            relationships[other.name] = {
                "towardOther": toward_other.to_dict(),
                "towardSpeaker": from_other.to_dict(),
            }
        observable_facts = {
            other.name: {
                "location": other.state.current_location,
                "currentActivity": other.state.current_action,
                "status": other.state.status,
            } for other in others
        }
        registry = self.tool_registry_factory(speaker.id)
        captured: dict = {}

        async def handler(content: str, action: str, emoji: str,
                          mental_update: dict | None = None,
                          referenced_fact_ids: list[str] | None = None,
                          appointment: dict | None = None) -> str:
            captured.update(
                content=content, action=action, emoji=emoji,
                mental_update=mental_update or {},
                referenced_fact_ids=referenced_fact_ids or [],
            )
            if appointment and self.appointment_handler is not None:
                accepted, note = await self.appointment_handler(speaker, others, appointment)
                await trace.log(
                    sim_time, speaker.id, "appointment",
                    "accepted" if accepted else "rejected", note[:160],
                )
                return f"回复已提交。{note}"
            return "回复已提交"

        registry.register("dialogue_reply",
                          DIALOGUE_REPLY_TOOL_DEF["function"]["description"],
                          DIALOGUE_REPLY_TOOL_DEF["function"]["parameters"], handler)
        await trace.log(sim_time, speaker.id, "dialogue", "dialogue_turn_start",
                        f"dialogue_id={session.id}, speaker={speaker.name}")
        result = await self.llm.function_call_loop(
            DIALOGUE_SYSTEM_PROMPT.format(
                others="、".join(p.name for p in others), sim_time=sim_time,
            ),
            f"{speaker.persona_text()}\n\n"
            f"持续心智状态：\n{speaker.mental_state.summary_for('dialogue', sim_timestamp(sim_time) or 0)}\n\n"
            f"当前可观察事实：\n{observable_facts}\n\n"
            f"与参与者的关系：\n{relationships}\n\n"
            f"可引用事实（确定陈述必须引用这里的ID）：\n{known_facts}\n\n"
            f"对方还不知道、可以主动分享的消息（分享时同样引用其ID）：\n{shareable_facts or '无'}\n\n"
            f"可约的地点（appointment.location 只能从这里选）：\n{meetup_places}\n\n"
            f"我今天的安排（约时间要避开固定安排）：\n{my_day or '没有别的安排'}\n\n"
            f"双方近期会话（不是冷却限制，可以自然承接）：\n{recent_dialogues or '无'}\n\n"
            f"相关长期记忆：\n{memories_to_text(memories)}\n\n"
            f"当前对话记录：\n" + "\n".join(session.conversation) +
            f"\n\n现在轮到{speaker.name}。",
            registry.get_definitions(), registry, max_calls=3,
        )
        return (
            captured.get("content") or result.get("content", ""),
            captured.get("action") or result.get("action", "continue"),
            captured.get("mental_update") or result.get("mental_update", {}),
            captured.get("referenced_fact_ids") or result.get("referenced_fact_ids", []),
        )

    async def _finish(self, session: DialogueSession, trace, sim_time: str) -> list[dict]:
        if session.location_id not in self._active_dialogues:
            return []
        conversation_text = "\n".join(session.conversation)
        names = [p.name for p in session.original_participants]
        evaluation = await self.llm.evaluate_dialogue(conversation_text, names)
        summary = evaluation.get("summary", conversation_text[-240:])
        valence = float(evaluation.get("valence", 0.0))
        affinity_delta = float(evaluation.get("affinity_delta", 0.0))
        trust_delta = float(evaluation.get("trust_delta", 0.0))
        score = min(0.74, 0.45 + abs(valence) * 0.18 + abs(affinity_delta) * 0.12)
        tier = "working" if score < 0.5 else "episodic"

        for person in session.original_participants:
            others = [p for p in session.original_participants if p.id != person.id]
            subjective = await self.llm.encode_dialogue_memory(
                person.persona_text(), conversation_text,
                [p.name for p in others], summary,
            )
            await self.memory.add(
                person.id, session.location_id, subjective, "dialogue",
                3 if tier == "working" else 6, sim_time=sim_time,
                participants=[p.id for p in others],
                emotion="积极" if valence > 0.2 else "消极" if valence < -0.2 else "中性",
                event_type="dialogue", tier=tier, fact_summary=summary,
                interpretation=subjective, formation_score=score,
                source_ids=[session.id],
            )
        for index, person in enumerate(session.original_participants):
            for other in session.original_participants[index + 1:]:
                await self.relationship_store.record_interaction(
                    person.id, other.id, summary, valence=valence,
                    affinity_delta=affinity_delta, trust_delta=trust_delta,
                    sim_time=sim_time, interaction_id=session.id,
                )
        ended_timestamp = sim_timestamp(sim_time) or 0
        await self.store.finish(session.id, sim_time, ended_timestamp, summary, valence)
        last_messages = session.conversation[-4:]
        for person in session.original_participants:
            for other in session.original_participants:
                if person.id == other.id:
                    continue
                person.mental_state.set_recent_interaction(other.id, InteractionContext(
                    agent_id=other.id,
                    dialogue_id=session.id,
                    ended_at=sim_time,
                    ended_timestamp=ended_timestamp,
                    summary=summary,
                    last_messages=last_messages,
                ))
        await trace.log(sim_time, "", "dialogue", "dialogue_end",
                        f"dialogue_id={session.id}, turns={session.turn_count}")
        self._active_dialogues.pop(session.location_id, None)
        return [{
            "type": "dialogue_end", "dialogueId": session.id,
            "interactionId": session.id,
            "agentIds": [p.id for p in session.original_participants],
            "location": session.location_id, "content": "对话结束",
        }]

    async def add_participant(self, new_agent, conversation_context=None, sim_time=""):
        session = self._active_dialogues.get(new_agent.state.current_location)
        if session and new_agent.id not in {p.id for p in session.participants}:
            session.participants.append(new_agent)
            session.original_participants.append(new_agent)

    def get_active_at_location(self, location_id: str):
        return self._active_dialogues.get(location_id)
