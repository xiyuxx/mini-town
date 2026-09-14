"""Persistence for behavior-relevant cognition, separate from runtime state."""

import json
import os
import aiosqlite
from dataclasses import asdict

from .cognition import (
    ActionOutcome, Belief, EmotionalState, EpisodeRecord, Goal, Intention,
    InteractionContext, LifePlan, PlanStep,
)
from .habits import Habit


class CognitionStore:
    """Store durable cognition snapshots without persisting transient movement."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    async def init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS cognition_snapshots (
                    agent_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL
                )
            """)
            await db.commit()

    async def save_agent(self, agent, habits) -> None:
        mental = agent.mental_state
        payload = {
            "goals": [asdict(item) for item in mental.goals],
            "intentions": [asdict(item) for item in mental.intentions],
            "beliefs": [asdict(item) for item in mental.beliefs],
            "attention": list(mental.attention),
            "emotion": asdict(mental.emotion),
            "life_plan": asdict(mental.life_plan) if mental.life_plan else None,
            "current_episode": asdict(mental.current_episode) if mental.current_episode else None,
            "recent_interactions": {
                key: asdict(value) for key, value in mental.recent_interactions.items()
            },
            "recent_outcomes": [asdict(item) for item in mental.outcomes],
            "last_decision": dict(mental.last_decision),
            "habits": [asdict(item) for item in habits.for_agent(agent.id)],
        }
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO cognition_snapshots(agent_id, state_json) VALUES (?, ?)",
                (agent.id, json.dumps(payload, ensure_ascii=False)),
            )
            await db.commit()

    async def load_agent(self, agent, habits) -> bool:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT state_json FROM cognition_snapshots WHERE agent_id = ?",
                (agent.id,),
            )
            row = await cursor.fetchone()
        if not row:
            return False
        try:
            payload = json.loads(row[0])
            mental = agent.mental_state
            mental.goals = [Goal(**item) for item in payload.get("goals", [])]
            mental.intentions = [Intention(**item) for item in payload.get("intentions", [])]
            mental.beliefs = [Belief(**item) for item in payload.get("beliefs", [])]
            mental.attention = [str(item) for item in payload.get("attention", [])]
            emotion = payload.get("emotion")
            if isinstance(emotion, dict):
                mental.emotion = EmotionalState(**emotion)
            life_plan = payload.get("life_plan")
            if isinstance(life_plan, dict):
                steps = [PlanStep(**step) for step in life_plan.pop("steps", [])]
                life_plan["steps"] = steps
                mental.life_plan = LifePlan(**life_plan)
            episode = payload.get("current_episode")
            if isinstance(episode, dict):
                mental.current_episode = EpisodeRecord(**episode)
            mental.recent_interactions = {
                str(key): InteractionContext(**value)
                for key, value in payload.get("recent_interactions", {}).items()
                if isinstance(value, dict)
            }
            mental.outcomes = [
                ActionOutcome(**item) for item in payload.get("recent_outcomes", [])
                if isinstance(item, dict)
            ][-20:]
            mental.last_decision = dict(payload.get("last_decision", {}))
            for item in payload.get("habits", []):
                habits.restore(Habit(**item))
            return True
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            return False

    async def clear(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM cognition_snapshots")
            await db.commit()
