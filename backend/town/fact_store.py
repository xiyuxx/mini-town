"""Durable fact ledger: what happened, and who has heard about it."""

import json
import os

import aiosqlite

from ..config import config
from .cognition import FactEvent


class FactStore:
    """Backing table for FactLedger so knowledge survives a restart.

    Facts are what a conversation can carry between agents; persisting them (and
    the growable known_by list) is what makes spreading news observable rather
    than something that evaporates when the process restarts.
    """

    def __init__(self):
        self.db_path = config.DB_PATH

    async def init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id TEXT PRIMARY KEY,
                    sim_time TEXT DEFAULT '',
                    sim_timestamp INTEGER DEFAULT 0,
                    type TEXT DEFAULT '',
                    agent_id TEXT DEFAULT '',
                    location TEXT DEFAULT '',
                    details TEXT DEFAULT '{}',
                    participants TEXT DEFAULT '[]',
                    source_ids TEXT DEFAULT '[]',
                    known_by TEXT DEFAULT '[]',
                    interaction_id TEXT DEFAULT ''
                )
            """)
            await db.commit()

    async def upsert(self, fact: FactEvent):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO facts
                   (id, sim_time, sim_timestamp, type, agent_id, location, details,
                    participants, source_ids, known_by, interaction_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (fact.id, fact.sim_time, int(fact.sim_timestamp), fact.type, fact.agent_id,
                 fact.location, json.dumps(fact.details, ensure_ascii=False),
                 json.dumps(fact.participants, ensure_ascii=False),
                 json.dumps(fact.source_ids, ensure_ascii=False),
                 json.dumps(fact.known_by, ensure_ascii=False), fact.interaction_id),
            )
            await db.commit()

    async def mark_known(self, fact: FactEvent):
        """Persist only the knowledge list of an existing fact."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE facts SET known_by = ? WHERE id = ?",
                (json.dumps(fact.known_by, ensure_ascii=False), fact.id),
            )
            await db.commit()

    async def load_recent(self, limit: int = 500) -> list[FactEvent]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT id, sim_time, sim_timestamp, type, agent_id, location, details, "
                "participants, source_ids, known_by, interaction_id FROM facts "
                "ORDER BY sim_timestamp DESC, rowid DESC LIMIT ?", (limit,),
            )
            rows = await cursor.fetchall()
        return [
            FactEvent(
                id=row[0], sim_time=row[1], sim_timestamp=row[2], type=row[3],
                agent_id=row[4], location=row[5],
                details=json.loads(row[6] or "{}"),
                participants=json.loads(row[7] or "[]"),
                source_ids=json.loads(row[8] or "[]"),
                known_by=json.loads(row[9] or "[]"),
                interaction_id=row[10],
            )
            for row in rows
        ]

    async def clear(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM facts")
            await db.commit()
