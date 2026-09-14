"""Structured trace logging for debugging agent behavior."""

import json
import aiosqlite
import os
from datetime import datetime
from ..config import config


class Trace:
    def __init__(self):
        self.db_path = config.DB_PATH

    async def init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time TEXT NOT NULL,
                    sim_time TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    layer TEXT NOT NULL,
                    event TEXT NOT NULL,
                    detail TEXT DEFAULT ''
                )
            """)
            await db.commit()

    async def log(self, sim_time: str, agent_id: str, layer: str, event: str, detail: str = ""):
        """Log a trace event. Layers: state, perceive, tool, llm, action, dialogue."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT INTO traces (time, sim_time, agent_id, layer, event, detail) VALUES (?, ?, ?, ?, ?, ?)",
                    (datetime.now().isoformat(), sim_time, agent_id, layer, event, detail),
                )
                await db.commit()
        except Exception:
            pass  # Trace failure must never break simulation

    async def get_recent(self, limit: int = 200) -> list[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM traces ORDER BY id DESC LIMIT ?", (limit,)
            )
            rows = await cursor.fetchall()
        return [
            {"id": r[0], "time": r[1], "simTime": r[2], "agentId": r[3],
             "layer": r[4], "event": r[5], "detail": r[6]}
            for r in reversed(rows)
        ]

    async def clear(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM traces")
            await db.commit()
