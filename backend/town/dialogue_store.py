"""Persistent dialogue sessions and messages."""

import json
import uuid
import aiosqlite


class DialogueStore:
    def __init__(self, db_path: str):
        self.db_path = db_path

    async def init_db(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS dialogue_sessions (
                    id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    started_timestamp INTEGER,
                    ended_at TEXT DEFAULT '',
                    ended_timestamp INTEGER,
                    location TEXT NOT NULL,
                    participants TEXT NOT NULL,
                    summary TEXT DEFAULT '',
                    valence REAL DEFAULT 0,
                    status TEXT DEFAULT 'active'
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS dialogue_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dialogue_id TEXT NOT NULL,
                    turn INTEGER NOT NULL,
                    sim_time TEXT NOT NULL,
                    speaker_id TEXT NOT NULL,
                    speaker_name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    action TEXT DEFAULT 'continue',
                    FOREIGN KEY(dialogue_id) REFERENCES dialogue_sessions(id)
                )
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_dialogue_time ON dialogue_sessions(started_timestamp)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_dialogue_message ON dialogue_messages(dialogue_id, turn)")
            await db.commit()

    async def create(self, sim_time: str, sim_timestamp: int, location: str,
                     participant_ids: list[str]) -> str:
        dialogue_id = f"dlg_{uuid.uuid4().hex[:12]}"
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO dialogue_sessions "
                "(id, started_at, started_timestamp, location, participants) VALUES (?, ?, ?, ?, ?)",
                (dialogue_id, sim_time, sim_timestamp, location,
                 json.dumps(participant_ids, ensure_ascii=False)),
            )
            await db.commit()
        return dialogue_id

    async def add_message(self, dialogue_id: str, turn: int, sim_time: str,
                          speaker_id: str, speaker_name: str, content: str,
                          action: str = "continue"):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO dialogue_messages "
                "(dialogue_id, turn, sim_time, speaker_id, speaker_name, content, action) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (dialogue_id, turn, sim_time, speaker_id, speaker_name, content, action),
            )
            await db.commit()

    async def finish(self, dialogue_id: str, sim_time: str, sim_timestamp: int,
                     summary: str, valence: float):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE dialogue_sessions SET ended_at = ?, ended_timestamp = ?, "
                "summary = ?, valence = ?, status = 'completed' WHERE id = ?",
                (sim_time, sim_timestamp, summary, valence, dialogue_id),
            )
            await db.commit()

    async def list_recent(self, limit: int = 30, agent_id: str | None = None) -> list[dict]:
        where = ""
        params: list = []
        if agent_id:
            where = "WHERE participants LIKE ?"
            params.append(f'%"{agent_id}"%')
        params.append(limit)
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                f"SELECT id, started_at, ended_at, location, participants, summary, valence, status "
                f"FROM dialogue_sessions {where} ORDER BY started_timestamp DESC LIMIT ?", params,
            )
            rows = await cursor.fetchall()
        return [self._session_dict(row) for row in rows]

    async def get_recent_between(self, agent_a: str, agent_b: str,
                                 before_timestamp: int | None = None) -> dict | None:
        params: list = [f'%"{agent_a}"%', f'%"{agent_b}"%']
        before = ""
        if before_timestamp is not None:
            before = "AND started_timestamp < ?"
            params.append(before_timestamp)
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT id FROM dialogue_sessions WHERE participants LIKE ? "
                f"AND participants LIKE ? AND status = 'completed' {before} "
                "ORDER BY ended_timestamp DESC LIMIT 1", params,
            )
            row = await cursor.fetchone()
        return await self.get(row[0]) if row else None

    async def get(self, dialogue_id: str) -> dict | None:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT id, started_at, ended_at, location, participants, summary, valence, status "
                "FROM dialogue_sessions WHERE id = ?", (dialogue_id,),
            )
            row = await cursor.fetchone()
            if not row:
                return None
            cursor = await db.execute(
                "SELECT turn, sim_time, speaker_id, speaker_name, content, action "
                "FROM dialogue_messages WHERE dialogue_id = ? ORDER BY turn", (dialogue_id,),
            )
            messages = await cursor.fetchall()
        result = self._session_dict(row)
        result["messages"] = [
            {"turn": item[0], "simTime": item[1], "speakerId": item[2],
             "speakerName": item[3], "content": item[4], "action": item[5]}
            for item in messages
        ]
        return result

    async def clear(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM dialogue_messages")
            await db.execute("DELETE FROM dialogue_sessions")
            await db.commit()

    @staticmethod
    def _session_dict(row: tuple) -> dict:
        return {
            "id": row[0], "startedAt": row[1], "endedAt": row[2],
            "location": row[3], "participants": json.loads(row[4]),
            "summary": row[5], "valence": round(float(row[6] or 0), 2), "status": row[7],
        }
