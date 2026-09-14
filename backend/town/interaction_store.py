"""Durable lifecycle records for world and agent interactions."""

import json
import os
import aiosqlite
from dataclasses import asdict, dataclass, field


@dataclass
class InteractionRecord:
    id: str
    sim_timestamp: int
    state_version: int
    initiator_id: str
    target_type: str = ""
    target_id: str = ""
    interaction_type: str = ""
    intention_id: str = ""
    status: str = "proposed"
    payload: dict = field(default_factory=dict)
    expected_effects: list[dict] = field(default_factory=list)
    actual_effects: list[dict] = field(default_factory=list)
    fact_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class InteractionStore:
    def __init__(self, db_path: str):
        self.db_path = db_path

    async def init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id TEXT PRIMARY KEY,
                    sim_timestamp INTEGER NOT NULL,
                    state_version INTEGER NOT NULL,
                    initiator_id TEXT NOT NULL,
                    target_type TEXT DEFAULT '',
                    target_id TEXT DEFAULT '',
                    interaction_type TEXT DEFAULT '',
                    intention_id TEXT DEFAULT '',
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    expected_effects TEXT NOT NULL,
                    actual_effects TEXT NOT NULL,
                    fact_ids TEXT NOT NULL
                )
            """)
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_interaction_time ON interactions(sim_timestamp)"
            )
            await db.commit()

    async def upsert(self, record: InteractionRecord):
        data = record.to_dict()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO interactions
                   (id, sim_timestamp, state_version, initiator_id, target_type,
                    target_id, interaction_type, intention_id, status, payload,
                    expected_effects, actual_effects, fact_ids)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.id, record.sim_timestamp, record.state_version,
                    record.initiator_id, record.target_type, record.target_id,
                    record.interaction_type, record.intention_id, record.status,
                    json.dumps(data["payload"], ensure_ascii=False),
                    json.dumps(data["expected_effects"], ensure_ascii=False),
                    json.dumps(data["actual_effects"], ensure_ascii=False),
                    json.dumps(data["fact_ids"], ensure_ascii=False),
                ),
            )
            await db.commit()

    async def get(self, interaction_id: str) -> InteractionRecord | None:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM interactions WHERE id = ?", (interaction_id,)
            )
            row = await cursor.fetchone()
        return self._from_row(row) if row else None

    async def list_recent(self, limit: int = 100) -> list[InteractionRecord]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM interactions ORDER BY sim_timestamp DESC, rowid DESC LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
        return [self._from_row(row) for row in rows]

    async def clear(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM interactions")
            await db.commit()

    @staticmethod
    def _from_row(row: tuple) -> InteractionRecord:
        return InteractionRecord(
            id=row[0], sim_timestamp=row[1], state_version=row[2], initiator_id=row[3],
            target_type=row[4], target_id=row[5], interaction_type=row[6],
            intention_id=row[7], status=row[8], payload=json.loads(row[9]),
            expected_effects=json.loads(row[10]), actual_effects=json.loads(row[11]),
            fact_ids=json.loads(row[12]),
        )
