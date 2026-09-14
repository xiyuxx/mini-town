"""SQLite-backed episodic memory with simulation-time retrieval."""

import aiosqlite
import json
import math
import os
import uuid
from dataclasses import dataclass
from datetime import datetime

from backend.config import config
from .sim_time import parse_sim_time, sim_timestamp


@dataclass
class Memory:
    id: str
    agent_id: str
    time: str
    location: str
    content: str
    importance: int
    type: str
    embedding: list[float] | None = None
    recall_count: int = 0
    created_at: str = ""
    sim_day: int | None = None
    sim_minute: int | None = None
    sim_timestamp: int | None = None
    participants: list[str] | None = None
    emotion: str = ""
    source_ids: list[str] | None = None
    event_type: str = ""
    tier: str = "episodic"
    fact_summary: str = ""
    interpretation: str = ""
    unresolved: bool = False
    future_intention: str = ""
    confidence: float = 1.0
    formation_score: float = 0.0


class MemoryStore:
    _SKIP_CONTENT = {"待机中", "休息中"}

    def __init__(self):
        self._last_content: dict[str, str] = {}
        self.db_path = config.DB_PATH
        self.embedding_provider = None

    def set_embedding_provider(self, provider) -> None:
        self.embedding_provider = provider

    async def init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    time TEXT NOT NULL,
                    location TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance INTEGER DEFAULT 5,
                    type TEXT NOT NULL DEFAULT 'observation'
                )
            """)
            columns = {
                "embedding": "TEXT",
                "recall_count": "INTEGER DEFAULT 0",
                "created_at": "TEXT DEFAULT ''",
                "sim_day": "INTEGER",
                "sim_minute": "INTEGER",
                "sim_timestamp": "INTEGER",
                "participants": "TEXT DEFAULT '[]'",
                "emotion": "TEXT DEFAULT ''",
                "source_ids": "TEXT DEFAULT '[]'",
                "event_type": "TEXT DEFAULT ''",
                "tier": "TEXT DEFAULT 'episodic'",
                "fact_summary": "TEXT DEFAULT ''",
                "interpretation": "TEXT DEFAULT ''",
                "unresolved": "INTEGER DEFAULT 0",
                "future_intention": "TEXT DEFAULT ''",
                "confidence": "REAL DEFAULT 1.0",
                "formation_score": "REAL DEFAULT 0.0",
            }
            for name, definition in columns.items():
                try:
                    await db.execute(f"ALTER TABLE memories ADD COLUMN {name} {definition}")
                except aiosqlite.OperationalError:
                    pass
            await db.execute("CREATE INDEX IF NOT EXISTS idx_agent_time ON memories(agent_id, sim_timestamp)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_type ON memories(type)")
            await db.commit()
        await self._backfill_time_columns()

    async def _backfill_time_columns(self):
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT id, time, created_at, sim_timestamp FROM memories"
            )
            rows = await cursor.fetchall()
            now = datetime.now().isoformat()
            for memory_id, time_text, created_at, timestamp in rows:
                parsed = parse_sim_time(time_text)
                updates = []
                params = []
                if not created_at:
                    updates.append("created_at = ?")
                    params.append(now)
                if timestamp is None and parsed:
                    updates.extend(["sim_day = ?", "sim_minute = ?", "sim_timestamp = ?"])
                    params.extend([parsed.day, parsed.hour * 60 + parsed.minute, parsed.timestamp])
                if updates:
                    params.append(memory_id)
                    await db.execute(
                        f"UPDATE memories SET {', '.join(updates)} WHERE id = ?", params
                    )
            await db.commit()

    async def add(
        self,
        agent_id: str,
        location: str,
        content: str,
        mem_type: str = "observation",
        importance: int = 5,
        embedding: list[float] | None = None,
        sim_time: str | None = None,
        participants: list[str] | None = None,
        emotion: str = "",
        source_ids: list[str] | None = None,
        event_type: str = "",
        tier: str = "episodic",
        fact_summary: str = "",
        interpretation: str = "",
        unresolved: bool = False,
        future_intention: str = "",
        confidence: float = 1.0,
        formation_score: float = 0.0,
    ) -> Memory | None:
        stripped = content.strip()
        if stripped in self._SKIP_CONTENT or len(stripped) < 4:
            return None
        if self._last_content.get(agent_id) == stripped:
            return None
        self._last_content[agent_id] = stripped

        parsed = parse_sim_time(sim_time)
        created_at = datetime.now().isoformat()
        if embedding is None and self.embedding_provider is not None and (importance >= 6 or event_type):
            try:
                embedding = await self.embedding_provider.embed_one(stripped)
            except Exception:
                embedding = None

        mem = Memory(
            id=f"mem_{uuid.uuid4().hex[:12]}",
            agent_id=agent_id,
            time=sim_time or created_at,
            location=location,
            content=stripped,
            importance=max(1, min(10, importance)),
            type=mem_type,
            embedding=embedding,
            created_at=created_at,
            sim_day=parsed.day if parsed else None,
            sim_minute=parsed.hour * 60 + parsed.minute if parsed else None,
            sim_timestamp=parsed.timestamp if parsed else None,
            participants=participants or [],
            emotion=emotion,
            source_ids=source_ids or [],
            event_type=event_type,
            tier=tier,
            fact_summary=fact_summary or stripped,
            interpretation=interpretation,
            unresolved=unresolved,
            future_intention=future_intention,
            confidence=confidence,
            formation_score=formation_score,
        )
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO memories
                   (id, agent_id, time, location, content, importance, type,
                    embedding, recall_count, created_at, sim_day, sim_minute,
                    sim_timestamp, participants, emotion, source_ids, event_type,
                    tier, fact_summary, interpretation, unresolved, future_intention,
                    confidence, formation_score)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    mem.id, mem.agent_id, mem.time, mem.location, mem.content,
                    mem.importance, mem.type,
                    json.dumps(embedding) if embedding else None,
                    mem.created_at, mem.sim_day, mem.sim_minute, mem.sim_timestamp,
                    json.dumps(mem.participants, ensure_ascii=False), mem.emotion,
                    json.dumps(mem.source_ids, ensure_ascii=False), mem.event_type,
                    mem.tier, mem.fact_summary, mem.interpretation,
                    int(mem.unresolved), mem.future_intention,
                    mem.confidence, mem.formation_score,
                ),
            )
            await db.commit()
        return mem

    async def get_recent(self, agent_id: str, hours: float = 4, limit: int = 20,
                         current_sim_timestamp: int | None = None) -> list[Memory]:
        where = "agent_id = ?"
        params: list = [agent_id]
        if current_sim_timestamp is not None:
            where += " AND sim_timestamp >= ? AND sim_timestamp <= ?"
            params.extend([current_sim_timestamp - int(hours * 60), current_sim_timestamp])
        params.append(limit)
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                f"SELECT * FROM memories WHERE {where} "
                "ORDER BY COALESCE(sim_timestamp, rowid) DESC, rowid DESC LIMIT ?",
                params,
            )
            rows = await cursor.fetchall()
        return [_row_to_memory(r) for r in rows]

    async def get_important(self, agent_id: str, hours: float = 24,
                            min_importance: int = 7, limit: int = 10,
                            current_sim_timestamp: int | None = None) -> list[Memory]:
        where = "agent_id = ? AND importance >= ?"
        params: list = [agent_id, min_importance]
        if current_sim_timestamp is not None:
            where += " AND sim_timestamp >= ? AND sim_timestamp <= ?"
            params.extend([current_sim_timestamp - int(hours * 60), current_sim_timestamp])
        params.append(limit)
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                f"SELECT * FROM memories WHERE {where} "
                "ORDER BY importance DESC, COALESCE(sim_timestamp, rowid) DESC LIMIT ?",
                params,
            )
            rows = await cursor.fetchall()
        return [_row_to_memory(r) for r in rows]

    async def get_by_location(self, agent_id: str, location: str,
                              hours: float = 24, limit: int = 10,
                              current_sim_timestamp: int | None = None) -> list[Memory]:
        where = "agent_id = ? AND location = ?"
        params: list = [agent_id, location]
        if current_sim_timestamp is not None:
            where += " AND sim_timestamp >= ? AND sim_timestamp <= ?"
            params.extend([current_sim_timestamp - int(hours * 60), current_sim_timestamp])
        params.append(limit)
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                f"SELECT * FROM memories WHERE {where} "
                "ORDER BY COALESCE(sim_timestamp, rowid) DESC LIMIT ?", params
            )
            rows = await cursor.fetchall()
        return [_row_to_memory(r) for r in rows]

    async def retrieve(self, agent_id: str, location: str,
                       nearby_agents: list[str] | None = None,
                       current_sim_timestamp: int | None = None,
                       query: str = "", limit: int | None = None) -> list[Memory]:
        recent = await self.get_recent(
            agent_id, config.MEMORY_RECENT_HOURS, 15, current_sim_timestamp
        )
        important = await self.get_important(
            agent_id, 72, 7, 10, current_sim_timestamp
        )
        loc = await self.get_by_location(
            agent_id, location, 72, 10, current_sim_timestamp
        )
        semantic = []
        if query.strip() and self.embedding_provider is not None:
            try:
                semantic = await self.semantic_search(
                    agent_id, query, self.embedding_provider, limit=8,
                    current_sim_timestamp=current_sim_timestamp,
                )
            except Exception:
                semantic = []
        seen: set[str] = set()
        result: list[Memory] = []
        for memory_list in (semantic, recent, important, loc):
            for memory in memory_list:
                if memory.id not in seen:
                    seen.add(memory.id)
                    result.append(memory)
        return result[:limit or config.MEMORY_MAX_RETRIEVE]

    async def get_open_loops(self, agent_id: str, limit: int = 10) -> list[Memory]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM memories WHERE agent_id = ? AND unresolved = 1 "
                "ORDER BY COALESCE(sim_timestamp, rowid) DESC LIMIT ?",
                (agent_id, limit),
            )
            rows = await cursor.fetchall()
        return [_row_to_memory(row) for row in rows]

    async def resolve_open_loop(self, memory_id: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("UPDATE memories SET unresolved = 0 WHERE id = ?", (memory_id,))
            await db.commit()

    async def get_all_for_agent(self, agent_id: str, limit: int = 100) -> list[Memory]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM memories WHERE agent_id = ? "
                "ORDER BY COALESCE(sim_timestamp, rowid) DESC LIMIT ?",
                (agent_id, limit),
            )
            rows = await cursor.fetchall()
        return [_row_to_memory(r) for r in rows]

    async def get_shared_between(self, agent_a: str, agent_b: str,
                                 limit: int = 20) -> list[Memory]:
        """Return memories that explicitly record the other participant."""
        memories = await self.get_all_for_agent(agent_a, limit=max(limit * 4, 40))
        shared = [memory for memory in memories if agent_b in (memory.participants or [])]
        shared.sort(
            key=lambda memory: (
                memory.importance,
                1 if memory.unresolved else 0,
                memory.sim_timestamp or 0,
            ),
            reverse=True,
        )
        return shared[:limit]

    async def get_in_sim_range(self, agent_id: str, start: int, end: int,
                               exclude_reflections: bool = False) -> list[Memory]:
        extra = " AND type != 'reflection'" if exclude_reflections else ""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM memories WHERE agent_id = ? "
                "AND sim_timestamp >= ? AND sim_timestamp <= ?" + extra +
                " ORDER BY sim_timestamp ASC",
                (agent_id, start, end),
            )
            rows = await cursor.fetchall()
        return [_row_to_memory(r) for r in rows]

    async def get_recent_for_agent_in_range(self, agent_id: str, start: str,
                                            end: str) -> list[Memory]:
        start_ts, end_ts = sim_timestamp(start), sim_timestamp(end)
        if start_ts is not None and end_ts is not None:
            return await self.get_in_sim_range(agent_id, start_ts, end_ts)
        return await self.get_recent(agent_id, limit=50)

    async def clear(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM memories")
            await db.commit()
        self._last_content.clear()

    async def apply_decay(self, agent_id: str, current_sim_timestamp: int | None = None):
        if current_sim_timestamp is None:
            return
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT id, sim_timestamp, importance, tier FROM memories "
                "WHERE agent_id = ? AND sim_timestamp IS NOT NULL", (agent_id,)
            )
            for memory_id, timestamp, importance, tier in await cursor.fetchall():
                age_days = max(0.0, (current_sim_timestamp - timestamp) / 1440.0)
                if age_days > 1:
                    decay_rate = 0.65 if tier == "working" else 0.995 if tier == "core" else 0.95
                    new_importance = max(1, round(importance * (decay_rate ** age_days)))
                    if new_importance != importance:
                        await db.execute(
                            "UPDATE memories SET importance = ? WHERE id = ?",
                            (new_importance, memory_id),
                        )
            await db.commit()

    async def consolidate(self, agent_id: str, memory_id: str):
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT recall_count, importance FROM memories WHERE id = ? AND agent_id = ?",
                (memory_id, agent_id),
            )
            row = await cursor.fetchone()
            if row is None:
                return
            old_count, importance = row[0] or 0, row[1]
            count = old_count + 1
            if old_count < 3 <= count:
                importance = min(10, importance + 2)
            await db.execute(
                "UPDATE memories SET recall_count = ?, importance = ? WHERE id = ?",
                (count, importance, memory_id),
            )
            await db.commit()

    async def merge_similar(self, agent_id: str):
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT id, time, location, content, importance, type, sim_day, sim_timestamp "
                "FROM memories WHERE agent_id = ? AND sim_day IS NOT NULL "
                "AND type NOT IN ('reflection', 'dialogue') ORDER BY sim_timestamp", (agent_id,)
            )
            rows = await cursor.fetchall()
        from collections import defaultdict
        groups = defaultdict(list)
        for row in rows:
            groups[(row[2], row[5], row[6])].append(row)
        for (location, mem_type, day), group in groups.items():
            if len(group) < 4:
                continue
            contents = [row[3] for row in group]
            ids = [row[0] for row in group]
            latest = max(row[7] for row in group)
            importance = min(10, round(sum(row[4] for row in group) / len(group)))
            summary = f"第{day}天在{location}的{mem_type}记录：" + "；".join(contents)
            async with aiosqlite.connect(self.db_path) as db:
                placeholders = ",".join("?" for _ in ids)
                await db.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", ids)
                await db.execute(
                    """INSERT INTO memories
                       (id, agent_id, time, location, content, importance, type,
                        recall_count, created_at, sim_day, sim_minute, sim_timestamp,
                        participants, emotion, source_ids)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, '[]', '', ?)""",
                    (f"mem_{uuid.uuid4().hex[:12]}", agent_id, group[-1][1], location,
                     summary, importance, mem_type, datetime.now().isoformat(), day,
                     latest % 1440, latest, json.dumps(ids)),
                )
                await db.commit()

    async def embed_all_memories(self, provider=None) -> None:
        provider = provider or self.embedding_provider
        if provider is None:
            return
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT id, content FROM memories WHERE embedding IS NULL")
            rows = await cursor.fetchall()
        for index in range(0, len(rows), 20):
            batch = rows[index:index + 20]
            embeddings = await provider.embed_many([row[1] for row in batch])
            async with aiosqlite.connect(self.db_path) as db:
                for (memory_id, _), embedding in zip(batch, embeddings):
                    await db.execute(
                        "UPDATE memories SET embedding = ? WHERE id = ?",
                        (json.dumps(embedding), memory_id),
                    )
                await db.commit()

    async def semantic_search(self, agent_id: str, query: str, provider=None,
                              limit: int = 5,
                              current_sim_timestamp: int | None = None) -> list[Memory]:
        provider = provider or self.embedding_provider
        if provider is None:
            return []
        query_embedding = await provider.embed_one(query)
        where = "agent_id = ? AND embedding IS NOT NULL"
        params: list = [agent_id]
        if current_sim_timestamp is not None:
            where += " AND sim_timestamp >= ? AND sim_timestamp <= ?"
            params.extend([current_sim_timestamp - 7 * 1440, current_sim_timestamp])
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(f"SELECT * FROM memories WHERE {where}", params)
            rows = await cursor.fetchall()
        scored = []
        for memory in map(_row_to_memory, rows):
            if memory.embedding:
                scored.append((_cosine_similarity(query_embedding, memory.embedding), memory))
        scored.sort(key=lambda item: item[0], reverse=True)
        result = [memory for _, memory in scored[:limit]]
        for memory in result:
            await self.consolidate(agent_id, memory.id)
        return result

    async def update_embedding(self, memory_id: str, embedding: list[float]) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE memories SET embedding = ? WHERE id = ?",
                (json.dumps(embedding), memory_id),
            )
            await db.commit()


def _json_list(value) -> list:
    if not value:
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (TypeError, json.JSONDecodeError):
        return []


def _row_to_memory(row: tuple) -> Memory:
    return Memory(
        id=row[0], agent_id=row[1], time=row[2], location=row[3],
        content=row[4], importance=row[5], type=row[6],
        embedding=_json_list(row[7]) or None,
        recall_count=(row[8] or 0) if len(row) > 8 else 0,
        created_at=(row[9] or "") if len(row) > 9 else "",
        sim_day=row[10] if len(row) > 10 else None,
        sim_minute=row[11] if len(row) > 11 else None,
        sim_timestamp=row[12] if len(row) > 12 else None,
        participants=_json_list(row[13]) if len(row) > 13 else [],
        emotion=(row[14] or "") if len(row) > 14 else "",
        source_ids=_json_list(row[15]) if len(row) > 15 else [],
        event_type=(row[16] or "") if len(row) > 16 else "",
        tier=(row[17] or "episodic") if len(row) > 17 else "episodic",
        fact_summary=(row[18] or "") if len(row) > 18 else "",
        interpretation=(row[19] or "") if len(row) > 19 else "",
        unresolved=bool(row[20]) if len(row) > 20 else False,
        future_intention=(row[21] or "") if len(row) > 21 else "",
        confidence=float(row[22] or 1.0) if len(row) > 22 else 1.0,
        formation_score=float(row[23] or 0.0) if len(row) > 23 else 0.0,
    )


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def memories_to_text(memories: list[Memory]) -> str:
    if not memories:
        return "（暂无相关记忆）"
    return "\n".join(
        f"- [{memory.time}] ({memory.location}, 重要性{memory.importance}) {memory.content}"
        for memory in memories
    )


def auto_importance(content: str, mem_type: str) -> int:
    base = 7 if mem_type == "dialogue" else 8 if mem_type == "reflection" else 5
    emotional_keywords = ["开心", "难过", "生气", "感动", "惊喜", "担心", "喜欢", "讨厌", "第一次"]
    if any(keyword in content for keyword in emotional_keywords):
        base = min(10, base + 2)
    return base
