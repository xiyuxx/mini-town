"""SQLite-backed episodic memory with simulation-time retrieval."""

import aiosqlite
import json
import os
import struct
import uuid
from dataclasses import dataclass
from datetime import datetime

from backend.config import config
from .embedding import EmbeddingProvider
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

    def _space_for(self, embedding: list[float] | None) -> str:
        if not embedding or self.embedding_provider is None:
            return ""
        return self.embedding_provider.space_for(embedding)

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
                # Vector space of the stored vector (model:dimension, or the
                # local hash scheme). Vectors from another space are stale, not
                # comparable, and get rebuilt by reembed_stale(). Appended last
                # so the positional row mapping stays valid for databases created
                # before this column existed.
                "embedding_space": "TEXT DEFAULT ''",
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
                # Stored without a vector; reembed_stale() retries it later
                # because the provider records the failure.
                embedding = None
        embedding_space = self._space_for(embedding)

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
                    embedding, embedding_space, recall_count, created_at, sim_day,
                    sim_minute, sim_timestamp, participants, emotion, source_ids,
                    event_type, tier, fact_summary, interpretation, unresolved,
                    future_intention, confidence, formation_score)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    mem.id, mem.agent_id, mem.time, mem.location, mem.content,
                    mem.importance, mem.type,
                    _encode_embedding(embedding), embedding_space,
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

    async def reinforce(self, agent_id: str, memory_ids: list[str]) -> None:
        """Count a genuine recall; the third recall makes a memory stickier.

        Explicit on purpose: lookups must not mutate the store, so callers name
        the memories they actually re-surfaced.
        """
        if not memory_ids:
            return
        placeholders = ",".join("?" for _ in memory_ids)
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                f"SELECT id, recall_count, importance FROM memories "
                f"WHERE agent_id = ? AND id IN ({placeholders})",
                (agent_id, *memory_ids),
            )
            rows = await cursor.fetchall()
            for memory_id, recall_count, importance in rows:
                previous = recall_count or 0
                count = previous + 1
                if previous < 3 <= count:
                    importance = min(10, importance + 2)
                await db.execute(
                    "UPDATE memories SET recall_count = ?, importance = ? WHERE id = ?",
                    (count, importance, memory_id),
                )
            await db.commit()

    async def merge_similar(self, agent_id: str):
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT id, time, location, content, importance, type, sim_day, sim_timestamp, "
                "event_type, tier FROM memories WHERE agent_id = ? AND sim_day IS NOT NULL "
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
            # The merged row keeps the newest source's provenance, otherwise the
            # summary becomes invisible to event-type based dedup.
            newest = max(group, key=lambda row: row[7])
            merged_id = f"mem_{uuid.uuid4().hex[:12]}"
            async with aiosqlite.connect(self.db_path) as db:
                placeholders = ",".join("?" for _ in ids)
                await db.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", ids)
                await db.execute(
                    """INSERT INTO memories
                       (id, agent_id, time, location, content, importance, type,
                        recall_count, created_at, sim_day, sim_minute, sim_timestamp,
                        participants, emotion, source_ids, event_type, tier, fact_summary)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, '[]', '', ?, ?, ?, ?)""",
                    (merged_id, agent_id, newest[1], location,
                     summary, importance, mem_type, datetime.now().isoformat(), day,
                     latest % 1440, latest, json.dumps(ids), newest[8], newest[9], summary),
                )
                await db.commit()
            await self._attach_embedding(merged_id, summary)

    async def _attach_embedding(self, memory_id: str, text: str) -> None:
        """Best-effort vector write for a single row (used by merges)."""
        provider = self.embedding_provider
        if provider is None or not text.strip():
            return
        try:
            vector = await provider.embed_one(text)
        except Exception:
            return
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE memories SET embedding = ?, embedding_space = ? WHERE id = ?",
                (_encode_embedding(vector), self._space_for(vector), memory_id),
            )
            await db.commit()

    async def reembed_stale(self, provider=None, max_rows: int | None = None) -> int:
        """Rebuild memories whose vector is missing or from another space.

        Returns the number of rows still stale afterwards. A live provider names
        the target space with one probe request, which also makes a bad key
        visible at startup. Fallback providers are skipped: they have nothing
        better to write, and overwriting stored model embeddings with hash
        vectors would destroy them.
        """
        provider = provider or self.embedding_provider
        if provider is None or provider.fallback:
            return 0
        budget = config.EMBEDDING_REEMBED_MAX_ROWS if max_rows is None else max_rows
        if not await self._count_rows():
            return 0
        try:
            probe = await provider.embed_many(["probe"])
        except Exception:
            return await self._count_stale_stale_unknown()
        stale_where = (
            "(embedding IS NULL OR embedding_space != ?) "
            "AND (importance >= 6 OR COALESCE(event_type, '') != '')"
        )
        params = [provider.space_for(probe[0])]
        processed = 0
        while budget <= 0 or processed < budget:
            batch_size = 20 if budget <= 0 else min(20, budget - processed)
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    f"SELECT id, content FROM memories WHERE {stale_where} "
                    "ORDER BY COALESCE(sim_timestamp, 0) DESC LIMIT ?",
                    (*params, batch_size),
                )
                batch = await cursor.fetchall()
            if not batch:
                break
            try:
                vectors = await provider.embed_many([content for _, content in batch])
            except Exception:
                # Provider recorded the failure; the rest is retried next startup.
                break
            space = provider.space_for(vectors[0])
            async with aiosqlite.connect(self.db_path) as db:
                for (memory_id, _), vector in zip(batch, vectors):
                    await db.execute(
                        "UPDATE memories SET embedding = ?, embedding_space = ? WHERE id = ?",
                        (_encode_embedding(vector), space, memory_id),
                    )
                await db.commit()
            processed += len(batch)
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                f"SELECT COUNT(*) FROM memories WHERE {stale_where}", params
            )
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def _count_rows(self) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM memories")
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def _count_stale_stale_unknown(self) -> int:
        """Stale rows counted without knowing the live space (probe failed)."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM memories WHERE embedding IS NULL "
                "AND (importance >= 6 OR COALESCE(event_type, '') != '')"
            )
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def semantic_search_scored(self, agent_id: str, query: str, provider=None,
                                     limit: int = 5,
                                     current_sim_timestamp: int | None = None
                                     ) -> list[tuple[Memory, float]]:
        """Score the agent's stored vectors against ``query``, highest first.

        Read-only: recalling a memory does not modify it. Only rows embedded in
        the same space as the query take part — vectors from another model or
        from the local hash scheme are rebuilt by reembed_stale() rather than
        being compared as if the numbers were comparable.
        """
        provider = provider or self.embedding_provider
        if provider is None or not query.strip():
            return []
        query_embedding = await provider.embed_one(query)
        where = ["agent_id = ?", "embedding IS NOT NULL", "embedding_space = ?"]
        params: list = [agent_id, provider.space_for(query_embedding)]
        if current_sim_timestamp is not None:
            where.append("(sim_timestamp IS NULL OR (sim_timestamp >= ? AND sim_timestamp <= ?))")
            params.extend([
                current_sim_timestamp - config.MEMORY_SEMANTIC_WINDOW_DAYS * 1440,
                current_sim_timestamp,
            ])
        sql = (
            "SELECT id, embedding FROM memories WHERE " + " AND ".join(where) +
            " ORDER BY COALESCE(sim_timestamp, 0) DESC"
        )
        if config.MEMORY_SEMANTIC_CANDIDATE_LIMIT > 0:
            sql += " LIMIT ?"
            params.append(config.MEMORY_SEMANTIC_CANDIDATE_LIMIT)
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(sql, params)
            rows = await cursor.fetchall()
        scored: list[tuple[float, str]] = []
        for memory_id, blob in rows:
            vector = _decode_embedding(blob)
            if not vector or len(vector) != len(query_embedding):
                continue
            scored.append((EmbeddingProvider.cosine_similarity(query_embedding, vector), memory_id))
        if not scored:
            return []
        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[:max(0, limit)]
        placeholders = ",".join("?" for _ in top)
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                f"SELECT * FROM memories WHERE id IN ({placeholders})",
                [memory_id for _, memory_id in top],
            )
            loaded = {row[0]: _row_to_memory(row) for row in await cursor.fetchall()}
        return [(loaded[memory_id], score) for score, memory_id in top if memory_id in loaded]

    async def semantic_search(self, agent_id: str, query: str, provider=None,
                              limit: int = 5,
                              current_sim_timestamp: int | None = None) -> list[Memory]:
        scored = await self.semantic_search_scored(
            agent_id, query, provider, limit, current_sim_timestamp
        )
        return [memory for memory, _ in scored]


def _encode_embedding(vector: list[float] | None) -> bytes | None:
    """Pack a vector as float32; JSON text costs ~5x the space for the same data."""
    if not vector:
        return None
    return struct.pack(f"<{len(vector)}f", *vector)


def _decode_embedding(value) -> list[float] | None:
    """Read a vector stored packed, or in the legacy JSON text format."""
    if not value:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if len(raw) % 4:
            return None
        return list(struct.unpack(f"<{len(raw) // 4}f", raw))
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(parsed, list):
            return None
        try:
            return [float(item) for item in parsed]
        except (TypeError, ValueError):
            return None
    return None


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
        embedding=_decode_embedding(row[7]),
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
