"""Contract tests for the embedding path: vector identity, read purity, recovery."""

import asyncio

import aiosqlite
import httpx
import pytest

from backend.config import config
from backend.town.embedding import EmbeddingProvider
from backend.town.engine import SimulationEngine
from backend.town.experience import ExperienceEvent, MemoryCandidate, MemoryFormation
from backend.town.memory import MemoryStore


def run(coro):
    return asyncio.run(coro)


class LiveStubProvider(EmbeddingProvider):
    """Answers locally while presenting itself as a live provider.

    Overrides the transport seam only, so space naming, pause/retry bookkeeping
    and the rest of the provider behave exactly as against a real endpoint.
    """

    def __init__(self, dims: int = 8):
        super().__init__()
        self.fallback = False
        self.model = "stub-model"
        self.dims = dims

    async def _post(self, texts):
        vectors = []
        for text in texts:
            vector = [0.0] * self.dims
            for index, char in enumerate(text):
                vector[index % self.dims] += 1.0 + (ord(char) % 7) / 10.0
            norm = sum(value * value for value in vector) ** 0.5 or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


class OutageStubProvider(EmbeddingProvider):
    """Live-looking provider whose transport is down."""

    def __init__(self):
        super().__init__()
        self.fallback = False
        self.model = "stub-model"

    async def _post(self, texts):
        raise httpx.ConnectError("provider down")


class FlakyStubProvider(LiveStubProvider):
    """Fails the first transport attempt, then behaves normally."""

    def __init__(self):
        super().__init__()
        self.attempts = 0

    async def _post(self, texts):
        self.attempts += 1
        if self.attempts == 1:
            raise httpx.ConnectError("transient")
        return await super()._post(texts)


async def make_store(db_path, provider=None) -> MemoryStore:
    store = MemoryStore()
    store.db_path = str(db_path)
    store.set_embedding_provider(provider)
    await store.init_db()
    return store


def test_semantic_lookup_leaves_memories_unchanged(tmp_path):
    async def scenario():
        provider = EmbeddingProvider()
        store = await make_store(tmp_path / "town.db", provider)
        memory = await store.add("a", "park", "在公园和老王下棋", "observation", 8,
                                 sim_time="第1天 08:00")
        now = memory.sim_timestamp + 60
        for _ in range(4):
            await store.semantic_search("a", "在公园和老王下棋", provider,
                                        current_sim_timestamp=now)
        stored = next(row for row in await store.get_recent("a", hours=100, limit=5,
                                                            current_sim_timestamp=now)
                      if row.id == memory.id)
        assert (stored.recall_count, stored.importance) == (0, 8)

    run(scenario())


def test_reinforce_counts_only_the_memories_it_is_given(tmp_path):
    async def scenario():
        provider = EmbeddingProvider()
        store = await make_store(tmp_path / "town.db", provider)
        recalled = await store.add("a", "park", "在公园和老王下棋", "observation", 6,
                                   sim_time="第1天 08:00")
        untouched = await store.add("a", "park", "在餐馆吃了早饭", "observation", 6,
                                    sim_time="第1天 09:00")
        for _ in range(3):
            await store.reinforce("a", [recalled.id])
        rows = {row.id: row for row in await store.get_all_for_agent("a")}
        assert rows[recalled.id].recall_count == 3
        assert rows[recalled.id].importance == 8
        assert rows[untouched.id].recall_count == 0
        assert rows[untouched.id].importance == 6

    run(scenario())


def test_vectors_from_another_space_are_never_compared(tmp_path):
    async def scenario():
        provider = EmbeddingProvider()
        store = await make_store(tmp_path / "town.db", provider)
        memory = await store.add("a", "park", "在公园和老王下棋", "observation", 8,
                                 sim_time="第1天 08:00")
        now = memory.sim_timestamp + 60
        assert await store.semantic_search("a", "在公园和老王下棋", provider,
                                           current_sim_timestamp=now)
        async with aiosqlite.connect(store.db_path) as db:
            await db.execute("UPDATE memories SET embedding_space = ? WHERE id = ?",
                             ("some-other-model:1024", memory.id))
            await db.commit()
        assert await store.semantic_search("a", "在公园和老王下棋", provider,
                                           current_sim_timestamp=now) == []

    run(scenario())


def test_reembed_stale_rebuilds_missing_and_foreign_vectors(tmp_path):
    async def scenario():
        store = await make_store(tmp_path / "town.db")
        contents = ["在公园和老王下棋", "早上去餐馆吃早饭", "在图书馆看了一下午的书"]
        for index, content in enumerate(contents):
            await store.add("a", "park", content, "observation", 8,
                            sim_time=f"第1天 {6 + index:02d}:00")
        provider = LiveStubProvider()
        store.set_embedding_provider(provider)
        assert await store.reembed_stale(provider) == 0
        async with aiosqlite.connect(store.db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM memories WHERE embedding IS NULL")
            assert (await cursor.fetchone())[0] == 0
            cursor = await db.execute("SELECT DISTINCT embedding_space FROM memories")
            assert {row[0] for row in await cursor.fetchall()} == {"stub-model:8"}
        hits = await store.semantic_search("a", contents[0], provider,
                                           current_sim_timestamp=9 * 60)
        assert hits and hits[0].content == contents[0]

    run(scenario())


def test_reembed_stale_keeps_model_vectors_when_provider_is_offline(tmp_path):
    async def scenario():
        store = await make_store(tmp_path / "town.db")
        await store.add("a", "park", "在公园和老王下棋", "observation", 8,
                        sim_time="第1天 08:00")
        live = LiveStubProvider()
        store.set_embedding_provider(live)
        await store.reembed_stale(live)
        async with aiosqlite.connect(store.db_path) as db:
            cursor = await db.execute("SELECT embedding FROM memories")
            stored_live = (await cursor.fetchone())[0]

        hash_provider = EmbeddingProvider()
        store.set_embedding_provider(hash_provider)
        assert await store.reembed_stale(hash_provider) == 0
        async with aiosqlite.connect(store.db_path) as db:
            cursor = await db.execute("SELECT embedding, embedding_space FROM memories")
            stored_after, space_after = await cursor.fetchone()
        assert stored_after == stored_live
        assert space_after == "stub-model:8"

    run(scenario())


def test_memories_without_parsable_time_stay_retrievable(tmp_path):
    async def scenario():
        provider = EmbeddingProvider()
        store = await make_store(tmp_path / "town.db", provider)
        dated = await store.add("a", "park", "在公园和老王下棋", "observation", 8,
                                sim_time="第1天 08:00")
        undated = await store.add("a", "park", "在公园和老王下棋，他连输了三局", "observation", 8)
        hits = await store.semantic_search("a", "在公园和老王下棋", provider,
                                           current_sim_timestamp=dated.sim_timestamp + 60)
        assert {memory.id for memory in hits} == {dated.id, undated.id}

    run(scenario())


def test_cosine_similarity_rejects_mismatched_dimensions():
    with pytest.raises(ValueError):
        EmbeddingProvider.cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


def test_provider_outage_does_not_discard_experiences(tmp_path, monkeypatch):
    async def scenario():
        monkeypatch.setattr(config, "EMBEDDING_MAX_RETRIES", 0)
        provider = OutageStubProvider()
        store = await make_store(tmp_path / "town.db", provider)
        await store.add("a", "park", "在公园和老王下棋", "observation", 8,
                        sim_time="第1天 08:00")
        formation = MemoryFormation(store, llm=None, trace=None)
        candidate = MemoryCandidate(
            event=ExperienceEvent(type="activity_outcome", location="park", facts={}, actors=["a"]),
            score=0.5, tier="working", importance=7,
            fact_summary="在公园和老王下棋", interpretation_hint="", emotion="",
        )
        assert await formation._is_redundant("a", candidate, 1000) is False
        # ...so the experience is stored without a vector instead of being dropped.
        stored = await store.add("a", "park", "在公园和老王下棋又赢了一局", "observation", 7,
                                 sim_time="第1天 09:00", event_type="activity_outcome")
        assert stored is not None and stored.embedding is None
        assert len(await store.get_all_for_agent("a")) == 2

    run(scenario())


def test_merge_keeps_the_merged_memory_retrievable(tmp_path):
    async def scenario():
        provider = EmbeddingProvider()
        store = await make_store(tmp_path / "town.db", provider)
        for index in range(4):
            await store.add("a", "park", f"在公园散步第{index}次，天气不错", "observation", 8,
                            sim_time=f"第1天 {6 + index:02d}:00", event_type="activity_outcome")
        await store.merge_similar("a")
        rows = await store.get_all_for_agent("a")
        assert len(rows) == 1
        merged = rows[0]
        assert merged.embedding
        assert merged.event_type == "activity_outcome"
        hits = await store.semantic_search("a", merged.content, provider,
                                           current_sim_timestamp=1000)
        assert hits and hits[0].id == merged.id

    run(scenario())


def test_transient_transport_failure_is_retried(tmp_path, monkeypatch):
    async def scenario():
        monkeypatch.setattr(config, "EMBEDDING_MAX_RETRIES", 1)
        provider = FlakyStubProvider()
        assert len(await provider.embed_one("公园下棋")) == 8
        assert provider.attempts == 2
        # A call that eventually succeeded must not read as a degraded provider.
        assert (provider.failures, provider.consecutive_failures, provider.degraded) == (0, 0, False)

    run(scenario())


def test_degraded_provider_is_visible_in_health(tmp_path, monkeypatch):
    async def scenario():
        monkeypatch.setattr(config, "EMBEDDING_MAX_RETRIES", 0)
        engine = SimulationEngine()
        engine.llm.fallback = True
        engine.memory.db_path = str(tmp_path / "town.db")
        engine.relationship_store.db_path = str(tmp_path / "town.db")
        engine.trace.db_path = str(tmp_path / "town.db")
        await engine.init()
        assert engine.get_state()["health"]["embedding"] == "fallback"

        engine.embedding_provider = OutageStubProvider()
        assert engine.get_state()["health"]["embedding"] == "connected"
        with pytest.raises(httpx.ConnectError):
            await engine.embedding_provider.embed_one("测试")
        health = engine.get_state()["health"]
        assert health["embedding"] == "degraded"
        assert "provider down" in health["embeddingError"]

    run(scenario())
