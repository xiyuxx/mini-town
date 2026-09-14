"""Embedding provider with explicit single and batch APIs."""

import asyncio
import hashlib
import math
import random
import re
import warnings

import httpx

from ..config import config

# Local hash vectors live in their own geometry. The version tag is stored next
# to every vector so that vectors from an older scheme — or from a different
# model — are recognised as stale instead of being compared as if comparable.
HASH_SPACE_VERSION = "token-hash-2"
HASH_DIMS = 1024
_RETRYABLE_STATUS = {408, 409, 425, 429}


def similarity_gates(provider) -> dict:
    """Similarity thresholds valid for the space ``provider`` embeds into.

    Token-hash vectors and model embeddings do not share a distance scale, so a
    single threshold pair cannot serve both.
    """
    if provider is not None and getattr(provider, "fallback", False):
        return {
            "association": config.MEMORY_HASH_ASSOCIATION_MIN_SIMILARITY,
            "redundancy": config.MEMORY_HASH_REDUNDANCY_MIN_SIMILARITY,
        }
    return {
        "association": config.MEMORY_ASSOCIATION_MIN_SIMILARITY,
        "redundancy": config.MEMORY_REDUNDANCY_MIN_SIMILARITY,
    }


class EmbeddingProvider:
    def __init__(self):
        self.api_key = config.EMBEDDING_API_KEY
        self.base_url = config.EMBEDDING_BASE_URL.rstrip("/")
        self.model = config.EMBEDDING_MODEL
        self.fallback = not self.api_key or self.api_key in ("", "placeholder")
        self._client: httpx.AsyncClient | None = None
        self._request_slots = asyncio.Semaphore(max(1, config.EMBEDDING_MAX_CONCURRENCY))
        # Request bookkeeping, so a failing provider shows up as degraded
        # instead of looking like "this agent has no relevant memories".
        self.requests = 0
        self.failures = 0
        self.consecutive_failures = 0
        self.last_error = ""
        if self.fallback:
            warnings.warn("Embedding disabled or no API key — using local token-hash vectors")

    # ── Vector identity ───────────────────────────────────────

    def space_for(self, vector: list[float]) -> str:
        """Name of the vector space a freshly produced vector belongs to."""
        if self.fallback:
            return f"{HASH_SPACE_VERSION}:{len(vector)}"
        return f"{self.model}:{len(vector)}"

    @property
    def degraded(self) -> bool:
        return not self.fallback and self.consecutive_failures > 0

    # ── Public API ────────────────────────────────────────────

    async def embed_one(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise TypeError("embed_one expects a string")
        if self.fallback:
            return self._token_hash_embed(text)
        return (await self._request([text]))[0]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise TypeError("embed_many expects a list of strings")
        if not texts:
            return []
        if self.fallback:
            return [self._token_hash_embed(text) for text in texts]
        return await self._request(texts)

    async def close(self) -> None:
        """Release the shared HTTP connection pool during application shutdown."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ── Transport ─────────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=config.EMBEDDING_TIMEOUT_SECONDS)
        return self._client

    async def _request(self, texts: list[str]) -> list[list[float]]:
        self.requests += 1
        try:
            vectors = await self._request_with_retry(texts)
        except Exception as exc:
            self.failures += 1
            self.consecutive_failures += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            if self.failures == 1:
                warnings.warn(
                    f"Embedding request failed — semantic retrieval degrades: {self.last_error}"
                )
            raise
        self.consecutive_failures = 0
        return vectors

    async def _request_with_retry(self, texts: list[str]) -> list[list[float]]:
        allowed = max(0, config.EMBEDDING_MAX_RETRIES)
        for attempt in range(allowed + 1):
            try:
                return await self._post(texts)
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                if attempt >= allowed or not self._is_retryable(exc):
                    raise
                # Jitter keeps a burst of agents from retrying in lockstep.
                backoff = min(0.5 * 2 ** attempt, 4.0)
                await asyncio.sleep(backoff * (1.0 + random.random() * 0.25))
        raise AssertionError("retry loop exited without a result")

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            return status in _RETRYABLE_STATUS or status >= 500
        return True

    async def _post(self, texts: list[str]) -> list[list[float]]:
        client = await self._get_client()
        async with self._request_slots:
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.model, "input": texts},
            )
        response.raise_for_status()
        data = response.json().get("data", [])
        if len(data) != len(texts):
            raise RuntimeError(f"embedding API returned {len(data)} vectors for {len(texts)} texts")
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        vectors = [[float(value) for value in item["embedding"]] for item in ordered]
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise RuntimeError(f"embedding API returned mixed dimensions: {sorted(dimensions)}")
        return vectors

    # ── Similarity ────────────────────────────────────────────

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            raise ValueError(
                f"cannot compare vectors of different dimensions: {len(a)} vs {len(b)}"
            )
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    @staticmethod
    def _token_hash_embed(text: str, dims: int = HASH_DIMS) -> list[float]:
        """Sparse token hashing fallback that preserves lexical similarity.

        The sign is taken from a hash bit that is independent of the bucket
        index, so unrelated tokens that collide contribute zero in expectation
        instead of always adding similarity.
        """
        vector = [0.0] * dims
        normalized = text.lower().strip()
        tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", normalized)
        tokens += [normalized[index:index + 2] for index in range(max(0, len(normalized) - 1))]
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            vector[value % dims] += 1.0 if (value >> 63) & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector
