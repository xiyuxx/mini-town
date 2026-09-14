"""Embedding provider with explicit single and batch APIs."""

import hashlib
import math
import re
import httpx

from ..config import config


class EmbeddingProvider:
    def __init__(self):
        self.api_key = config.EMBEDDING_API_KEY
        self.base_url = config.EMBEDDING_BASE_URL.rstrip("/")
        self.model = config.EMBEDDING_MODEL
        self.fallback = not self.api_key or self.api_key in ("", "placeholder")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=15.0)
        return self._client

    async def embed_one(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise TypeError("embed_one expects a string")
        if self.fallback:
            return self._token_hash_embed(text)
        embeddings = await self._request([text])
        return embeddings[0]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise TypeError("embed_many expects a list of strings")
        if not texts:
            return []
        if self.fallback:
            return [self._token_hash_embed(text) for text in texts]
        return await self._request(texts)

    async def embed(self, value):
        """Compatibility shim for old callers."""
        if isinstance(value, str):
            return await self.embed_one(value)
        if isinstance(value, list):
            return await self.embed_many(value)
        raise TypeError("embed expects a string or list of strings")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return await self.embed_many(texts)

    async def _request(self, texts: list[str]) -> list[list[float]]:
        client = await self._get_client()
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
        return [item["embedding"] for item in ordered]

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    @staticmethod
    def _token_hash_embed(text: str, dims: int = 1024) -> list[float]:
        """Sparse token hashing fallback that preserves lexical similarity."""
        vector = [0.0] * dims
        normalized = text.lower().strip()
        tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", normalized)
        tokens += [normalized[index:index + 2] for index in range(max(0, len(normalized) - 1))]
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % dims
            vector[index] += 1.0 if value & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector
