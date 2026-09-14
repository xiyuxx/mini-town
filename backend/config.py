"""Updated configuration for Phase 1."""

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


class Config:
    # ── DeepSeek LLM ──
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    # DeepSeek V4 supports both modes on the same model.  Use the fast mode
    # for high-frequency work and reserve thinking for low-frequency synthesis.
    LLM_DEFAULT_MODE: str = os.getenv("LLM_DEFAULT_MODE", "chat")
    LLM_REASONING_EFFORT: str = os.getenv("LLM_REASONING_EFFORT", "high")
    # Prevent a burst of secondary LLM work from saturating the provider.
    LLM_MAX_CONCURRENCY: int = int(os.getenv("LLM_MAX_CONCURRENCY", "4"))
    LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))

    # ── Bailian Embedding ──
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "")
    EMBEDDING_BASE_URL: str = os.getenv(
        "EMBEDDING_BASE_URL",
        "https://ws-gm39fo0e2wutrc92.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    )
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
    EMBEDDING_MAX_CONCURRENCY: int = int(os.getenv("EMBEDDING_MAX_CONCURRENCY", "4"))
    EMBEDDING_TIMEOUT_SECONDS: float = float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "15"))
    EMBEDDING_MAX_RETRIES: int = int(os.getenv("EMBEDDING_MAX_RETRIES", "2"))
    # Backfill budget per startup for memories stored without a usable vector.
    EMBEDDING_REEMBED_MAX_ROWS: int = int(os.getenv("EMBEDDING_REEMBED_MAX_ROWS", "500"))

    # ── Database ──
    DB_PATH: str = os.path.join(os.path.dirname(__file__), "data", "town.db")

    # ── Simulation ──
    MOVE_SPEED: int = 2  # Two cells per tick keeps travel visible without consuming whole schedule blocks
    TICK_INTERVAL_MINUTES: int = 5
    REALTIME_TICK_SECONDS: float = 6.0
    SIM_START_HOUR: int = 6
    SIM_END_HOUR: int = 22
    MEMORY_RECENT_HOURS: int = 4
    MEMORY_MAX_RETRIEVE: int = 30
    REFLECTION_INTERVAL_HOURS: int = 2
    # Semantic recall scans at most this many candidate vectors per lookup
    # (0 = unbounded). Keeps a large memory table from turning every lookup
    # into a full-table vector scan.
    MEMORY_SEMANTIC_CANDIDATE_LIMIT: int = int(os.getenv("MEMORY_SEMANTIC_CANDIDATE_LIMIT", "2000"))
    MEMORY_SEMANTIC_WINDOW_DAYS: int = int(os.getenv("MEMORY_SEMANTIC_WINDOW_DAYS", "7"))
    # Similarity gates for model embeddings and for the local token-hash vectors.
    # The two spaces have different geometry, so they cannot share thresholds.
    MEMORY_ASSOCIATION_MIN_SIMILARITY: float = float(os.getenv("MEMORY_ASSOCIATION_MIN_SIMILARITY", "0.75"))
    MEMORY_REDUNDANCY_MIN_SIMILARITY: float = float(os.getenv("MEMORY_REDUNDANCY_MIN_SIMILARITY", "0.88"))
    # Hash-space gates, calibrated on measured pairs: restatements of one event
    # score 0.46-0.72, distinct events <= 0.34, near-identical text 0.90-0.98.
    MEMORY_HASH_ASSOCIATION_MIN_SIMILARITY: float = float(os.getenv("MEMORY_HASH_ASSOCIATION_MIN_SIMILARITY", "0.40"))
    MEMORY_HASH_REDUNDANCY_MIN_SIMILARITY: float = float(os.getenv("MEMORY_HASH_REDUNDANCY_MIN_SIMILARITY", "0.85"))
    MEMORY_INTERRUPT_MIN_SCORE: float = float(os.getenv("MEMORY_INTERRUPT_MIN_SCORE", "0.7"))
    SCHEDULE_DEVIATION_PROBABILITY: float = 0.0  # Disabled — too chaotic
    TOOL_CALL_MAX_PER_DECISION: int = 5
    TOOL_CALL_MAX_PER_DIALOGUE: int = 2

    # ── Weather ──
    WEATHER_CHANGE_PROBABILITY: float = 0.08  # Per 5-minute tick chance of weather change
    EXTREME_WEATHER_PROBABILITY: float = 0.02
    INFRASTRUCTURE_EVENT_PROBABILITY: float = 0.005  # ~0.7% per simulated day

    # ── Relationship ──
    RELATIONSHIP_FAMILIARITY_TALK_INCREMENT: float = 0.3
    RELATIONSHIP_FAMILIARITY_COLOCATION_INCREMENT: float = 0.1  # Being in same place
    RELATIONSHIP_AFFINITY_POSITIVE_INCREMENT: float = 0.2
    RELATIONSHIP_AFFINITY_NEGATIVE_INCREMENT: float = -0.3
    MAX_ANCHORS_PER_RELATIONSHIP: int = 10

    # ── Conversation pacing ──
    # The same two agents stay quiet with each other inside this window, and no
    # agent opens more than this many conversations per simulated day. Without
    # them the most available pair talks to each other all day and the town's
    # conversation budget never reaches the rest of the roster.
    DIALOGUE_PAIR_COOLDOWN_MINUTES: int = int(os.getenv("DIALOGUE_PAIR_COOLDOWN_MINUTES", "60"))
    DIALOGUE_MAX_PER_AGENT_PER_DAY: int = int(os.getenv("DIALOGUE_MAX_PER_AGENT_PER_DAY", "6"))


config = Config()
