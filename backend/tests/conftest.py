import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import pytest

from backend.config import config


@pytest.fixture(autouse=True)
def isolated_database(tmp_path, monkeypatch):
    """Give every test a fresh default SQLite path."""
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "town.db"))


@pytest.fixture(autouse=True)
def offline_embeddings(monkeypatch):
    """Keep the suite off the network: embeddings always use local hash vectors.

    A developer .env holding a real EMBEDDING_API_KEY would otherwise make the
    suite call the live provider — slow, nondeterministic, and billed.
    """
    monkeypatch.setattr(config, "EMBEDDING_API_KEY", "")
