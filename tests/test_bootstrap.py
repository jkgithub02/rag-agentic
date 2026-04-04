from __future__ import annotations

from pathlib import Path

import pytest

import src.bootstrap as bootstrap
from src.core.config import Settings


class _LockingVectorDb:
    def __init__(self, settings: Settings) -> None:
        del settings
        raise RuntimeError("database lock file is already used by another process")


def test_get_vector_db_reports_friendly_lock_message(monkeypatch: pytest.MonkeyPatch) -> None:
    bootstrap.get_vector_db.cache_clear()
    monkeypatch.setattr(
        bootstrap,
        "get_settings",
        lambda: Settings(vector_db_path=Path(".data/qdrant-test")),
    )
    monkeypatch.setattr(bootstrap, "VectorDbManager", _LockingVectorDb)

    with pytest.raises(RuntimeError, match="Vector database path is locked"):
        bootstrap.get_vector_db()

    bootstrap.get_vector_db.cache_clear()
