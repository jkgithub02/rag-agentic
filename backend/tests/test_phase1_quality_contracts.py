from __future__ import annotations

from src.core.config import Settings
from src.core.models import ConflictPolicy, ResponseCategory


def test_phase1_settings_contracts() -> None:
    settings = Settings(_env_file=None)
    assert settings.embedding_provider == "ollama"
    assert settings.reasoning_enabled is True
    assert settings.retrieval_top_k > 0


def test_response_categories_contract() -> None:
    expected = {
        "clarification",
        "safe_fail",
    }
    actual = {category.value for category in ResponseCategory}
    assert actual == expected


def test_upload_conflict_policy_contract() -> None:
    expected = {"ask", "replace", "keep_both"}
    actual = {policy.value for policy in ConflictPolicy}
    assert actual == expected
