from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.evaluation.ragas_comprehensive import build_question_bank, run_comprehensive_evaluation


def test_question_bank_covers_all_requested_categories() -> None:
    questions = build_question_bank()
    assert len(questions) == 30

    counts: dict[str, int] = {}
    for question in questions:
        counts[question.category] = counts.get(question.category, 0) + 1

    assert counts == {
        "straightforward_factual": 9,
        "precise_attribution": 4,
        "cross_document_confusion": 3,
        "inference_grounding": 3,
        "safe_fail_unanswerable": 4,
        "ambiguous_rewrite": 4,
        "semantic_mismatch": 3,
    }


def test_question_bank_queries_are_unique() -> None:
    questions = build_question_bank()
    queries = [question.query for question in questions]
    assert len(queries) == len(set(queries))


@pytest.mark.integration
def test_ragas_comprehensive_live_run() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    reports_dir = backend_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "ragas_comprehensive_report.json"

    report = run_comprehensive_evaluation(
        api_url=os.environ.get("RAGAS_API_URL", "http://127.0.0.1:8000"),
        output_path=output_path,
        ollama_base_url=os.environ.get("RAGAS_OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_embedding_model=os.environ.get("RAGAS_OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
    )

    assert report["question_count"] == 30
    assert isinstance(report["mean_scores"], dict)
    assert len(report["rows"]) == 30
    assert output_path.exists()
