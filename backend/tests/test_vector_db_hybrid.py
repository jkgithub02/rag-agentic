from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.embeddings import Embeddings

from src.core.config import Settings
from src.core.models import EvidenceChunk
from src.db.vector_db import VectorDbManager


class _TestEmbeddings(Embeddings):
    def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            float("context" in lowered or "relationships" in lowered),
            float("objective" in lowered or "self-supervised" in lowered),
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float("attention" in lowered or "context" in lowered),
                    float("masked language model" in lowered or "objective" in lowered),
                ]
            )
        return vectors


class _FakeQdrantClient:
    def __init__(self, points: list[SimpleNamespace]) -> None:
        self._points = points

    def query_points(self, **_: object) -> SimpleNamespace:
        return SimpleNamespace(points=list(self._points))


def _point(chunk_id: str, source: str, text: str, score: float) -> SimpleNamespace:
    return SimpleNamespace(
        payload={"chunk_id": chunk_id, "source": source, "text": text},
        score=score,
    )


def _make_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **settings_overrides: object) -> VectorDbManager:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        documents_dir=docs_dir,
        vector_db_path=tmp_path / "qdrant",
        retrieval_top_k=4,
        min_relevance_score=0.05,
        **settings_overrides,
    )
    monkeypatch.setattr(
        VectorDbManager,
        "_build_embeddings",
        staticmethod(lambda _settings: _TestEmbeddings()),
    )
    return VectorDbManager(settings)


def test_hybrid_retrieval_fuses_dense_and_sparse_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _make_manager(
        tmp_path,
        monkeypatch,
        retrieval_mode="hybrid",
        retrieval_dense_weight=0.4,
        retrieval_sparse_weight=0.6,
        retrieval_neighbor_span=0,
    )

    (manager._settings.documents_dir / "doc_attention.txt").write_text("attention context", encoding="utf-8")
    (manager._settings.documents_dir / "doc_objective.txt").write_text("masked language model objective", encoding="utf-8")

    manager._chunk_lookup = {
        "attn-0001": EvidenceChunk(
            chunk_id="attn-0001",
            source="doc_attention.txt",
            text="Attention mechanisms aggregate context across tokens.",
            score=0.0,
        ),
        "obj-0001": EvidenceChunk(
            chunk_id="obj-0001",
            source="doc_objective.txt",
            text="BERT uses masked language model objective during pretraining.",
            score=0.0,
        ),
    }
    manager._client = _FakeQdrantClient(
        [
            _point("attn-0001", "doc_attention.txt", "Attention mechanisms aggregate context.", 0.93),
            _point("obj-0001", "doc_objective.txt", "BERT masked language model objective.", 0.35),
        ]
    )

    hits = manager.search("What self-supervised objective is used?", top_k=2)

    assert len(hits) >= 2
    assert hits[0].chunk_id == "obj-0001"


def test_sparse_mode_can_answer_when_dense_hits_are_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _make_manager(
        tmp_path,
        monkeypatch,
        retrieval_mode="sparse",
        retrieval_neighbor_span=0,
    )

    (manager._settings.documents_dir / "doc_attention.txt").write_text("attention context", encoding="utf-8")

    manager._chunk_lookup = {
        "attn-0001": EvidenceChunk(
            chunk_id="attn-0001",
            source="doc_attention.txt",
            text="Attention mechanisms aggregate context across tokens.",
            score=0.0,
        )
    }
    manager._client = _FakeQdrantClient([])

    hits = manager.search("How is context aggregated?", top_k=1)

    assert len(hits) == 1
    assert hits[0].chunk_id == "attn-0001"


def test_dense_mode_uses_dense_results_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _make_manager(
        tmp_path,
        monkeypatch,
        retrieval_mode="dense",
        retrieval_neighbor_span=0,
    )

    (manager._settings.documents_dir / "doc_attention.txt").write_text("attention context", encoding="utf-8")
    (manager._settings.documents_dir / "doc_objective.txt").write_text("objective", encoding="utf-8")

    manager._chunk_lookup = {
        "attn-0001": EvidenceChunk(
            chunk_id="attn-0001",
            source="doc_attention.txt",
            text="Attention mechanisms aggregate context across tokens.",
            score=0.0,
        ),
        "obj-0001": EvidenceChunk(
            chunk_id="obj-0001",
            source="doc_objective.txt",
            text="BERT uses masked language model objective during pretraining.",
            score=0.0,
        ),
    }
    manager._client = _FakeQdrantClient(
        [
            _point("attn-0001", "doc_attention.txt", "Attention mechanisms aggregate context.", 0.88),
            _point("obj-0001", "doc_objective.txt", "BERT objective", 0.10),
        ]
    )

    hits = manager.search("How does the model learn word relationships?", top_k=2)

    assert len(hits) >= 2
    assert hits[0].chunk_id == "attn-0001"
