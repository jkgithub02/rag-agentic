from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from src.agent.tools import AgentTools
from src.core.config import Settings
from src.core.models import ConflictPolicy, GroundingResult, GroundingStatus, QueryAnalysisOutput
from src.db.vector_db import VectorDbManager
from src.orchestration.pipeline import AgenticPipeline
from src.services.trace_store import TraceStore
from src.services.upload_service import UploadService


class _DeterministicEmbeddings(Embeddings):
    """Simple local embeddings for deterministic integration tests."""

    _TOKENS = (
        "jason",
        "kong",
        "resume",
        "backend",
        "agentic",
        "rag",
        "zephyria",
        "project",
    )

    @classmethod
    def _to_vector(cls, text: str) -> list[float]:
        lowered = text.lower()
        vector: list[float] = [float(lowered.count(token)) for token in cls._TOKENS]
        # Keep vectors non-zero for cosine distance in Qdrant.
        if sum(vector) == 0.0:
            vector[-1] = 1.0
        return vector

    def embed_query(self, text: str) -> list[float]:
        return self._to_vector(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._to_vector(text) for text in texts]


class _DeterministicReasoner:
    """Reasoner stub that keeps orchestration model-driven without external LLM calls."""

    def summarize_conversation(self, history: list[dict[str, str]]) -> str:
        del history
        return ""

    def analyze_query(
        self,
        *,
        query: str,
        conversation_summary: str | None = None,
    ) -> tuple[QueryAnalysisOutput, str]:
        del conversation_summary
        return (
            QueryAnalysisOutput(
                is_clear=True,
                rewritten_query=query,
                clarification_needed=None,
                prompt_version="test-v1",
            ),
            "llm",
        )

    def synthesize_answer(
        self, *, query: str, chunks: list, subqueries: list[str] | None = None
    ) -> tuple[str, list[str], str, str | None]:
        del query, subqueries
        if not chunks:
            raise ValueError("No chunks to synthesize from.")
        answer = f"Evidence found in {chunks[0].source}."
        return answer, [chunks[0].chunk_id], "llm", "test-v1"

    def assess_grounding(
        self,
        *,
        answer: str,
        citations: list[str],
        evidence: list[str],
    ) -> tuple[GroundingResult, str, str | None]:
        del answer, citations, evidence
        return GroundingResult(status=GroundingStatus.SUPPORTED, reason="Grounded."), "llm", "test-v1"


@pytest.fixture
def e2e_settings(tmp_path: Path) -> Settings:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        documents_dir=docs_dir,
        vector_db_path=tmp_path / "qdrant",
        retrieval_top_k=4,
        min_relevance_score=0.05,
        ambiguity_margin=0.03,
    )


@pytest.fixture(autouse=True)
def deterministic_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        VectorDbManager,
        "_build_embeddings",
        staticmethod(lambda settings: _DeterministicEmbeddings()),
    )


def test_e2e_upload_indexes_real_chunks(e2e_settings: Settings) -> None:
    vector_db = VectorDbManager(e2e_settings)
    upload_service = UploadService(settings=e2e_settings, vector_db=vector_db)

    response = upload_service.upload_bytes(
        filename="jason_kong_resume.txt",
        content=(
            b"Jason Kong resume. Zephyria project owner. "
            b"Agentic RAG backend implementation experience."
        ),
        conflict_policy=ConflictPolicy.REPLACE,
    )

    assert response.status == "success"
    assert response.stored_filename == "jason_kong_resume.txt"
    assert response.chunks_added is not None
    assert response.chunks_added > 0

    docs = upload_service.list_documents()
    assert len(docs) == 1
    assert docs[0]["filename"] == "jason_kong_resume.txt"
    assert int(docs[0]["chunks_indexed"]) > 0


def test_e2e_search_then_fetch_by_ids_returns_indexed_source(e2e_settings: Settings) -> None:
    vector_db = VectorDbManager(e2e_settings)
    upload_service = UploadService(settings=e2e_settings, vector_db=vector_db)
    tools = AgentTools(vector_db)

    upload_service.upload_bytes(
        filename="jason_kong_resume.txt",
        content=b"Jason Kong resume with Zephyria project highlights.",
        conflict_policy=ConflictPolicy.REPLACE,
    )

    hits = tools.search_chunks("Zephyria project", top_k=3)
    assert len(hits) > 0
    assert hits[0].source == "jason_kong_resume.txt"

    fetched = tools.fetch_chunks_by_ids([hit.chunk_id for hit in hits])
    assert len(fetched) == len(hits)
    assert all(chunk.source == "jason_kong_resume.txt" for chunk in fetched)


def test_e2e_pipeline_query_uses_retrieved_chunks_and_citations(e2e_settings: Settings) -> None:
    vector_db = VectorDbManager(e2e_settings)
    upload_service = UploadService(settings=e2e_settings, vector_db=vector_db)
    tools = AgentTools(vector_db)
    trace_store = TraceStore(storage_dir=e2e_settings.vector_db_path.parent / "traces")

    upload_service.upload_bytes(
        filename="jason_kong_resume.txt",
        content=(
            b"Jason Kong resume. Zephyria project owner. "
            b"Built agentic RAG backend services and retrieval tooling."
        ),
        conflict_policy=ConflictPolicy.REPLACE,
    )

    pipeline = AgenticPipeline(
        settings=e2e_settings,
        tools=tools,
        trace_store=trace_store,
        reasoner=_DeterministicReasoner(),
    )

    response = pipeline.ask("Summarise Zephyria project from Jason Kong resume")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    assert response.safe_fail is False
    assert len(response.citations) > 0
    assert any(citation.startswith("jason_kong_resume.txt#") for citation in response.citations)

    retrieve_events = [event for event in trace.events if event.stage == "retrieve"]
    assert len(retrieve_events) == 1
    hits = retrieve_events[0].payload.get("hits", [])
    assert isinstance(hits, list)
    assert len(hits) > 0
    assert hits[0]["source"] == "jason_kong_resume.txt"

    fetch_events = [event for event in trace.events if event.stage == "tool_fetch_chunks_by_ids"]
    assert len(fetch_events) == 1
    fetched_ids = fetch_events[0].payload.get("chunk_ids", [])
    assert isinstance(fetched_ids, list)
    assert len(fetched_ids) > 0


def test_e2e_pipeline_mixed_top_sources_remains_answerable(e2e_settings: Settings) -> None:
    vector_db = VectorDbManager(e2e_settings)
    upload_service = UploadService(settings=e2e_settings, vector_db=vector_db)
    tools = AgentTools(vector_db)
    trace_store = TraceStore(storage_dir=e2e_settings.vector_db_path.parent / "traces")

    upload_service.upload_bytes(
        filename="jason_kong_resume.txt",
        content=b"Resume profile with backend experience.",
        conflict_policy=ConflictPolicy.REPLACE,
    )
    upload_service.upload_bytes(
        filename="jason_wong_resume.txt",
        content=b"Resume profile with backend experience.",
        conflict_policy=ConflictPolicy.REPLACE,
    )

    pipeline = AgenticPipeline(
        settings=e2e_settings,
        tools=tools,
        trace_store=trace_store,
        reasoner=_DeterministicReasoner(),
    )

    response = pipeline.ask("Summarise the resume profile")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    assert response.safe_fail is False
    assert len(response.citations) > 0

    validate_events = [event for event in trace.events if event.stage == "validate"]
    assert len(validate_events) == 1
    assert validate_events[0].payload.get("status") == "pass"
