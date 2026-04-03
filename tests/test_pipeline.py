from __future__ import annotations

from pathlib import Path

from src.config import Settings
from src.models import (
    EvidenceChunk,
    GroundingResult,
    GroundingStatus,
    QueryRewriteOutput,
)
from src.orchestration.pipeline import AgenticPipeline
from src.trace_store import TraceStore


class FakeTools:
    def search_chunks(self, query: str, top_k: int) -> list[EvidenceChunk]:
        del top_k
        if "quantum" in query.lower():
            return []
        return [
            EvidenceChunk(
                chunk_id="bert-0001",
                source="bert.pdf",
                text="BERT uses masked language modeling.",
                score=0.7,
            )
        ]

    def fetch_chunks_by_ids(self, chunk_ids: list[str]) -> list[EvidenceChunk]:
        del chunk_ids
        return []


class FakeReasoner:
    def __init__(
        self,
        *,
        grounding_status: GroundingStatus = GroundingStatus.SUPPORTED,
        grounding_reason: str = "Grounded in evidence.",
    ) -> None:
        self._grounding_status = grounding_status
        self._grounding_reason = grounding_reason

    def rewrite_query(self, query: str) -> tuple[QueryRewriteOutput, str]:
        output = QueryRewriteOutput(rewritten_query=f"{query} in detail?", prompt_version="v1.0.0")
        return output, "llm"

    def rewrite_for_retry(
        self,
        *,
        original_query: str,
        retry_reason: str,
        evidence: list[str],
    ) -> tuple[QueryRewriteOutput, str]:
        del retry_reason, evidence
        output = QueryRewriteOutput(
            rewritten_query=f"{original_query} Focus on bert.pdf?",
            prompt_version="v1.0.0",
        )
        return output, "llm"

    def assess_grounding(
        self,
        *,
        answer: str,
        citations: list[str],
        evidence: list[str],
    ) -> tuple[GroundingResult, str, str | None]:
        del answer, citations, evidence
        return (
            GroundingResult(status=self._grounding_status, reason=self._grounding_reason),
            "llm",
            "v1.0.0",
        )


class AmbiguousTools:
    def search_chunks(self, query: str, top_k: int) -> list[EvidenceChunk]:
        del top_k
        return [
            EvidenceChunk(
                chunk_id="a-0001",
                source="A.pdf",
                text=f"A evidence for {query}",
                score=0.5,
            ),
            EvidenceChunk(
                chunk_id="b-0001",
                source="B.pdf",
                text=f"B evidence for {query}",
                score=0.49,
            ),
        ]

    def fetch_chunks_by_ids(self, chunk_ids: list[str]) -> list[EvidenceChunk]:
        del chunk_ids
        return []


def _settings() -> Settings:
    return Settings(
        documents_dir=Path("documents"),
        retrieval_top_k=3,
        min_relevance_score=0.1,
        ambiguity_margin=0.03,
        max_retry_count=1,
    )


def test_safe_fail_path() -> None:
    pipeline = AgenticPipeline(settings=_settings(), tools=FakeTools(), trace_store=TraceStore())
    response = pipeline.ask("What quantum method does BERT use?")
    assert response.safe_fail is True


def test_supported_answer_path() -> None:
    pipeline = AgenticPipeline(settings=_settings(), tools=FakeTools(), trace_store=TraceStore())
    response = pipeline.ask("What is BERT pretraining?")
    assert response.safe_fail is False
    assert len(response.citations) > 0


def test_reasoner_rewrite_is_used_and_traced() -> None:
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=FakeReasoner(),
    )
    response = pipeline.ask("What is BERT")
    trace = trace_store.get(response.trace_id)
    assert trace is not None
    assert trace.rewritten_query == "What is BERT in detail?"
    understand_events = [event for event in trace.events if event.stage == "understand"]
    assert len(understand_events) == 1
    assert understand_events[0].payload["rewrite_source"] == "llm"


def test_vague_query_returns_clarification() -> None:
    trace_store = TraceStore()
    pipeline = AgenticPipeline(settings=_settings(), tools=FakeTools(), trace_store=trace_store)
    response = pipeline.ask("hi")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    assert response.safe_fail is True
    assert "specific question" in response.answer.lower()
    stages = [event.stage for event in trace.events]
    assert "clarify" in stages
    assert "retrieve" not in stages


def test_ambiguous_after_retry_budget_safe_fails() -> None:
    settings = Settings(
        documents_dir=Path("documents"),
        retrieval_top_k=3,
        min_relevance_score=0.1,
        ambiguity_margin=0.05,
        max_retry_count=1,
    )
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=settings,
        tools=AmbiguousTools(),
        trace_store=trace_store,
        reasoner=FakeReasoner(),
    )
    response = pipeline.ask("What is the best source?")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    assert response.safe_fail is True
    assert response.answer == settings.safe_fail_message
    stages = [event.stage for event in trace.events]
    assert "retry" in stages
    assert "retry_exhausted" in stages


def test_retry_uses_reasoner_rewrite_and_trace_source() -> None:
    settings = Settings(
        documents_dir=Path("documents"),
        retrieval_top_k=3,
        min_relevance_score=0.1,
        ambiguity_margin=0.05,
        max_retry_count=1,
    )
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=settings,
        tools=AmbiguousTools(),
        trace_store=trace_store,
        reasoner=FakeReasoner(),
    )
    response = pipeline.ask("What is the best source?")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    retry_events = [event for event in trace.events if event.stage == "retry"]
    assert len(retry_events) == 1
    assert retry_events[0].payload["rewrite_source"] == "llm"
    assert trace.rewritten_query.endswith("bert.pdf?")


def test_reasoner_grounding_unsupported_forces_safe_fail() -> None:
    settings = _settings()
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=settings,
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=FakeReasoner(
            grounding_status=GroundingStatus.UNSUPPORTED,
            grounding_reason="Answer is not supported by provided evidence.",
        ),
    )
    response = pipeline.ask("What is BERT pretraining?")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    assert response.safe_fail is True
    assert response.answer == settings.safe_fail_message
    assert response.citations == []

    verify_events = [event for event in trace.events if event.stage == "verify_grounding"]
    assert len(verify_events) == 1
    assert verify_events[0].payload["status"] == GroundingStatus.UNSUPPORTED
    assert verify_events[0].payload["grounding_source"] == "llm"


def test_reasoner_grounding_supported_keeps_answer() -> None:
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=FakeReasoner(),
    )
    response = pipeline.ask("What is BERT pretraining?")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    assert response.safe_fail is False
    assert len(response.citations) > 0

    verify_events = [event for event in trace.events if event.stage == "verify_grounding"]
    assert len(verify_events) == 1
    assert verify_events[0].payload["status"] == GroundingStatus.SUPPORTED
    assert verify_events[0].payload["grounding_source"] == "llm"
