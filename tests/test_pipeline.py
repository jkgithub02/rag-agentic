from __future__ import annotations

from pathlib import Path

from src.core.config import Settings
from src.core.models import (
    EvidenceChunk,
    GroundingResult,
    GroundingStatus,
    QueryRewriteOutput,
    ResponseCategory,
)
from src.orchestration.pipeline import AgenticPipeline
from src.services.trace_store import TraceStore


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
        synthesis_answer: str = "BERT pretraining uses masked language modeling.",
        synthesis_chunk_ids: list[str] | None = None,
    ) -> None:
        self._grounding_status = grounding_status
        self._grounding_reason = grounding_reason
        self._synthesis_answer = synthesis_answer
        self._synthesis_chunk_ids = synthesis_chunk_ids or ["bert-0001"]

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

    def synthesize_answer(
        self,
        *,
        query: str,
        chunks: list[EvidenceChunk],
    ) -> tuple[str, list[str], str, str | None]:
        del query, chunks
        return self._synthesis_answer, self._synthesis_chunk_ids, "llm", "v1.0.0"


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


class HydratingTools:
    def __init__(self, *, return_fetched: bool) -> None:
        self.return_fetched = return_fetched
        self.fetch_calls: list[list[str]] = []

    def search_chunks(self, query: str, top_k: int) -> list[EvidenceChunk]:
        del query, top_k
        return [
            EvidenceChunk(
                chunk_id="bert-0001",
                source="bert.pdf",
                text="Search hit one.",
                score=0.8,
            ),
            EvidenceChunk(
                chunk_id="bert-0002",
                source="bert.pdf",
                text="Search hit two.",
                score=0.7,
            ),
        ]

    def fetch_chunks_by_ids(self, chunk_ids: list[str]) -> list[EvidenceChunk]:
        self.fetch_calls.append(list(chunk_ids))
        if not self.return_fetched:
            return []
        return [
            EvidenceChunk(
                chunk_id="bert-0001",
                source="bert.pdf",
                text="Hydrated evidence content.",
                score=1.0,
            )
        ]


class FakeResponsePolicy:
    def __init__(self) -> None:
        self.calls: list[tuple[ResponseCategory, str, str | None, int]] = []

    def render(
        self,
        *,
        category: ResponseCategory,
        query: str,
        reason: str | None = None,
        evidence_count: int = 0,
    ) -> tuple[str, str]:
        self.calls.append((category, query, reason, evidence_count))
        return f"Naturalized[{category.value}] for: {query}", "v1.0.0"


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


def test_reasoner_generation_is_used_and_traced() -> None:
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=FakeReasoner(
            synthesis_answer="BERT pretraining includes MLM and NSP.",
            synthesis_chunk_ids=["bert-0001"],
        ),
    )
    response = pipeline.ask("What is BERT pretraining?")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    assert "MLM and NSP" in response.answer
    assert response.citations == ["bert.pdf#bert-0001"]

    generate_events = [event for event in trace.events if event.stage == "generate"]
    assert len(generate_events) == 1
    assert generate_events[0].payload["generation_source"] == "llm"


def test_reasoner_generation_unknown_chunk_ids_fallback_to_top_chunks() -> None:
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=FakeReasoner(synthesis_chunk_ids=["missing-id"]),
    )
    response = pipeline.ask("What is BERT pretraining?")

    assert response.citations == ["bert.pdf#bert-0001"]


def test_hydrate_tool_is_called_on_supported_path() -> None:
    trace_store = TraceStore()
    tools = HydratingTools(return_fetched=True)
    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=tools,
        trace_store=trace_store,
        reasoner=FakeReasoner(),
    )

    response = pipeline.ask("What is BERT pretraining?")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    assert tools.fetch_calls == [["bert-0001", "bert-0002"]]
    stages = [event.stage for event in trace.events]
    assert "tool_fetch_chunks_by_ids" in stages
    assert "hydrate" in stages


def test_hydrate_fallback_uses_retrieved_chunks_when_fetch_empty() -> None:
    trace_store = TraceStore()
    tools = HydratingTools(return_fetched=False)
    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=tools,
        trace_store=trace_store,
        reasoner=FakeReasoner(
            synthesis_answer="Using fallback retrieval chunks.",
            synthesis_chunk_ids=["bert-0002"],
        ),
    )

    response = pipeline.ask("What is BERT pretraining?")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    assert response.citations == ["bert.pdf#bert-0002"]
    hydrate_events = [event for event in trace.events if event.stage == "hydrate"]
    assert len(hydrate_events) == 1
    assert hydrate_events[0].payload["used_fallback"] is True


def test_response_policy_used_for_clarification() -> None:
    trace_store = TraceStore()
    policy = FakeResponsePolicy()
    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=FakeTools(),
        trace_store=trace_store,
        response_policy=policy,
    )

    response = pipeline.ask("hi")

    assert response.safe_fail is True
    assert response.answer == "Naturalized[clarification] for: hi"
    assert policy.calls[0][0] == ResponseCategory.CLARIFICATION


def test_response_policy_used_for_retry_exhausted_safe_fail() -> None:
    settings = Settings(
        documents_dir=Path("documents"),
        retrieval_top_k=3,
        min_relevance_score=0.1,
        ambiguity_margin=0.05,
        max_retry_count=1,
    )
    trace_store = TraceStore()
    policy = FakeResponsePolicy()
    pipeline = AgenticPipeline(
        settings=settings,
        tools=AmbiguousTools(),
        trace_store=trace_store,
        reasoner=FakeReasoner(),
        response_policy=policy,
    )

    response = pipeline.ask("What is the best source?")

    assert response.safe_fail is True
    assert response.answer == "Naturalized[retry_exhausted] for: What is the best source?"
    assert any(call[0] == ResponseCategory.RETRY_EXHAUSTED for call in policy.calls)


def test_response_policy_naturalizes_grounding_reason() -> None:
    trace_store = TraceStore()
    policy = FakeResponsePolicy()
    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=FakeReasoner(),
        response_policy=policy,
    )
    response = pipeline.ask("What is BERT pretraining?")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    verify_events = [event for event in trace.events if event.stage == "verify_grounding"]
    assert len(verify_events) == 1
    assert verify_events[0].payload["reason_source"] == "llm-policy"
    assert any(call[0] == ResponseCategory.GROUNDING_REASON for call in policy.calls)
