from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from src.core.config import Settings
from src.core.models import (
    EvidenceChunk,
    GroundingResult,
    GroundingStatus,
    PipelineTrace,
    QueryAnalysisOutput,
)
from src.orchestration.pipeline import AgenticPipeline
from src.services.llm_client import LLMInvocationError
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
        if "bert-0001" in chunk_ids:
            return [
                EvidenceChunk(
                    chunk_id="bert-0001",
                    source="bert.pdf",
                    text="BERT uses masked language modeling.",
                    score=0.9,
                )
            ]
        return []


class FakeReasoner:
    def __init__(
        self,
        *,
        grounding_status: GroundingStatus = GroundingStatus.SUPPORTED,
        grounding_reason: str = "Grounded in evidence.",
        synthesis_answer: str = "BERT pretraining uses masked language modeling.",
        synthesis_chunk_ids: list[str] | None = None,
        coverage_insufficient: bool = False,
        coverage_missing_terms: list[str] | None = None,
    ) -> None:
        self._grounding_status = grounding_status
        self._grounding_reason = grounding_reason
        self._synthesis_answer = synthesis_answer
        self._synthesis_chunk_ids = synthesis_chunk_ids or ["bert-0001"]
        self._coverage_insufficient = coverage_insufficient
        self._coverage_missing_terms = coverage_missing_terms or []

    def summarize_conversation(self, history: list[dict[str, str]]) -> str:
        if not history:
            return ""
        last_user = next((item["content"] for item in reversed(history) if item["role"] == "user"), "")
        return f"Conversation summary: prior user intent around '{last_user}'."

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

    def detect_insufficient_coverage(
        self,
        *,
        query: str,
        answer: str,
        chunks: list[EvidenceChunk],
    ) -> tuple[bool, list[str], str, str | None]:
        del query, answer, chunks
        return self._coverage_insufficient, self._coverage_missing_terms, "llm", "v1.0.0"

    def assess_query_clarity(
        self,
        *,
        query: str,
        conversation_summary: str | None = None,
    ) -> tuple[bool, str, str | None]:
        del conversation_summary
        return (query.strip().lower() in {"hi", "hello", "hey"}, "clarity check", "v1.0.0")

    def analyze_query(
        self,
        *,
        query: str,
        conversation_summary: str | None = None,
    ) -> tuple[QueryAnalysisOutput, str]:
        rewritten_query = f"{query} in detail?"
        clarify_needed, reason, prompt_version = self.assess_query_clarity(
            query=query,
            conversation_summary=conversation_summary,
        )
        if clarify_needed:
            return (
                QueryAnalysisOutput(
                    is_clear=False,
                    rewritten_query=None,
                    clarification_needed=reason,
                    prompt_version=prompt_version,
                ),
                "llm",
            )
        return (
            QueryAnalysisOutput(
                is_clear=True,
                rewritten_query=rewritten_query,
                clarification_needed=None,
                prompt_version="v1.0.0",
            ),
            "llm",
        )


class SummaryAwareReasoner(FakeReasoner):
    def summarize_conversation(self, history: list[dict[str, str]]) -> str:
        if not history:
            return ""
        lines = [f"{item['role']}: {item['content']}" for item in history]
        return "\n".join(lines)

    def assess_query_clarity(
        self,
        *,
        query: str,
        conversation_summary: str | None = None,
    ) -> tuple[bool, str, str | None]:
        lowered = query.strip().lower()
        if lowered in {"the nlp model", "that model", "it"} and conversation_summary:
            if "bert" in conversation_summary.lower():
                return (False, "resolved by conversation summary", "v1.0.0")
        return super().assess_query_clarity(query=query, conversation_summary=conversation_summary)


class AlwaysUnclearReasoner(FakeReasoner):
    def assess_query_clarity(
        self,
        *,
        query: str,
        conversation_summary: str | None = None,
    ) -> tuple[bool, str, str | None]:
        del query, conversation_summary
        return True, "force clarification", "v1.0.0"


class ConfirmationAwareReasoner(FakeReasoner):
    def analyze_query(
        self,
        *,
        query: str,
        conversation_summary: str | None = None,
    ) -> tuple[QueryAnalysisOutput, str]:
        lowered = query.strip().lower()
        summary = (conversation_summary or "").lower()

        if lowered.startswith("who is "):
            return (
                QueryAnalysisOutput(
                    is_clear=False,
                    rewritten_query=None,
                    clarification_needed="Do you want me to search available documents?",
                    prompt_version="v1.0.0",
                ),
                "llm",
            )

        if "yes" in lowered and "available documents" in lowered and "who is jason kong" in summary:
            return (
                QueryAnalysisOutput(
                    is_clear=True,
                    rewritten_query="Who is Jason Kong in available documents?",
                    clarification_needed=None,
                    prompt_version="v1.0.0",
                ),
                "llm",
            )

        return super().analyze_query(query=query, conversation_summary=conversation_summary)


class BrokenSynthesisReasoner(FakeReasoner):
    def synthesize_answer(
        self,
        *,
        query: str,
        chunks: list[EvidenceChunk],
    ) -> tuple[str, list[str], str, str | None]:
        del query, chunks
        raise LLMInvocationError("synthesize_answer produced invalid citation ids.")


class BrokenAnalyzeReasoner(FakeReasoner):
    def analyze_query(
        self,
        *,
        query: str,
        conversation_summary: str | None = None,
    ) -> tuple[QueryAnalysisOutput, str]:
        del query, conversation_summary
        raise LLMInvocationError("analyze_query returned non-JSON payload")


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


class SameSourceCloseScoreTools:
    def search_chunks(self, query: str, top_k: int) -> list[EvidenceChunk]:
        del query, top_k
        return [
            EvidenceChunk(
                chunk_id="resume-0001",
                source="resume.pdf",
                text="Jason Kong is a software engineer.",
                score=0.51,
            ),
            EvidenceChunk(
                chunk_id="resume-0002",
                source="resume.pdf",
                text="Experience includes AI projects.",
                score=0.50,
            ),
        ]

    def fetch_chunks_by_ids(self, chunk_ids: list[str]) -> list[EvidenceChunk]:
        records = {
            "resume-0001": EvidenceChunk(
                chunk_id="resume-0001",
                source="resume.pdf",
                text="Jason Kong is a software engineer.",
                score=0.9,
            ),
            "resume-0002": EvidenceChunk(
                chunk_id="resume-0002",
                source="resume.pdf",
                text="Experience includes AI projects.",
                score=0.88,
            ),
        }
        return [records[item] for item in chunk_ids if item in records]


class ExplicitSourceMixedTools:
    def search_chunks(self, query: str, top_k: int) -> list[EvidenceChunk]:
        del query, top_k
        return [
            EvidenceChunk(
                chunk_id="a-0001",
                source="A.pdf",
                text="A document evidence.",
                score=0.51,
            ),
            EvidenceChunk(
                chunk_id="b-0001",
                source="B.pdf",
                text="B document evidence.",
                score=0.50,
            ),
        ]

    def fetch_chunks_by_ids(self, chunk_ids: list[str]) -> list[EvidenceChunk]:
        records = {
            "a-0001": EvidenceChunk(
                chunk_id="a-0001",
                source="A.pdf",
                text="A document evidence.",
                score=0.9,
            ),
            "b-0001": EvidenceChunk(
                chunk_id="b-0001",
                source="B.pdf",
                text="B document evidence.",
                score=0.89,
            ),
        }
        return [records[item] for item in chunk_ids if item in records]


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


class MismatchTools:
    def search_chunks(self, query: str, top_k: int) -> list[EvidenceChunk]:
        del query, top_k
        return [
            EvidenceChunk(
                chunk_id="bert-0001",
                source="bert.pdf",
                text="BERT uses masked language modeling.",
                score=0.8,
            )
        ]

    def fetch_chunks_by_ids(self, chunk_ids: list[str]) -> list[EvidenceChunk]:
        if "bert-0001" in chunk_ids:
            return [
                EvidenceChunk(
                    chunk_id="bert-0001",
                    source="bert.pdf",
                    text="BERT uses masked language modeling.",
                    score=0.9,
                )
            ]
        return []


def _settings() -> Settings:
    return Settings(
        documents_dir=Path("documents"),
        retrieval_top_k=3,
        min_relevance_score=0.1,
        ambiguity_margin=0.03,
    )


def test_safe_fail_path() -> None:
    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=FakeTools(),
        trace_store=TraceStore(),
        reasoner=FakeReasoner(),
    )
    response = pipeline.ask("What quantum method does BERT use?")
    assert response.safe_fail is True


def test_supported_answer_path() -> None:
    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=FakeTools(),
        trace_store=TraceStore(),
        reasoner=FakeReasoner(),
    )
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
    rewrite_events = [event for event in trace.events if event.stage == "rewrite_query"]
    assert len(rewrite_events) == 1
    assert rewrite_events[0].payload["rewrite_source"] == "llm"


def test_vague_query_returns_clarification() -> None:
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=FakeReasoner(),
    )
    response = pipeline.ask("hi")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    assert response.safe_fail is True
    assert response.answer == "clarity check"
    stages = [event.stage for event in trace.events]
    assert "clarify" in stages
    assert "retrieve" not in stages


def test_analyze_query_failure_falls_back_to_clarification() -> None:
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=BrokenAnalyzeReasoner(),
    )

    response = pipeline.ask("Summarise the resume")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    assert response.safe_fail is True
    assert response.answer == _settings().clarification_message
    rewrite_events = [event for event in trace.events if event.stage == "rewrite_query"]
    assert len(rewrite_events) == 1
    assert rewrite_events[0].payload["rewrite_source"] == "fallback-rule"
    assert rewrite_events[0].payload["analysis_error"] is not None


def test_concrete_intent_still_clarifies_when_model_marks_unclear() -> None:
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=AlwaysUnclearReasoner(),
    )

    response = pipeline.ask("summarise the resume")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    stages = [event.stage for event in trace.events]
    assert "clarify" in stages
    assert "retrieve" not in stages


def test_explicit_source_query_still_clarifies_when_model_marks_unclear() -> None:
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=AlwaysUnclearReasoner(),
    )

    response = pipeline.ask("Summarise resume.pdf")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    stages = [event.stage for event in trace.events]
    assert "clarify" in stages
    assert "retrieve" not in stages


def test_clarification_uses_model_message() -> None:
    settings = _settings()
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=settings,
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=AlwaysUnclearReasoner(),
    )

    response = pipeline.ask("hi")

    assert response.safe_fail is True
    assert response.answer == "force clarification"


def test_followup_query_uses_thread_context_summary() -> None:
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=SummaryAwareReasoner(),
    )

    first = pipeline.ask("Tell me about BERT", thread_id="session-followup")
    assert first.safe_fail is False

    followup = pipeline.ask("the NLP model", thread_id="session-followup")
    followup_trace = trace_store.get(followup.trace_id)

    assert followup_trace is not None
    rewrite_events = [event for event in followup_trace.events if event.stage == "rewrite_query"]
    assert len(rewrite_events) == 1
    assert rewrite_events[0].payload["clarify_needed"] is False


def test_confirmation_followup_uses_prior_user_context_without_reclarifying(tmp_path: Path) -> None:
    settings = Settings(
        documents_dir=Path("documents"),
        vector_db_path=tmp_path / "qdrant",
        retrieval_top_k=3,
        min_relevance_score=0.1,
        ambiguity_margin=0.03,
    )
    trace_store = TraceStore(storage_dir=tmp_path / "traces")
    pipeline = AgenticPipeline(
        settings=settings,
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=ConfirmationAwareReasoner(),
    )

    thread_id = "session-confirmation"
    first = pipeline.ask("Who is Jason Kong", thread_id=thread_id)
    first_trace = trace_store.get(first.trace_id)

    assert first_trace is not None
    assert first.safe_fail is True
    assert pipeline._load_thread_history(thread_id) == [
        {"role": "user", "content": "Who is Jason Kong"}
    ]

    second = pipeline.ask("yes, in the available documents", thread_id=thread_id)
    second_trace = trace_store.get(second.trace_id)

    assert second_trace is not None
    assert second.safe_fail is False
    second_stages = [event.stage for event in second_trace.events]
    assert "clarify" not in second_stages
    assert "retrieve" in second_stages


def test_ambiguous_results_fail_without_retry() -> None:
    settings = Settings(
        documents_dir=Path("documents"),
        retrieval_top_k=3,
        min_relevance_score=0.1,
        ambiguity_margin=0.05,
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
    assert "retry" not in stages


def test_close_scores_same_source_do_not_trigger_ambiguity_retry() -> None:

    settings = Settings(
        documents_dir=Path("documents"),
        retrieval_top_k=3,
        min_relevance_score=0.1,
        ambiguity_margin=0.05,
    )
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=settings,
        tools=SameSourceCloseScoreTools(),
        trace_store=trace_store,
        reasoner=FakeReasoner(synthesis_chunk_ids=["resume-0001"]),
    )

    response = pipeline.ask("summarise the resume")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    assert response.safe_fail is False
    stages = [event.stage for event in trace.events]
    assert "retry" not in stages


def test_explicit_source_query_uses_model_analysis_when_query_is_clear() -> None:
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=FakeReasoner(),
    )

    response = pipeline.ask("Summarise Resume_Jason_Kong.pdf")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    assert response.safe_fail is False
    rewrite_events = [event for event in trace.events if event.stage == "rewrite_query"]
    assert len(rewrite_events) == 1
    assert rewrite_events[0].payload["rewrite_source"] == "llm"
    assert "resume_jason_kong.pdf" in rewrite_events[0].payload["rewritten"].lower()
    stages = [event.stage for event in trace.events]
    assert "clarify" not in stages
    assert "retrieve" in stages


def test_named_resume_query_uses_model_analysis_for_routing() -> None:
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=FakeReasoner(),
    )

    response = pipeline.ask("Summarise Jason Kong's resume")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    assert response.safe_fail is False
    rewrite_events = [event for event in trace.events if event.stage == "rewrite_query"]
    assert len(rewrite_events) == 1
    assert rewrite_events[0].payload["rewrite_source"] == "llm"
    assert "jason kong" in rewrite_events[0].payload["rewritten"].lower()
    stages = [event.stage for event in trace.events]
    assert "clarify" not in stages
    assert "retrieve" in stages


def test_search_across_docs_followup_bypasses_clarification() -> None:
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=FakeReasoner(),
    )

    pipeline.ask("Summarise Jason Kong's resume", thread_id="session-search-intent")
    followup = pipeline.ask("search across all docs", thread_id="session-search-intent")
    trace = trace_store.get(followup.trace_id)

    assert trace is not None
    rewrite_events = [event for event in trace.events if event.stage == "rewrite_query"]
    assert len(rewrite_events) == 1
    assert rewrite_events[0].payload["rewrite_source"] == "llm"
    assert "search across all docs" in rewrite_events[0].payload["rewritten"].lower()
    stages = [event.stage for event in trace.events]
    assert "clarify" not in stages
    assert "retrieve" in stages


def test_ambiguous_top_two_sources_fail_without_retry() -> None:
    settings = Settings(
        documents_dir=Path("documents"),
        retrieval_top_k=3,
        min_relevance_score=0.1,
        ambiguity_margin=0.05,
    )
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=settings,
        tools=ExplicitSourceMixedTools(),
        trace_store=trace_store,
        reasoner=FakeReasoner(synthesis_chunk_ids=["a-0001"]),
    )

    response = pipeline.ask("Summarise A.pdf")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    assert response.safe_fail is True
    assert response.citations == []
    stages = [event.stage for event in trace.events]
    assert "retry" not in stages


def test_ambiguous_flow_does_not_emit_retry_events() -> None:
    settings = Settings(
        documents_dir=Path("documents"),
        retrieval_top_k=3,
        min_relevance_score=0.1,
        ambiguity_margin=0.05,
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
    assert len(retry_events) == 0


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


def test_reasoner_generation_unknown_chunk_ids_safe_fails() -> None:
    settings = _settings()
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=settings,
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=FakeReasoner(synthesis_chunk_ids=["missing-id"]),
    )
    response = pipeline.ask("What is BERT pretraining?")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    assert response.safe_fail is True
    assert response.answer == settings.safe_fail_message
    assert response.citations == []


def test_reasoner_synthesis_exception_safe_fails() -> None:
    settings = _settings()
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=settings,
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=BrokenSynthesisReasoner(),
    )

    response = pipeline.ask("what about Jason Wong?")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    assert response.safe_fail is True
    assert response.answer == settings.safe_fail_message
    assert response.citations == []


def test_conversation_recall_query_is_model_routed_not_rule_shortcut() -> None:
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=SummaryAwareReasoner(),
    )
    thread_id = f"session-recall-{uuid4().hex}"

    first = pipeline.ask("summarise the resume", thread_id=thread_id)
    assert first.safe_fail is False

    recall = pipeline.ask("what were we talking about", thread_id=thread_id)
    trace = trace_store.get(recall.trace_id)

    assert trace is not None
    assert recall.safe_fail is False
    stages = [event.stage for event in trace.events]
    assert "retrieve" in stages
    assert "clarify" not in stages


def test_thread_history_persists_across_pipeline_instances(tmp_path: Path) -> None:
    settings = Settings(
        documents_dir=Path("documents"),
        vector_db_path=tmp_path / "qdrant",
    )

    first_pipeline = AgenticPipeline(
        settings=settings,
        tools=FakeTools(),
        trace_store=TraceStore(storage_dir=tmp_path / "traces-a"),
        reasoner=SummaryAwareReasoner(),
    )
    first_pipeline.ask("summarise the resume", thread_id="session-persist")

    second_trace_store = TraceStore(storage_dir=tmp_path / "traces-b")
    second_pipeline = AgenticPipeline(
        settings=settings,
        tools=FakeTools(),
        trace_store=second_trace_store,
        reasoner=SummaryAwareReasoner(),
    )
    recall = second_pipeline.ask("what were we talking about", thread_id="session-persist")
    trace = second_trace_store.get(recall.trace_id)

    assert trace is not None
    assert recall.safe_fail is False
    rewrite_events = [event for event in trace.events if event.stage == "rewrite_query"]
    assert len(rewrite_events) == 1
    assert "summarise the resume" in rewrite_events[0].payload["conversation_summary"].lower()


def test_fetch_chunks_by_ids_is_used_in_reference_retrieval_flow() -> None:
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
    assert len(tools.fetch_calls) == 1
    assert tools.fetch_calls[0] == ["bert-0001", "bert-0002"]
    stages = [event.stage for event in trace.events]
    assert "tool_fetch_chunks_by_ids" in stages
    assert "hydrate" not in stages


def test_missing_fetch_step_does_not_break_generation() -> None:
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
    assert response.safe_fail is False
    assert all(event.stage != "hydrate_fallback" for event in trace.events)


def test_no_hit_safe_fail_uses_deterministic_message() -> None:
    settings = _settings()
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=settings,
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=FakeReasoner(),
    )

    response = pipeline.ask("What quantum method does BERT use?")

    assert response.safe_fail is True
    assert response.answer == settings.safe_fail_message
def test_unsupported_grounding_uses_default_safe_fail_message() -> None:
    settings = _settings()
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=settings,
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=FakeReasoner(
            grounding_status=GroundingStatus.UNSUPPORTED,
            grounding_reason="Unsupported by evidence.",
        ),
    )

    response = pipeline.ask("What is BERT pretraining?")

    assert response.safe_fail is True
    assert response.answer == settings.safe_fail_message


def test_query_coverage_guard_forces_unsupported() -> None:
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=MismatchTools(),
        trace_store=trace_store,
        reasoner=FakeReasoner(
            grounding_status=GroundingStatus.SUPPORTED,
            grounding_reason="Looks supported.",
            synthesis_answer="No quantum method is described for BERT.",
            synthesis_chunk_ids=["bert-0001"],
            coverage_insufficient=True,
            coverage_missing_terms=["quantum"],
        ),
    )

    response = pipeline.ask("What quantum method does BERT use?")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    assert response.safe_fail is True
    assert response.citations == []
    verify_events = [event for event in trace.events if event.stage == "verify_grounding"]
    assert len(verify_events) == 1
    assert verify_events[0].payload["grounding_source"] == "llm-insufficient-coverage"


def test_summary_query_with_sparse_single_source_evidence_still_honors_coverage_gate() -> None:
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=FakeReasoner(
            grounding_status=GroundingStatus.SUPPORTED,
            grounding_reason="Looks supported.",
            coverage_insufficient=True,
            coverage_missing_terms=["experience"],
        ),
    )

    response = pipeline.ask("Summarise the resume")

    assert response.safe_fail is True
    assert response.citations == []


def test_pipeline_invokes_graph_with_thread_id_config(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeGraph:
        def __init__(self) -> None:
            self.invocations: list[tuple[dict[str, object], dict[str, object]]] = []

        def invoke(
            self,
            payload: dict[str, object],
            config: dict[str, object],
        ) -> dict[str, object]:
            self.invocations.append((payload, config))
            trace = PipelineTrace(original_query="What is BERT?", rewritten_query="What is BERT?")
            return {
                "answer": "BERT pretraining uses masked language modeling.",
                "citations": ["bert.pdf#bert-0001"],
                "safe_fail": False,
                "trace": trace,
            }

    fake_graph = FakeGraph()
    monkeypatch.setattr(
        "src.orchestration.pipeline.build_pipeline_graph",
        lambda *, nodes, edges: fake_graph,
    )

    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=FakeTools(),
        trace_store=TraceStore(),
        reasoner=FakeReasoner(),
    )

    response = pipeline.ask("What is BERT?")

    assert response.safe_fail is False
    assert len(fake_graph.invocations) == 1
    payload, config = fake_graph.invocations[0]
    assert payload == {"query": "What is BERT?", "history": []}
    assert "configurable" in config
    assert isinstance(config["configurable"].get("thread_id"), str)
    assert config["configurable"]["thread_id"]


def test_pipeline_uses_provided_thread_id(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeGraph:
        def __init__(self) -> None:
            self.invocations: list[tuple[dict[str, object], dict[str, object]]] = []

        def invoke(
            self,
            payload: dict[str, object],
            config: dict[str, object],
        ) -> dict[str, object]:
            self.invocations.append((payload, config))
            trace = PipelineTrace(original_query="What is BERT?", rewritten_query="What is BERT?")
            return {
                "answer": "BERT pretraining uses masked language modeling.",
                "citations": ["bert.pdf#bert-0001"],
                "safe_fail": False,
                "trace": trace,
            }

    fake_graph = FakeGraph()
    monkeypatch.setattr(
        "src.orchestration.pipeline.build_pipeline_graph",
        lambda *, nodes, edges: fake_graph,
    )

    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=FakeTools(),
        trace_store=TraceStore(),
        reasoner=FakeReasoner(),
    )

    pipeline.ask("What is BERT?", thread_id="session-123")

    _, config = fake_graph.invocations[0]
    assert config["configurable"]["thread_id"] == "session-123"
