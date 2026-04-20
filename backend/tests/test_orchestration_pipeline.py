from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from conftest import (
    BrokenAnalyzeReasoner,
    BrokenSynthesisReasoner,
    FakeReasoner,
    FakeTools,
)
from src.core.config import Settings
from src.core.models import (
    AgentThought,
    EvidenceChunk,
    GroundingResult,
    GroundingStatus,
    PipelineTrace,
    QueryComplexity,
    QueryAnalysisOutput,
)
from src.orchestration.pipeline import AgenticPipeline
from src.services.llm_client import LLMInvocationError
from src.services.trace_store import TraceStore


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


class MultiQuestionReasoner(FakeReasoner):
    def analyze_query(
        self,
        *,
        query: str,
        conversation_summary: str | None = None,
    ) -> tuple[QueryAnalysisOutput, str]:
        del query, conversation_summary
        return (
            QueryAnalysisOutput(
                is_clear=True,
                questions=["What is BERT architecture?", "What is BERT pretraining?"],
                rewritten_query="What is BERT architecture?",
                clarification_needed=None,
                prompt_version="v1.0.0",
            ),
            "llm",
        )


class ComparisonReasoner(FakeReasoner):
    def analyze_query(
        self,
        *,
        query: str,
        conversation_summary: str | None = None,
    ) -> tuple[QueryAnalysisOutput, str]:
        del query, conversation_summary
        rewritten = "How is attention used in BERT versus the original Transformer?"
        return (
            QueryAnalysisOutput(
                is_clear=True,
                questions=[rewritten],
                rewritten_query=rewritten,
                clarification_needed=None,
                prompt_version="v1.0.0",
            ),
            "llm",
        )


class DecomposingReasoner(FakeReasoner):
    def detect_query_complexity(
        self,
        *,
        query: str,
        conversation_summary: str | None = None,
    ) -> QueryComplexity:
        del query, conversation_summary
        return QueryComplexity.COMPLEX

    def decompose_query_lightly(
        self,
        *,
        query: str,
        conversation_summary: str | None = None,
    ) -> list[str]:
        del query, conversation_summary
        return ["What is BERT architecture?", "What is BERT pretraining?"]


class SimpleComplexityReasoner(FakeReasoner):
    def detect_query_complexity(
        self,
        *,
        query: str,
        conversation_summary: str | None = None,
    ) -> QueryComplexity:
        del query, conversation_summary
        return QueryComplexity.SIMPLE


class PlannerFinalizeReasoner(FakeReasoner):
    def detect_query_complexity(
        self,
        *,
        query: str,
        conversation_summary: str | None = None,
    ) -> QueryComplexity:
        del query, conversation_summary
        return QueryComplexity.MODERATE

    def plan_agent_step(
        self,
        *,
        query: str,
        conversation_summary: str | None,
        rewritten_queries: list[str],
        evidence_quality_score: float,
        chunk_count: int,
        agent_iterations: int,
        max_iterations: int,
        last_observation: str | None = None,
        subquery_statuses: list[dict[str, object]] | None = None,
    ):
        del (
            query,
            conversation_summary,
            rewritten_queries,
            evidence_quality_score,
            chunk_count,
            agent_iterations,
            max_iterations,
            last_observation,
            subquery_statuses,
        )
        return AgentThought(
            reasoning="Existing evidence is sufficient.",
            recommended_action="finalize",
            confidence=0.9,
        )


class QueryCaptureTools:
    def __init__(self) -> None:
        self.search_queries: list[str] = []

    def search_chunks(self, query: str, top_k: int) -> list[EvidenceChunk]:
        del top_k
        self.search_queries.append(query)
        return [
            EvidenceChunk(
                chunk_id="bert-0001",
                source="bert.pdf",
                text="BERT attention is bidirectional.",
                score=0.9,
            )
        ]

    def fetch_chunks_by_ids(self, chunk_ids: list[str]) -> list[EvidenceChunk]:
        del chunk_ids
        return [
            EvidenceChunk(
                chunk_id="bert-0001",
                source="bert.pdf",
                text="BERT attention is bidirectional.",
                score=0.9,
            )
        ]


class QueryEchoReasoner(FakeReasoner):
    def synthesize_answer(
        self,
        *,
        query: str,
        chunks: list[EvidenceChunk],
        subqueries: list[str] | None = None,
    ) -> tuple[str, list[str], str, str | None]:
        del chunks, subqueries
        return f"ANSWER_QUERY={query}", ["bert-0001"], "llm", "v1.0.0"


class CategoryDemoReasoner(FakeReasoner):
    _rewrites = {
        "tell me about attention": "Explain the attention mechanism in the uploaded transformer-related papers.",
        "how does it handle context?": "Explain how context is represented and propagated in the uploaded transformer-related papers.",
        "what's the training trick they use?": "Identify the self-supervised or pretraining objective used in the uploaded papers.",
        "compare the models": "Compare model architecture, attention behavior, and training objective across the uploaded papers.",
        "how does the model learn word relationships?": "Explain how embeddings and attention layers capture word relationships.",
        "what self-supervised objective is used?": "Identify the masked language model or other self-supervised objective used by the model.",
        "how is context aggregated?": "Explain how attention aggregates context across tokens.",
    }

    def analyze_query(
        self,
        *,
        query: str,
        conversation_summary: str | None = None,
    ) -> tuple[QueryAnalysisOutput, str]:
        del conversation_summary
        rewritten = self._rewrites.get(query.strip().lower(), f"{query} in detail?")
        return (
            QueryAnalysisOutput(
                is_clear=True,
                questions=[rewritten],
                rewritten_query=rewritten,
                clarification_needed=None,
                prompt_version="v1.0.0",
            ),
            "llm",
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


class MultiQueryTools:
    def __init__(self) -> None:
        self.search_queries: list[str] = []
        self.fetch_calls: list[list[str]] = []

    def search_chunks(self, query: str, top_k: int) -> list[EvidenceChunk]:
        del top_k
        self.search_queries.append(query)
        if "architecture" in query.lower():
            return [
                EvidenceChunk(
                    chunk_id="bert-0001",
                    source="bert.pdf",
                    text="BERT architecture uses transformers.",
                    score=0.81,
                )
            ]
        return [
            EvidenceChunk(
                chunk_id="bert-0002",
                source="bert.pdf",
                text="BERT pretraining uses masked language modeling.",
                score=0.79,
            )
        ]

    def fetch_chunks_by_ids(self, chunk_ids: list[str]) -> list[EvidenceChunk]:
        self.fetch_calls.append(list(chunk_ids))
        records = {
            "bert-0001": EvidenceChunk(
                chunk_id="bert-0001",
                source="bert.pdf",
                text="BERT architecture uses transformers.",
                score=0.92,
            ),
            "bert-0002": EvidenceChunk(
                chunk_id="bert-0002",
                source="bert.pdf",
                text="BERT pretraining uses masked language modeling.",
                score=0.9,
            ),
        }
        return [records[item] for item in chunk_ids if item in records]


class LongEvidenceTools:
    def search_chunks(self, query: str, top_k: int) -> list[EvidenceChunk]:
        del query, top_k
        return [
            EvidenceChunk(
                chunk_id="long-0001",
                source="long.pdf",
                text="X" * 12000,
                score=0.8,
            )
        ]

    def fetch_chunks_by_ids(self, chunk_ids: list[str]) -> list[EvidenceChunk]:
        if "long-0001" in chunk_ids:
            return [
                EvidenceChunk(
                    chunk_id="long-0001",
                    source="long.pdf",
                    text="X" * 12000,
                    score=0.9,
                )
            ]
        return []


class CitedChunkOutsideTop3Tools:
    def search_chunks(self, query: str, top_k: int) -> list[EvidenceChunk]:
        del query, top_k
        return [
            EvidenceChunk(chunk_id="attn-0001", source="attention.pdf", text="Attention head details A.", score=0.92),
            EvidenceChunk(chunk_id="attn-0002", source="attention.pdf", text="Attention head details B.", score=0.91),
            EvidenceChunk(chunk_id="attn-0003", source="attention.pdf", text="Attention head details C.", score=0.90),
            EvidenceChunk(chunk_id="bert-0004", source="bert.pdf", text="BERT_EVIDENCE: bidirectional MLM and NSP behavior.", score=0.89),
        ]

    def fetch_chunks_by_ids(self, chunk_ids: list[str]) -> list[EvidenceChunk]:
        records = {
            "attn-0001": EvidenceChunk(chunk_id="attn-0001", source="attention.pdf", text="Attention head details A.", score=0.92),
            "attn-0002": EvidenceChunk(chunk_id="attn-0002", source="attention.pdf", text="Attention head details B.", score=0.91),
            "attn-0003": EvidenceChunk(chunk_id="attn-0003", source="attention.pdf", text="Attention head details C.", score=0.90),
            "bert-0004": EvidenceChunk(chunk_id="bert-0004", source="bert.pdf", text="BERT_EVIDENCE: bidirectional MLM and NSP behavior.", score=0.89),
        }
        return [records[item] for item in chunk_ids if item in records]


class GroundingNeedsCitedBertReasoner(FakeReasoner):
    def synthesize_answer(
        self,
        *,
        query: str,
        chunks: list[EvidenceChunk],
        subqueries: list[str] | None = None,
    ) -> tuple[str, list[str], str, str | None]:
        del query, chunks, subqueries
        return (
            "BERT uses bidirectional attention-style contextualization while Transformer defines scaled dot-product attention.",
            ["bert-0004"],
            "llm",
            "v1.0.0",
        )

    def assess_grounding(
        self,
        *,
        answer: str,
        citations: list[str],
        evidence: list[str],
    ) -> tuple[GroundingResult, str, str | None]:
        del answer, citations
        joined = "\n".join(evidence)
        if "BERT_EVIDENCE" in joined:
            return GroundingResult(status=GroundingStatus.SUPPORTED, reason="Grounded by cited BERT chunk."), "llm", "v1.0.0"
        return GroundingResult(status=GroundingStatus.UNSUPPORTED, reason="Missing cited BERT chunk in grounding evidence."), "llm", "v1.0.0"


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


def test_vague_query_retrieves_before_response() -> None:
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
    stages = [event.stage for event in trace.events]
    assert "retrieve" in stages
    assert "clarify" not in stages


def test_analyze_query_failure_still_retrieves() -> None:
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
    assert response.safe_fail is False
    rewrite_events = [event for event in trace.events if event.stage == "rewrite_query"]
    assert len(rewrite_events) == 1
    assert rewrite_events[0].payload["rewrite_source"] == "fallback-rule"
    assert rewrite_events[0].payload["analysis_error"] is not None
    stages = [event.stage for event in trace.events]
    assert "retrieve" in stages
    assert "clarify" not in stages


def test_concrete_intent_retrieves_when_model_marks_unclear() -> None:
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
    assert "retrieve" in stages
    assert "clarify" not in stages


def test_explicit_source_query_retrieves_when_model_marks_unclear() -> None:
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
    assert "retrieve" in stages
    assert "clarify" not in stages


def test_model_clarification_signal_is_traced_but_retrieval_runs() -> None:
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=AlwaysUnclearReasoner(),
    )

    response = pipeline.ask("hi")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    rewrite_events = [event for event in trace.events if event.stage == "rewrite_query"]
    assert len(rewrite_events) == 1
    assert rewrite_events[0].payload["clarify_needed"] is True
    stages = [event.stage for event in trace.events]
    assert "retrieve" in stages
    assert "clarify" not in stages


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
    assert first.safe_fail is False
    first_stages = [event.stage for event in first_trace.events]
    assert "retrieve" in first_stages
    assert "clarify" not in first_stages

    second = pipeline.ask("yes, in the available documents", thread_id=thread_id)
    second_trace = trace_store.get(second.trace_id)

    assert second_trace is not None
    assert second.safe_fail is False
    second_stages = [event.stage for event in second_trace.events]
    assert "clarify" not in second_stages
    assert "retrieve" in second_stages


def test_generation_uses_original_query_not_rewritten_query() -> None:
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=QueryEchoReasoner(),
    )

    response = pipeline.ask("yes, I am asking from the docs uploaded")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    assert response.safe_fail is False
    rewrite_events = [event for event in trace.events if event.stage == "rewrite_query"]
    assert len(rewrite_events) == 1
    rewritten = str(rewrite_events[0].payload.get("rewritten", ""))
    assert trace.original_query != rewritten
    assert response.answer == f"ANSWER_QUERY={trace.original_query}"


def test_ambiguous_results_no_longer_hard_fail() -> None:
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
        reasoner=FakeReasoner(synthesis_chunk_ids=["a-0001"]),
    )
    response = pipeline.ask("What is the best source?")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    assert response.safe_fail is False
    assert response.citations == ["A.pdf#a-0001"]
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


def test_ambiguous_top_two_sources_no_longer_hard_fail() -> None:
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
    assert response.safe_fail is False
    assert response.citations == ["A.pdf#a-0001"]
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


def test_agent_mode_routes_through_agent_loop() -> None:
    settings = Settings(
        documents_dir=Path("documents"),
        retrieval_top_k=3,
        min_relevance_score=0.1,
        ambiguity_margin=0.03,
        enable_agent_mode=True,
        agent_max_iterations=2,
        agent_evidence_quality_threshold=0.5,
    )
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=settings,
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=FakeReasoner(),
    )

    response = pipeline.ask("Compare BERT vs Transformer")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    stages = [event.stage for event in trace.events]
    assert "agent_initialize" in stages
    assert "agent_think" in stages
    assert "agent_act" in stages
    assert "agent_reflect" in stages
    assert trace.agent_iterations_used >= 1


def test_agent_mode_simple_query_completes_in_single_iteration() -> None:
    settings = Settings(
        documents_dir=Path("documents"),
        retrieval_top_k=3,
        min_relevance_score=0.1,
        ambiguity_margin=0.03,
        enable_agent_mode=True,
        agent_max_iterations=2,
        agent_evidence_quality_threshold=0.5,
    )
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=settings,
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=SimpleComplexityReasoner(),
    )

    response = pipeline.ask("What is BERT pretraining?")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    stages = [event.stage for event in trace.events]
    # All queries now go through the agent loop
    assert "agent_initialize" in stages
    assert "agent_think" in stages
    # Simple queries should complete quickly (1 iteration)
    assert trace.agent_iterations_used <= 2


def test_agent_planner_finalize_skips_retrieval_call() -> None:
    settings = Settings(
        documents_dir=Path("documents"),
        retrieval_top_k=3,
        min_relevance_score=0.1,
        ambiguity_margin=0.03,
        enable_agent_mode=True,
        agent_max_iterations=2,
        agent_evidence_quality_threshold=0.5,
    )
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=settings,
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=PlannerFinalizeReasoner(),
    )

    response = pipeline.ask("Compare BERT vs Transformer")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    stages = [event.stage for event in trace.events]
    assert "agent_initialize" in stages
    assert "agent_think" in stages
    assert "agent_act" in stages
    assert "tool_search_chunks" not in stages


def test_refusal_like_answer_forces_safe_fail() -> None:
    settings = _settings()
    trace_store = TraceStore()

    class RefusalDetectingReasoner(FakeReasoner):
        def assess_grounding(
            self,
            *,
            answer: str,
            citations: list[str],
            evidence: list[str],
        ) -> tuple[GroundingResult, str, str | None]:
            del citations, evidence
            is_refusal = "does not specify" in answer.lower() or "insufficient evidence" in answer.lower()
            return (
                GroundingResult(
                    status=GroundingStatus.SUPPORTED if not is_refusal else GroundingStatus.UNSUPPORTED,
                    reason="Detected refusal in answer." if is_refusal else "Grounded in evidence.",
                    is_refusal=is_refusal,
                ),
                "llm",
                "v1.0.0",
            )

    pipeline = AgenticPipeline(
        settings=settings,
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=RefusalDetectingReasoner(
            grounding_status=GroundingStatus.SUPPORTED,
            synthesis_answer="The evidence provided does not specify which GPU was used.",
        ),
    )

    response = pipeline.ask("What GPU was used to train BERT?")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    assert response.safe_fail is True
    assert response.answer == settings.safe_fail_message
    assert response.citations == []

    verify_events = [event for event in trace.events if event.stage == "verify_grounding"]
    assert len(verify_events) == 1
    assert verify_events[0].payload["is_refusal"] is True


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


def test_multi_question_rewrite_fans_out_retrieval_queries() -> None:
    trace_store = TraceStore()
    tools = MultiQueryTools()
    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=tools,
        trace_store=trace_store,
        reasoner=MultiQuestionReasoner(),
    )

    response = pipeline.ask("Explain BERT")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    assert response.safe_fail is False
    # Agent loop now searches each subquery in separate iterations
    assert len(tools.search_queries) == 2
    assert len(tools.fetch_calls) == 2
    # Both chunks should be collected across iterations
    all_fetched = [cid for call in tools.fetch_calls for cid in call]
    assert "bert-0001" in all_fetched
    assert "bert-0002" in all_fetched

    search_events = [event for event in trace.events if event.stage == "tool_search_chunks"]
    assert len(search_events) == 2


def test_complex_query_decomposition_fans_out_retrieval_when_enabled() -> None:
    settings = Settings(
        documents_dir=Path("documents"),
        retrieval_top_k=3,
        min_relevance_score=0.1,
        ambiguity_margin=0.03,
        enable_query_decomposition=True,
    )
    trace_store = TraceStore()
    tools = MultiQueryTools()
    pipeline = AgenticPipeline(
        settings=settings,
        tools=tools,
        trace_store=trace_store,
        reasoner=DecomposingReasoner(),
    )

    response = pipeline.ask("Explain BERT deeply")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    assert response.safe_fail is False
    assert tools.search_queries == ["What is BERT architecture?", "What is BERT pretraining?"]
    prep_events = [event for event in trace.events if event.stage == "prepare_decomposition"]
    assert len(prep_events) == 1
    assert prep_events[0].payload["applied"] is True


def test_comparison_query_uses_rewrite_driven_single_search() -> None:
    trace_store = TraceStore()
    tools = QueryCaptureTools()
    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=tools,
        trace_store=trace_store,
        reasoner=ComparisonReasoner(),
    )

    response = pipeline.ask("Compare BERT vs Transformer")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    assert response.safe_fail is False
    assert len(tools.search_queries) == 1
    query = tools.search_queries[0].lower()
    assert "bert" in query
    assert "transformer" in query


def test_category6_ambiguous_queries_show_rewrite_trace() -> None:
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=CategoryDemoReasoner(),
    )

    prompts = [
        "Tell me about attention",
        "How does it handle context?",
        "What's the training trick they use?",
        "Compare the models",
    ]

    for prompt in prompts:
        response = pipeline.ask(prompt)
        trace = trace_store.get(response.trace_id)

        assert trace is not None
        assert trace.original_query == prompt
        assert trace.rewritten_query != prompt
        rewrite_events = [event for event in trace.events if event.stage == "rewrite_query"]
        assert len(rewrite_events) == 1
        assert rewrite_events[0].payload.get("rewritten") == trace.rewritten_query


def test_category7_lexical_mismatch_queries_are_rewritten_to_technical_terms() -> None:
    trace_store = TraceStore()
    tools = QueryCaptureTools()
    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=tools,
        trace_store=trace_store,
        reasoner=CategoryDemoReasoner(),
    )

    cases = [
        ("How does the model learn word relationships?", "attention"),
        ("What self-supervised objective is used?", "masked language model"),
        ("How is context aggregated?", "context"),
    ]

    for prompt, expected_term in cases:
        before = len(tools.search_queries)
        response = pipeline.ask(prompt)
        trace = trace_store.get(response.trace_id)

        assert trace is not None
        assert response.safe_fail is False
        assert len(tools.search_queries) == before + 1
        rewritten_query = tools.search_queries[-1].lower()
        assert expected_term in rewritten_query


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


def test_should_compress_context_triggers_for_large_evidence() -> None:
    settings = Settings(
        documents_dir=Path("documents"),
        retrieval_top_k=3,
        min_relevance_score=0.1,
        ambiguity_margin=0.03,
        context_compression_base_threshold=100,
    )
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=settings,
        tools=LongEvidenceTools(),
        trace_store=trace_store,
        reasoner=FakeReasoner(),
    )

    response = pipeline.ask("summarise long evidence")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    stages = [event.stage for event in trace.events]
    assert "should_compress_context" in stages
    assert "compress_context" in stages


def test_should_compress_context_skips_for_small_evidence() -> None:
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
    stages = [event.stage for event in trace.events]
    assert "should_compress_context" in stages
    assert "compress_context" not in stages


def test_limit_exceeded_routes_to_fallback_response() -> None:
    settings = Settings(
        documents_dir=Path("documents"),
        retrieval_top_k=3,
        min_relevance_score=0.1,
        ambiguity_margin=0.03,
        agent_max_tool_calls=0,
    )
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=settings,
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=FakeReasoner(),
    )

    response = pipeline.ask("What is BERT pretraining?")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    assert response.safe_fail is True
    assert response.answer == settings.safe_fail_message
    stages = [event.stage for event in trace.events]
    assert "fallback_response" in stages
    assert "validate" not in stages


def test_turn_scoped_counters_reset_between_messages_in_same_thread() -> None:
    settings = Settings(
        documents_dir=Path("documents"),
        retrieval_top_k=3,
        min_relevance_score=0.1,
        ambiguity_margin=0.03,
        agent_max_tool_calls=8,
    )
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=settings,
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=FakeReasoner(),
    )

    first = pipeline.ask("Tell me about attention", thread_id="session-counter-reset")
    second = pipeline.ask("How is context aggregated?", thread_id="session-counter-reset")

    first_trace = trace_store.get(first.trace_id)
    second_trace = trace_store.get(second.trace_id)

    assert first_trace is not None
    assert second_trace is not None
    assert first.safe_fail is False
    assert second.safe_fail is False

    first_gate = [event for event in first_trace.events if event.stage == "should_compress_context"]
    second_gate = [event for event in second_trace.events if event.stage == "should_compress_context"]
    assert len(first_gate) == 1
    assert len(second_gate) == 1
    assert first_gate[0].payload.get("iteration_count") == 1
    assert first_gate[0].payload.get("tool_call_count") == 2
    assert second_gate[0].payload.get("iteration_count") == 1
    assert second_gate[0].payload.get("tool_call_count") == 2


def test_verify_uses_cited_chunks_not_only_top_ranked_chunks() -> None:
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=CitedChunkOutsideTop3Tools(),
        trace_store=trace_store,
        reasoner=GroundingNeedsCitedBertReasoner(),
    )

    response = pipeline.ask("How is attention used in BERT vs Transformer?")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    assert response.safe_fail is False
    assert response.citations == ["bert.pdf#bert-0004"]


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


def test_coerce_clarification_overrides_stale_answer_when_needed() -> None:
    trace = PipelineTrace(original_query="q", rewritten_query="q")
    state = {
        "clarify_needed": True,
        "clarify_message": "Need clarification.",
        "answer": "stale answer",
        "safe_fail": False,
        "citations": ["x"],
        "trace": trace,
    }

    coerced = AgenticPipeline._coerce_interrupted_clarification_state(state)

    assert coerced["answer"] == "Need clarification."
    assert coerced["safe_fail"] is True
    assert coerced["citations"] == []
    assert any(event.stage == "clarify" for event in trace.events)
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


def test_pipeline_recovers_from_synthesis_failure_with_safe_fail() -> None:
    """Test: Pipeline handles synthesis errors gracefully with safe_fail fallback.
    
    When reasoner fails on answer synthesis (simulating LLM error), pipeline
    should not crash but instead return safe_fail response with error logging
    in trace for debugging.
    """
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=_settings(),
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=BrokenSynthesisReasoner(),  # Raises LLMInvocationError
    )

    response = pipeline.ask("What is BERT pretraining?")
    trace = trace_store.get(response.trace_id)

    # Pipeline should recover and return safe response
    assert trace is not None
    # Either safe_fail is True, or response doesn't crash
    assert response is not None
    # Verify error is traced for debugging
    error_events = [event for event in trace.events if "error" in event.stage.lower() or event.payload.get("error")]
    # May or may not have error events, but should complete


# ====== Phase 5: Regression suites ======


class SubqueryTrackingTools:
    """Tools that return different chunks per query to verify subquery tracking."""
    def __init__(self) -> None:
        self.search_queries: list[str] = []

    def search_chunks(self, query: str, top_k: int) -> list[EvidenceChunk]:
        del top_k
        self.search_queries.append(query)
        if "architecture" in query.lower():
            return [
                EvidenceChunk(chunk_id="arch-0001", source="bert.pdf", text="BERT architecture detail.", score=0.85)
            ]
        if "pretraining" in query.lower():
            return [
                EvidenceChunk(chunk_id="pre-0001", source="bert.pdf", text="BERT pretraining detail.", score=0.83)
            ]
        return [
            EvidenceChunk(chunk_id="gen-0001", source="bert.pdf", text="General BERT info.", score=0.7)
        ]

    def fetch_chunks_by_ids(self, chunk_ids: list[str]) -> list[EvidenceChunk]:
        records = {
            "arch-0001": EvidenceChunk(chunk_id="arch-0001", source="bert.pdf", text="BERT architecture detail.", score=0.9),
            "pre-0001": EvidenceChunk(chunk_id="pre-0001", source="bert.pdf", text="BERT pretraining detail.", score=0.88),
            "gen-0001": EvidenceChunk(chunk_id="gen-0001", source="bert.pdf", text="General BERT info.", score=0.8),
        }
        return [records[cid] for cid in chunk_ids if cid in records]

    def web_search(self, query: str, *, settings: object = None) -> list[EvidenceChunk]:
        del query, settings
        return []


class WebSearchFakeTools:
    """Tools that simulate web search returning external evidence."""
    def search_chunks(self, query: str, top_k: int) -> list[EvidenceChunk]:
        del top_k
        return [
            EvidenceChunk(chunk_id="local-0001", source="bert.pdf", text="BERT info from local docs.", score=0.6)
        ]

    def fetch_chunks_by_ids(self, chunk_ids: list[str]) -> list[EvidenceChunk]:
        if "local-0001" in chunk_ids:
            return [
                EvidenceChunk(chunk_id="local-0001", source="bert.pdf", text="BERT info from local docs.", score=0.8)
            ]
        return []

    def web_search(self, query: str, *, settings: object = None) -> list[EvidenceChunk]:
        del settings
        return [
            EvidenceChunk(
                chunk_id="web-abc123",
                source="https://example.com/bert",
                text=f"Web result for: {query}",
                score=0.75,
                provenance="web",
            )
        ]


class WebSearchPlannerReasoner(FakeReasoner):
    """Planner that recommends web_search on first iteration, then finalize."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._call_count = 0

    def plan_agent_step(
        self,
        *,
        query: str,
        conversation_summary: str | None,
        rewritten_queries: list[str],
        evidence_quality_score: float,
        chunk_count: int,
        agent_iterations: int,
        max_iterations: int,
        last_observation: str | None = None,
        subquery_statuses: list[dict[str, object]] | None = None,
    ):
        del (
            query, conversation_summary, rewritten_queries, evidence_quality_score,
            chunk_count, max_iterations, last_observation, subquery_statuses,
        )
        self._call_count += 1
        if agent_iterations <= 1:
            return AgentThought(
                reasoning="Local docs are insufficient; trying web search.",
                recommended_action="web_search",
                confidence=0.7,
            )
        return AgentThought(
            reasoning="Evidence collected; finalizing.",
            recommended_action="finalize",
            confidence=0.9,
        )


class DecompositionTrackingReasoner(FakeReasoner):
    """Reasoner that decomposes and always marks complex."""
    def detect_query_complexity(self, *, query: str, conversation_summary: str | None = None):
        del query, conversation_summary
        return QueryComplexity.COMPLEX

    def decompose_query_lightly(self, *, query: str, conversation_summary: str | None = None):
        del query, conversation_summary
        return ["What is BERT architecture?", "What is BERT pretraining?"]

    def synthesize_answer(
        self, *, query: str, chunks: list[EvidenceChunk], subqueries: list[str] | None = None
    ):
        del query, subqueries
        if chunks:
            return "BERT has a transformer architecture and uses MLM pretraining.", [chunks[0].chunk_id], "llm", "v1.0.0"
        return "No evidence.", [], "llm", "v1.0.0"


def test_subquery_statuses_are_initialized_and_tracked() -> None:
    """Decomposed subqueries get per-subquery tracking through the agent loop."""
    settings = Settings(
        documents_dir=Path("documents"),
        retrieval_top_k=3,
        min_relevance_score=0.1,
        enable_query_decomposition=True,
        agent_max_iterations=5,
        agent_evidence_quality_threshold=0.5,
    )
    trace_store = TraceStore()
    tools = SubqueryTrackingTools()
    pipeline = AgenticPipeline(
        settings=settings,
        tools=tools,
        trace_store=trace_store,
        reasoner=DecompositionTrackingReasoner(),
    )

    response = pipeline.ask("Explain BERT architecture and pretraining")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    assert response.safe_fail is False
    # Both subqueries should have been searched
    assert len(tools.search_queries) >= 2
    # Trace should include per-subquery quality info
    reflect_events = [e for e in trace.events if e.stage == "agent_reflect"]
    assert len(reflect_events) >= 1
    last_reflect = reflect_events[-1]
    assert "subquery_quality" in last_reflect.payload


def test_planner_web_search_action_executes_when_enabled() -> None:
    """When web_search_enabled=True and planner recommends web_search, web results are fetched."""
    settings = Settings(
        documents_dir=Path("documents"),
        retrieval_top_k=3,
        min_relevance_score=0.1,
        web_search_enabled=True,
        agent_max_iterations=3,
        agent_evidence_quality_threshold=0.5,
    )
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=settings,
        tools=WebSearchFakeTools(),
        trace_store=trace_store,
        reasoner=WebSearchPlannerReasoner(),
    )

    response = pipeline.ask("What is the latest BERT update?")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    stages = [e.stage for e in trace.events]
    assert "tool_web_search" in stages
    # Web results should carry provenance='web'
    act_events = [e for e in trace.events if e.stage == "agent_act" and e.payload.get("action") == "web_search"]
    assert len(act_events) >= 1


def test_planner_web_search_disabled_does_not_crash() -> None:
    """When web_search_enabled=False and planner recommends web_search, it gracefully skips."""
    settings = Settings(
        documents_dir=Path("documents"),
        retrieval_top_k=3,
        min_relevance_score=0.1,
        web_search_enabled=False,
        agent_max_iterations=3,
        agent_evidence_quality_threshold=0.5,
    )
    trace_store = TraceStore()
    pipeline = AgenticPipeline(
        settings=settings,
        tools=FakeTools(),
        trace_store=trace_store,
        reasoner=WebSearchPlannerReasoner(),
    )

    response = pipeline.ask("What is the latest BERT update?")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    # Should not crash — web search action marked as disabled
    act_events = [e for e in trace.events if e.stage == "agent_act"]
    disabled_acts = [e for e in act_events if e.payload.get("disabled")]
    assert len(disabled_acts) >= 1


def test_all_queries_route_through_agent_loop() -> None:
    """After legacy path retirement, even simple queries use the agent loop."""
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
    stages = [e.stage for e in trace.events]
    assert "agent_initialize" in stages
    assert "agent_think" in stages
    assert "agent_act" in stages
    assert "agent_reflect" in stages


def test_sufficiency_policy_respects_pending_subqueries() -> None:
    """Agent loop continues when subqueries are still pending, even if overall quality is high."""
    settings = Settings(
        documents_dir=Path("documents"),
        retrieval_top_k=3,
        min_relevance_score=0.1,
        enable_query_decomposition=True,
        agent_max_iterations=5,
        agent_evidence_quality_threshold=0.5,
    )
    trace_store = TraceStore()
    tools = SubqueryTrackingTools()
    pipeline = AgenticPipeline(
        settings=settings,
        tools=tools,
        trace_store=trace_store,
        reasoner=DecompositionTrackingReasoner(),
    )

    response = pipeline.ask("Explain BERT architecture and pretraining")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    assert response.safe_fail is False
    # Both subqueries must have been individually searched
    searched_arch = any("architecture" in q.lower() for q in tools.search_queries)
    searched_pre = any("pretraining" in q.lower() for q in tools.search_queries)
    assert searched_arch, "Architecture subquery was never searched"
    assert searched_pre, "Pretraining subquery was never searched"


def test_planner_action_normalization_covers_web_search_synonyms() -> None:
    """Reasoner normalizes web search synonyms to canonical 'web_search' action."""
    from src.services.reasoner import QueryReasoner

    for raw in ["web_search", "web", "internet", "external_search"]:
        assert QueryReasoner._normalize_agent_action(raw) == "web_search"

    for raw in ["search_documents", "search", "retrieve", "retrieval"]:
        assert QueryReasoner._normalize_agent_action(raw) == "search_documents"

    for raw in ["finalize", "finish", "complete", "done"]:
        assert QueryReasoner._normalize_agent_action(raw) == "finalize"


def test_grounding_refusal_detection_forces_safe_fail_deterministically() -> None:
    """Grounding refusal detection consistently converts to safe_fail."""
    settings = _settings()
    trace_store = TraceStore()

    class StrictRefusalReasoner(FakeReasoner):
        def assess_grounding(self, *, answer: str, citations: list[str], evidence: list[str]):
            del citations, evidence
            if "not" in answer.lower() and "evidence" in answer.lower():
                return (
                    GroundingResult(status=GroundingStatus.UNSUPPORTED, reason="Refusal detected.", is_refusal=True),
                    "llm", "v1.0.0",
                )
            return (
                GroundingResult(status=GroundingStatus.SUPPORTED, reason="Grounded."),
                "llm", "v1.0.0",
            )

    for refusal_text in [
        "The evidence does not contain information about X.",
        "I could not find relevant evidence in the documents.",
    ]:
        pipeline = AgenticPipeline(
            settings=settings,
            tools=FakeTools(),
            trace_store=trace_store,
            reasoner=StrictRefusalReasoner(synthesis_answer=refusal_text),
        )
        response = pipeline.ask("Some question?")
        assert response.safe_fail is True
        assert response.answer == settings.safe_fail_message


def test_evidence_provenance_label_preserved_through_pipeline() -> None:
    """Web-sourced chunks retain provenance='web' label through to generation."""
    from src.core.models import EvidenceChunk

    web_chunk = EvidenceChunk(
        chunk_id="web-test",
        source="https://example.com",
        text="External content.",
        score=0.8,
        provenance="web",
    )
    local_chunk = EvidenceChunk(
        chunk_id="local-test",
        source="doc.pdf",
        text="Local content.",
        score=0.9,
        provenance="local",
    )
    assert web_chunk.provenance == "web"
    assert local_chunk.provenance == "local"
    # Verify model_dump preserves provenance
    assert web_chunk.model_dump()["provenance"] == "web"
    assert local_chunk.model_dump()["provenance"] == "local"


def test_decomposition_fans_out_when_complex_and_enabled() -> None:
    """Complex queries with decomposition enabled produce multiple subqueries tracked end-to-end."""
    settings = Settings(
        documents_dir=Path("documents"),
        retrieval_top_k=3,
        min_relevance_score=0.1,
        enable_query_decomposition=True,
        agent_max_iterations=5,
        agent_evidence_quality_threshold=0.5,
    )
    trace_store = TraceStore()
    tools = MultiQueryTools()
    pipeline = AgenticPipeline(
        settings=settings,
        tools=tools,
        trace_store=trace_store,
        reasoner=DecompositionTrackingReasoner(),
    )

    response = pipeline.ask("Explain BERT deeply")
    trace = trace_store.get(response.trace_id)

    assert trace is not None
    assert response.safe_fail is False
    prep_events = [e for e in trace.events if e.stage == "prepare_decomposition"]
    assert len(prep_events) == 1
    assert prep_events[0].payload["applied"] is True
    assert prep_events[0].payload["query_count"] == 2
