from __future__ import annotations

from pathlib import Path

from src.config import Settings
from src.models import EvidenceChunk, QueryRewriteOutput
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
    def rewrite_query(self, query: str) -> tuple[QueryRewriteOutput, str]:
        output = QueryRewriteOutput(rewritten_query=f"{query} in detail?", prompt_version="v1.0.0")
        return output, "llm"


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
