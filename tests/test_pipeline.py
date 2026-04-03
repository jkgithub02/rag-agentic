from __future__ import annotations

from pathlib import Path

from agentic_rag.config import Settings
from agentic_rag.models import EvidenceChunk
from agentic_rag.pipeline import AgenticPipeline
from agentic_rag.trace_store import TraceStore


class AmbiguousThenResolvedRetriever:
    def retrieve(self, query: str, top_k: int) -> list[EvidenceChunk]:
        if "Focus specifically" in query:
            return [
                EvidenceChunk(
                    chunk_id="attention-0001",
                    source="attention is all you need.pdf",
                    text="The transformer removes recurrence and uses self-attention.",
                    score=0.78,
                )
            ]

        return [
            EvidenceChunk(
                chunk_id="attention-0001",
                source="attention is all you need.pdf",
                text="Transformer architecture details.",
                score=0.2,
            ),
            EvidenceChunk(
                chunk_id="bert-0001",
                source="bert.pdf",
                text="BERT pre-training details.",
                score=0.19,
            ),
        ]


class EmptyRetriever:
    def retrieve(self, query: str, top_k: int) -> list[EvidenceChunk]:
        return []


def _settings() -> Settings:
    return Settings(
        documents_dir=Path("documents"),
        index_file=Path(".data/test_chunks.json"),
        min_relevance_score=0.12,
        ambiguity_margin=0.03,
        max_retry_count=1,
    )


def test_retry_loop_triggers_for_ambiguous_retrieval() -> None:
    pipeline = AgenticPipeline(
        settings=_settings(),
        retriever=AmbiguousThenResolvedRetriever(),
        trace_store=TraceStore(),
    )

    output, trace = pipeline.run("What are the key innovations?")

    assert trace.retry_triggered is True
    assert trace.original_query != trace.rewritten_query
    assert output.safe_fail is False
    assert len(output.citations) > 0


def test_safe_fail_when_no_evidence() -> None:
    pipeline = AgenticPipeline(
        settings=_settings(),
        retriever=EmptyRetriever(),
        trace_store=TraceStore(),
    )

    output, trace = pipeline.run("What is the learning rate schedule in the paper?")

    assert trace.retry_triggered is False
    assert output.safe_fail is True
    assert output.citations == []
    assert "sufficient evidence" in output.answer.lower()
