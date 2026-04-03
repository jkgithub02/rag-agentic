from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agentic_rag.config import Settings
from agentic_rag.models import (
    AskResponse,
    EvidenceChunk,
    GroundingResult,
    GroundingStatus,
    PipelineOutput,
    PipelineTrace,
    TraceEvent,
    ValidationResult,
    ValidationStatus,
)
from agentic_rag.retrieval import Retriever
from agentic_rag.trace_store import TraceStore


class PipelineState(TypedDict, total=False):
    """Internal state carried across LangGraph nodes."""

    query: str
    original_query: str
    rewritten_query: str
    retrieved_chunks: list[EvidenceChunk]
    validation: ValidationResult
    retry_count: int
    retry_triggered: bool
    retry_reason: str | None
    answer: str
    citations: list[str]
    safe_fail: bool
    grounding: GroundingResult
    trace: PipelineTrace


class AgenticPipeline:
    """Executes the interview-focused agentic RAG control loop."""

    def __init__(self, settings: Settings, retriever: Retriever, trace_store: TraceStore) -> None:
        self._settings = settings
        self._retriever = retriever
        self._trace_store = trace_store
        self._graph = self._build_graph()

    def ask(self, query: str) -> AskResponse:
        output, _ = self.run(query)
        return AskResponse(**output.model_dump())

    def run(self, query: str) -> tuple[PipelineOutput, PipelineTrace]:
        final_state = self._graph.invoke({"query": query})
        trace = final_state["trace"]
        self._trace_store.save(trace)

        output = PipelineOutput(
            answer=final_state["answer"],
            citations=final_state["citations"],
            safe_fail=final_state["safe_fail"],
            trace_id=trace.trace_id,
        )
        return output, trace

    def _build_graph(self):
        graph = StateGraph(PipelineState)
        graph.add_node("understand", self._node_understand)
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("validate", self._node_validate)
        graph.add_node("retry", self._node_retry)
        graph.add_node("generate", self._node_generate)
        graph.add_node("verify_grounding", self._node_verify_grounding)
        graph.add_node("answer", self._node_answer)

        graph.add_edge(START, "understand")
        graph.add_edge("understand", "retrieve")
        graph.add_edge("retrieve", "validate")
        graph.add_conditional_edges(
            "validate",
            self._route_after_validate,
            {
                "retry": "retry",
                "generate": "generate",
            },
        )
        graph.add_edge("retry", "retrieve")
        graph.add_edge("generate", "verify_grounding")
        graph.add_edge("verify_grounding", "answer")
        graph.add_edge("answer", END)
        return graph.compile()

    def _node_understand(self, state: PipelineState) -> PipelineState:
        original_query = " ".join(state["query"].split())
        rewritten_query = self._rewrite_query(original_query)
        trace = PipelineTrace(original_query=original_query, rewritten_query=rewritten_query)
        self._event(trace, "understand", {"original": original_query, "rewritten": rewritten_query})
        return {
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "retry_count": 0,
            "retry_triggered": False,
            "retry_reason": None,
            "trace": trace,
        }

    def _node_retrieve(self, state: PipelineState) -> PipelineState:
        chunks = self._retriever.retrieve(state["rewritten_query"], self._settings.retrieval_top_k)
        trace = state["trace"]
        stage = "retrieve_retry" if state["retry_count"] > 0 else "retrieve"
        self._event(trace, stage, {"hits": self._serialize_chunks(chunks)})
        return {"retrieved_chunks": chunks, "trace": trace}

    def _node_validate(self, state: PipelineState) -> PipelineState:
        validation = self._validate(state["retrieved_chunks"])
        trace = state["trace"]
        stage = "validate_retry" if state["retry_count"] > 0 else "validate"
        self._event(trace, stage, validation.model_dump())
        return {"validation": validation, "trace": trace}

    def _node_retry(self, state: PipelineState) -> PipelineState:
        validation = state["validation"]
        rewritten_query = self._rewrite_for_retry(
            state["original_query"],
            state["retrieved_chunks"],
        )
        retry_count = state["retry_count"] + 1
        trace = state["trace"]
        trace.retry_triggered = True
        trace.retry_reason = validation.reason
        trace.rewritten_query = rewritten_query
        self._event(
            trace,
            "retry",
            {"attempt": retry_count, "reason": validation.reason, "rewritten": rewritten_query},
        )
        return {
            "retry_count": retry_count,
            "retry_triggered": True,
            "retry_reason": validation.reason,
            "rewritten_query": rewritten_query,
            "trace": trace,
        }

    def _route_after_validate(self, state: PipelineState) -> str:
        validation = state["validation"]
        if (
            validation.status == ValidationStatus.RETRY
            and state["retry_count"] < self._settings.max_retry_count
        ):
            return "retry"
        return "generate"

    def _node_generate(self, state: PipelineState) -> PipelineState:
        validation = state["validation"]
        trace = state["trace"]

        if validation.status == ValidationStatus.FAIL:
            answer = self._settings.safe_fail_message
            citations: list[str] = []
            safe_fail = True
            self._event(trace, "generate", {"safe_fail": True, "answer": answer})
            return {
                "answer": answer,
                "citations": citations,
                "safe_fail": safe_fail,
                "trace": trace,
            }

        answer, citations = self._generate(state["original_query"], state["retrieved_chunks"])
        self._event(
            trace,
            "generate",
            {"safe_fail": False, "answer_preview": answer[:300], "citations": citations},
        )
        return {
            "answer": answer,
            "citations": citations,
            "safe_fail": False,
            "trace": trace,
        }

    def _node_verify_grounding(self, state: PipelineState) -> PipelineState:
        answer = state["answer"]
        citations = state["citations"]
        chunks = state["retrieved_chunks"]
        trace = state["trace"]

        grounding = self._verify_grounding(answer, citations, chunks)
        safe_fail = state["safe_fail"]
        if grounding.status == GroundingStatus.UNSUPPORTED:
            safe_fail = True
            answer = self._settings.safe_fail_message
            citations = []

        self._event(trace, "verify_grounding", grounding.model_dump())
        return {
            "grounding": grounding,
            "safe_fail": safe_fail,
            "answer": answer,
            "citations": citations,
            "trace": trace,
        }

    def _node_answer(self, state: PipelineState) -> PipelineState:
        trace = state["trace"]
        trace.final_grounding_status = state["grounding"].status
        self._event(
            trace,
            "answer",
            {
                "safe_fail": state["safe_fail"],
                "citation_count": len(state["citations"]),
                "grounding_status": state["grounding"].status,
            },
        )
        return {"trace": trace}

    @staticmethod
    def _event(trace: PipelineTrace, stage: str, payload: dict[str, object]) -> None:
        trace.events.append(TraceEvent(stage=stage, payload=payload))

    @staticmethod
    def _rewrite_query(query: str) -> str:
        # Keep initial rewrite conservative to preserve user intent and traceability.
        return query.rstrip("?") + "?"

    @staticmethod
    def _rewrite_for_retry(original_query: str, chunks: Iterable[EvidenceChunk]) -> str:
        sources = [chunk.source for chunk in chunks]
        if not sources:
            return f"{original_query} Focus on one specific paper title and key contribution."

        dominant_source, _ = Counter(sources).most_common(1)[0]
        return f"{original_query} Focus specifically on {dominant_source} and cite concrete claims."

    def _validate(self, chunks: list[EvidenceChunk]) -> ValidationResult:
        if not chunks:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                reason="No retrieval hits were found.",
                confidence=0.0,
            )

        top_score = chunks[0].score
        if top_score < self._settings.min_relevance_score:
            return ValidationResult(
                status=ValidationStatus.RETRY,
                reason="Top retrieval score is below relevance threshold.",
                confidence=top_score,
            )

        if len(chunks) >= 2:
            score_delta = chunks[0].score - chunks[1].score
            if (
                score_delta < self._settings.ambiguity_margin
                and chunks[0].source != chunks[1].source
            ):
                return ValidationResult(
                    status=ValidationStatus.RETRY,
                    reason="Top evidence is ambiguous across different sources.",
                    confidence=top_score,
                )

        return ValidationResult(
            status=ValidationStatus.PASS,
            reason="Evidence quality passed validation.",
            confidence=top_score,
        )

    @staticmethod
    def _generate(query: str, chunks: list[EvidenceChunk]) -> tuple[str, list[str]]:
        selected = chunks[:2]
        evidence_summaries = [f"[{item.chunk_id}] {item.text[:260]}" for item in selected]
        citations = [f"{item.source}#{item.chunk_id}" for item in selected]

        answer = (
            f"Question: {query}\n\n"
            "Grounded synthesis from retrieved evidence:\n"
            f"- {evidence_summaries[0]}"
        )
        if len(evidence_summaries) > 1:
            answer += f"\n- {evidence_summaries[1]}"
        return answer, citations

    @staticmethod
    def _verify_grounding(
        answer: str,
        citations: list[str],
        chunks: list[EvidenceChunk],
    ) -> GroundingResult:
        if not answer.strip() or not chunks:
            return GroundingResult(
                status=GroundingStatus.UNSUPPORTED,
                reason="Answer was generated without evidence.",
            )

        if not citations:
            return GroundingResult(
                status=GroundingStatus.PARTIAL,
                reason="Answer exists but no citations were produced.",
            )

        chunk_ids = {chunk.chunk_id for chunk in chunks}
        cited_ids = {citation.split("#")[-1] for citation in citations}
        if not cited_ids.issubset(chunk_ids):
            return GroundingResult(
                status=GroundingStatus.PARTIAL,
                reason="At least one citation does not map to retrieved evidence.",
            )

        return GroundingResult(
            status=GroundingStatus.SUPPORTED,
            reason="Answer is grounded in retrieved evidence with valid citations.",
        )

    @staticmethod
    def _serialize_chunks(chunks: list[EvidenceChunk]) -> list[dict[str, object]]:
        return [
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "score": round(chunk.score, 4),
                "text_preview": chunk.text[:160],
                "text": chunk.text,
            }
            for chunk in chunks
        ]
