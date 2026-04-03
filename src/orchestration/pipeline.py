from __future__ import annotations

from collections import Counter
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.agent.tools import AgentTools
from src.config import Settings
from src.models import (
    AskResponse,
    GroundingResult,
    GroundingStatus,
    PipelineTrace,
    QueryRewriteOutput,
    TraceEvent,
    ValidationResult,
    ValidationStatus,
)
from src.reasoner import QueryReasoner
from src.trace_store import TraceStore


class PipelineState(TypedDict, total=False):
    query: str
    original_query: str
    rewritten_query: str
    clarify_needed: bool
    retry_count: int
    validation: ValidationResult
    answer: str
    citations: list[str]
    safe_fail: bool
    grounding: GroundingResult
    trace: PipelineTrace
    chunks: list


class AgenticPipeline:
    """ LangGraph pipeline with explicit tool usage."""

    def __init__(
        self,
        settings: Settings,
        tools: AgentTools,
        trace_store: TraceStore,
        reasoner: QueryReasoner | None = None,
    ) -> None:
        self._settings = settings
        self._tools = tools
        self._trace_store = trace_store
        self._reasoner = reasoner
        self._graph = self._build_graph()

    def ask(self, query: str) -> AskResponse:
        state = self._graph.invoke({"query": query})
        trace: PipelineTrace = state["trace"]
        self._trace_store.save(trace)

        return AskResponse(
            answer=state["answer"],
            citations=state["citations"],
            safe_fail=state["safe_fail"],
            trace_id=trace.trace_id,
        )

    def _build_graph(self):
        graph = StateGraph(PipelineState)
        graph.add_node("understand", self._understand)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("validate", self._validate)
        graph.add_node("retry", self._retry)
        graph.add_node("retry_exhausted", self._retry_exhausted)
        graph.add_node("clarify", self._clarify)
        graph.add_node("generate", self._generate)
        graph.add_node("verify", self._verify)
        graph.add_node("finish", self._finish)

        graph.add_edge(START, "understand")
        graph.add_conditional_edges(
            "understand",
            self._route_after_understand,
            {"clarify": "clarify", "retrieve": "retrieve"},
        )
        graph.add_edge("clarify", "finish")
        graph.add_edge("retrieve", "validate")
        graph.add_conditional_edges(
            "validate",
            self._route_after_validate,
            {"retry": "retry", "retry_exhausted": "retry_exhausted", "generate": "generate"},
        )
        graph.add_edge("retry", "retrieve")
        graph.add_edge("retry_exhausted", "generate")
        graph.add_edge("generate", "verify")
        graph.add_edge("verify", "finish")
        graph.add_edge("finish", END)
        return graph.compile()

    def _understand(self, state: PipelineState) -> PipelineState:
        original_query = " ".join(state["query"].split())
        rewrite_result: QueryRewriteOutput | None = None
        rewrite_source = "fallback-no-reasoner"

        if self._reasoner is not None:
            rewrite_result, rewrite_source = self._reasoner.rewrite_query(original_query)

        rewritten_query = (
            rewrite_result.rewritten_query
            if rewrite_result is not None
            else original_query.rstrip("?") + "?"
        )
        clarify_needed = self._needs_clarification(original_query)
        trace = PipelineTrace(original_query=original_query, rewritten_query=rewritten_query)
        self._event(
            trace,
            "understand",
            {
                "original": original_query,
                "rewritten": rewritten_query,
                "rewrite_source": rewrite_source,
                "prompt_version": rewrite_result.prompt_version if rewrite_result else None,
                "clarify_needed": clarify_needed,
            },
        )
        return {
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "clarify_needed": clarify_needed,
            "retry_count": 0,
            "trace": trace,
        }

    def _route_after_understand(self, state: PipelineState) -> str:
        if state.get("clarify_needed"):
            return "clarify"
        return "retrieve"

    def _clarify(self, state: PipelineState) -> PipelineState:
        trace = state["trace"]
        answer = self._settings.clarification_message
        grounding = GroundingResult(
            status=GroundingStatus.UNSUPPORTED,
            reason="Query is too vague and requires clarification.",
        )
        self._event(
            trace,
            "clarify",
            {"message": answer},
        )
        return {
            "answer": answer,
            "citations": [],
            "safe_fail": True,
            "grounding": grounding,
            "trace": trace,
        }

    def _retrieve(self, state: PipelineState) -> PipelineState:
        trace = state["trace"]
        self._event(trace, "tool_search_chunks", {"query": state["rewritten_query"]})
        chunks = self._tools.search_chunks(state["rewritten_query"], self._settings.retrieval_top_k)
        stage = "retrieve_retry" if state["retry_count"] > 0 else "retrieve"
        self._event(trace, stage, {"hits": [c.model_dump() for c in chunks]})
        return {"chunks": chunks, "trace": trace}

    def _validate(self, state: PipelineState) -> PipelineState:
        chunks = state["chunks"]
        if not chunks:
            validation = ValidationResult(
                status=ValidationStatus.FAIL,
                reason="No retrieval hits.",
                confidence=0.0,
            )
        else:
            top_score = chunks[0].score
            if top_score < self._settings.min_relevance_score:
                validation = ValidationResult(
                    status=ValidationStatus.RETRY,
                    reason="Top score below threshold.",
                    confidence=top_score,
                )
            elif (
                len(chunks) >= 2
                and (chunks[0].score - chunks[1].score) < self._settings.ambiguity_margin
            ):
                validation = ValidationResult(
                    status=ValidationStatus.RETRY,
                    reason="Top two hits are ambiguous.",
                    confidence=top_score,
                )
            else:
                validation = ValidationResult(
                    status=ValidationStatus.PASS,
                    reason="Validation passed.",
                    confidence=top_score,
                )

        trace = state["trace"]
        stage = "validate_retry" if state["retry_count"] > 0 else "validate"
        self._event(trace, stage, validation.model_dump())
        return {"validation": validation, "trace": trace}

    def _route_after_validate(self, state: PipelineState) -> str:
        if (
            state["validation"].status == ValidationStatus.RETRY
            and state["retry_count"] < self._settings.max_retry_count
        ):
            return "retry"
        if state["validation"].status == ValidationStatus.RETRY:
            return "retry_exhausted"
        return "generate"

    def _retry(self, state: PipelineState) -> PipelineState:
        trace = state["trace"]
        sources = [chunk.source for chunk in state["chunks"]]
        evidence = [chunk.text[:220] for chunk in state["chunks"][:3]]
        rewrite_source = "fallback-no-reasoner"
        prompt_version: str | None = None

        if self._reasoner is not None:
            rewrite_result, rewrite_source = self._reasoner.rewrite_for_retry(
                original_query=state["original_query"],
                retry_reason=state["validation"].reason,
                evidence=evidence,
            )
            rewritten = rewrite_result.rewritten_query
            prompt_version = rewrite_result.prompt_version
        elif sources:
            dominant_source, _ = Counter(sources).most_common(1)[0]
            rewritten = f"{state['original_query']} Focus on {dominant_source}."
        else:
            rewritten = f"{state['original_query']} Use one specific paper title."

        retry_count = state["retry_count"] + 1
        trace.retry_triggered = True
        trace.retry_reason = state["validation"].reason
        trace.rewritten_query = rewritten
        self._event(
            trace,
            "retry",
            {
                "attempt": retry_count,
                "rewritten": rewritten,
                "rewrite_source": rewrite_source,
                "prompt_version": prompt_version,
            },
        )
        return {"retry_count": retry_count, "rewritten_query": rewritten, "trace": trace}

    def _retry_exhausted(self, state: PipelineState) -> PipelineState:
        trace = state["trace"]
        forced = ValidationResult(
            status=ValidationStatus.FAIL,
            reason=(
                "Retry budget exhausted before disambiguation. "
                f"Last reason: {state['validation'].reason}"
            ),
            confidence=state["validation"].confidence,
        )
        self._event(trace, "retry_exhausted", forced.model_dump())
        return {"validation": forced, "trace": trace}

    def _generate(self, state: PipelineState) -> PipelineState:
        trace = state["trace"]
        if state["validation"].status == ValidationStatus.FAIL:
            answer = self._settings.safe_fail_message
            self._event(trace, "generate", {"safe_fail": True, "answer": answer})
            return {"answer": answer, "citations": [], "safe_fail": True, "trace": trace}

        chunks = state["chunks"][:2]
        citations = [f"{chunk.source}#{chunk.chunk_id}" for chunk in chunks]
        answer_lines = [f"- [{chunk.chunk_id}] {chunk.text[:220]}" for chunk in chunks]
        answer = "Question: " + state["original_query"] + "\n\n" + "\n".join(answer_lines)
        self._event(trace, "generate", {"safe_fail": False, "citations": citations})
        return {"answer": answer, "citations": citations, "safe_fail": False, "trace": trace}

    def _verify(self, state: PipelineState) -> PipelineState:
        trace = state["trace"]
        answer = state["answer"]
        citations = state["citations"]
        safe_fail = state["safe_fail"]

        if state["safe_fail"]:
            grounding = GroundingResult(status=GroundingStatus.UNSUPPORTED, reason="No evidence.")
            grounding_source = "pre-gated-safe-fail"
            prompt_version: str | None = None
        else:
            grounding_source = "fallback-no-reasoner"
            prompt_version: str | None = None
            evidence = [chunk.text[:240] for chunk in state.get("chunks", [])[:3]]

            if self._reasoner is not None:
                grounding, grounding_source, prompt_version = self._reasoner.assess_grounding(
                    answer=answer,
                    citations=citations,
                    evidence=evidence,
                )
            elif citations:
                grounding = GroundingResult(
                    status=GroundingStatus.SUPPORTED,
                    reason="Citations present.",
                )
            else:
                grounding = GroundingResult(status=GroundingStatus.PARTIAL, reason="No citations.")

            if grounding.status == GroundingStatus.UNSUPPORTED:
                answer = self._settings.safe_fail_message
                citations = []
                safe_fail = True

        payload = grounding.model_dump()
        payload["grounding_source"] = grounding_source
        payload["prompt_version"] = prompt_version

        self._event(trace, "verify_grounding", payload)
        return {
            "answer": answer,
            "citations": citations,
            "safe_fail": safe_fail,
            "grounding": grounding,
            "trace": trace,
        }

    def _finish(self, state: PipelineState) -> PipelineState:
        trace = state["trace"]
        trace.final_grounding_status = state["grounding"].status
        self._event(trace, "answer", {"safe_fail": state["safe_fail"]})
        return {"trace": trace}

    @staticmethod
    def _event(trace: PipelineTrace, stage: str, payload: dict[str, object]) -> None:
        trace.events.append(TraceEvent(stage=stage, payload=payload))

    @staticmethod
    def _needs_clarification(query: str) -> bool:
        normalized = " ".join(query.lower().split())
        vague_inputs = {
            "hi",
            "hello",
            "hey",
            "yo",
            "sup",
            "help",
            "?",
            "what",
            "why",
            "how",
        }
        if normalized in vague_inputs:
            return True
        return len(normalized) < 4
