from __future__ import annotations

from src.agent.tools import AgentTools
from src.core.config import Settings
from src.core.models import (
    GroundingResult,
    GroundingStatus,
    PipelineTrace,
    ResponseCategory,
    TraceEvent,
    ValidationResult,
    ValidationStatus,
)
from src.orchestration.graph_state import PipelineState
from src.services.reasoner import QueryReasoner


class PipelineNodes:
    def __init__(
        self,
        *,
        settings: Settings,
        tools: AgentTools,
        reasoner: QueryReasoner | None,
    ) -> None:
        if reasoner is None:
            raise ValueError("PipelineNodes requires a QueryReasoner instance.")
        self._settings = settings
        self._tools = tools
        self._reasoner = reasoner

    def summarize_history(self, state: PipelineState) -> PipelineState:
        history = state.get("history", [])
        conversation_summary = self._reasoner.summarize_conversation(history)
        return {"conversation_summary": conversation_summary}

    def rewrite_query(self, state: PipelineState) -> PipelineState:
        original_query = " ".join(state["query"].split())
        conversation_summary = state.get("conversation_summary", "")
        clarify_needed = True
        rewritten_query = original_query
        clarify_message = self._settings.clarification_message
        rewrite_source = "fallback-rule"
        clarity_reason = "Fallback clarification."
        clarity_prompt_version: str | None = None
        analysis_prompt_version: str | None = None
        analysis_source = "fallback-rule"
        analysis_error: str | None = None

        try:
            analysis, analysis_source = self._reasoner.analyze_query(
                query=original_query,
                conversation_summary=conversation_summary,
            )
            clarify_needed = not analysis.is_clear
            rewritten_query = analysis.rewritten_query or original_query
            clarify_message = analysis.clarification_needed or self._settings.clarification_message
            rewrite_source = analysis_source
            clarity_reason = (
                "Model requested clarification." if clarify_needed else "Model marked query clear."
            )
            clarity_prompt_version = analysis.prompt_version
            analysis_prompt_version = analysis.prompt_version
        except Exception as exc:
            analysis_error = str(exc)
        clarity_source = analysis_source
        trace = PipelineTrace(original_query=original_query, rewritten_query=rewritten_query)
        self._event(
            trace,
            "rewrite_query",
            {
                "original": original_query,
                "conversation_summary": conversation_summary,
                "rewritten": rewritten_query,
                "rewrite_source": rewrite_source,
                "prompt_version": analysis_prompt_version,
                "clarify_needed": clarify_needed,
                "clarify_reason": clarity_reason,
                "clarify_prompt_version": clarity_prompt_version,
                "clarity_source": clarity_source,
                "analysis_error": analysis_error,
            },
        )
        return {
            "original_query": original_query,
            "conversation_summary": conversation_summary,
            "rewritten_query": rewritten_query,
            "clarify_needed": clarify_needed,
            "clarify_message": clarify_message,
            "trace": trace,
        }

    def clarify(self, state: PipelineState) -> PipelineState:
        trace = state["trace"]
        answer = state.get("clarify_message") or self._settings.clarification_message
        response_source = "settings-default"
        prompt_version: str | None = None
        grounding = GroundingResult(
            status=GroundingStatus.UNSUPPORTED,
            reason="Query is too vague and requires clarification.",
        )
        self._event(
            trace,
            "clarify",
            {
                "message": answer,
                "response_category": ResponseCategory.CLARIFICATION,
                "response_source": response_source,
                "prompt_version": prompt_version,
            },
        )
        return {
            "answer": answer,
            "citations": [],
            "safe_fail": True,
            "grounding": grounding,
            "trace": trace,
        }

    def retrieve(self, state: PipelineState) -> PipelineState:
        trace = state["trace"]
        self._event(trace, "tool_search_chunks", {"query": state["rewritten_query"]})
        search_hits = self._tools.search_chunks(state["rewritten_query"], self._settings.retrieval_top_k)

        chunk_ids = [chunk.chunk_id for chunk in search_hits]
        self._event(trace, "tool_fetch_chunks_by_ids", {"chunk_ids": chunk_ids})
        fetched_chunks = self._tools.fetch_chunks_by_ids(chunk_ids)

        if fetched_chunks:
            score_by_id = {chunk.chunk_id: chunk.score for chunk in search_hits}
            chunks = [
                chunk.model_copy(update={"score": score_by_id.get(chunk.chunk_id, chunk.score)})
                for chunk in fetched_chunks
            ]
        else:
            chunks = search_hits

        self._event(trace, "retrieve", {"hits": [c.model_dump() for c in chunks]})
        return {"chunks": chunks, "trace": trace}

    def validate(self, state: PipelineState) -> PipelineState:
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
                    status=ValidationStatus.FAIL,
                    reason="Top score below threshold.",
                    confidence=top_score,
                )
            elif (
                len(chunks) >= 2
                and (chunks[0].score - chunks[1].score) < self._settings.ambiguity_margin
                and chunks[0].source != chunks[1].source
            ):
                validation = ValidationResult(
                    status=ValidationStatus.FAIL,
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
        self._event(trace, "validate", validation.model_dump())
        return {"validation": validation, "trace": trace}

    def generate(self, state: PipelineState) -> PipelineState:
        trace = state["trace"]
        if state["validation"].status == ValidationStatus.FAIL:
            answer, response_category, response_source, prompt_version = (
                self._generate_safe_fail(state)
            )
            self._event(
                trace,
                "generate",
                {
                    "safe_fail": True,
                    "answer": answer,
                    "response_category": response_category,
                    "response_source": response_source,
                    "prompt_version": prompt_version,
                },
            )
            return {"answer": answer, "citations": [], "safe_fail": True, "trace": trace}

        try:
            answer, citations, generation_source, prompt_version = self._generate_supported(state)
        except Exception as exc:
            forced_state = {
                **state,
                "validation": ValidationResult(
                    status=ValidationStatus.FAIL,
                    reason="Generated answer could not be grounded in retrieved evidence.",
                    confidence=state["validation"].confidence,
                ),
            }
            answer, response_category, response_source, prompt_version = (
                self._generate_safe_fail(forced_state)
            )
            self._event(
                trace,
                "generate",
                {
                    "safe_fail": True,
                    "answer": answer,
                    "response_category": response_category,
                    "response_source": response_source,
                    "prompt_version": prompt_version,
                    "fallback_reason": str(exc),
                },
            )
            return {"answer": answer, "citations": [], "safe_fail": True, "trace": trace}

        self._event(
            trace,
            "generate",
            {
                "safe_fail": False,
                "citations": citations,
                "generation_source": generation_source,
                "prompt_version": prompt_version,
            },
        )
        return {"answer": answer, "citations": citations, "safe_fail": False, "trace": trace}

    def verify(self, state: PipelineState) -> PipelineState:
        trace = state["trace"]
        answer = state["answer"]
        citations = state["citations"]
        safe_fail = state["safe_fail"]
        coverage_missing_terms: list[str] = []
        coverage_source: str | None = None
        coverage_prompt_version: str | None = None

        if state["safe_fail"]:
            grounding = GroundingResult(status=GroundingStatus.UNSUPPORTED, reason="No evidence.")
            grounding_source = "pre-gated-safe-fail"
            prompt_version: str | None = None
        else:
            evidence = [chunk.text[:240] for chunk in state.get("chunks", [])[:3]]
            grounding, grounding_source, prompt_version = self._reasoner.assess_grounding(
                answer=answer,
                citations=citations,
                evidence=evidence,
            )

            (
                grounding,
                grounding_source,
                prompt_version,
                coverage_missing_terms,
                coverage_source,
                coverage_prompt_version,
            ) = self._apply_coverage_gate(
                state=state,
                answer=answer,
                grounding=grounding,
                grounding_source=grounding_source,
                prompt_version=prompt_version,
            )

            if grounding.status == GroundingStatus.UNSUPPORTED:
                answer = self._settings.safe_fail_message
                citations = []
                safe_fail = True

        payload = grounding.model_dump()
        payload["grounding_source"] = grounding_source
        payload["prompt_version"] = prompt_version
        payload["coverage_missing_terms"] = coverage_missing_terms
        payload["coverage_source"] = coverage_source
        payload["coverage_prompt_version"] = coverage_prompt_version
        payload["reason_source"] = "model"
        payload["reason_prompt_version"] = None

        self._event(trace, "verify_grounding", payload)
        return {
            "answer": answer,
            "citations": citations,
            "safe_fail": safe_fail,
            "grounding": grounding,
            "trace": trace,
        }

    def finish(self, state: PipelineState) -> PipelineState:
        trace = state["trace"]
        trace.final_grounding_status = state["grounding"].status
        self._event(trace, "answer", {"safe_fail": state["safe_fail"]})
        return {"trace": trace}

    @staticmethod
    def _event(trace: PipelineTrace, stage: str, payload: dict[str, object]) -> None:
        trace.events.append(TraceEvent(stage=stage, payload=payload))

    def _generate_safe_fail(
        self, state: PipelineState
    ) -> tuple[str, ResponseCategory, str, str | None]:
        answer = self._settings.safe_fail_message
        response_category = ResponseCategory.SAFE_FAIL
        response_source = "settings-default"
        prompt_version: str | None = None
        no_hits = state["validation"].reason.strip().lower().startswith("no retrieval hits")
        if no_hits:
            return answer, response_category, response_source, prompt_version
        return answer, response_category, response_source, prompt_version

    def _generate_supported(
        self, state: PipelineState
    ) -> tuple[str, list[str], str, str | None]:
        chunks = state.get("chunks", [])
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        answer, citation_chunk_ids, generation_source, prompt_version = (
            self._reasoner.synthesize_answer(query=state["original_query"], chunks=chunks)
        )
        citations = [
            f"{chunk_by_id[chunk_id].source}#{chunk_id}"
            for chunk_id in citation_chunk_ids
            if chunk_id in chunk_by_id
        ]
        if len(citations) != len(citation_chunk_ids):
            raise ValueError(
                "synthesize_answer returned citation_chunk_ids not present in evidence."
            )
        return answer, citations, generation_source, prompt_version

    def _apply_coverage_gate(
        self,
        *,
        state: PipelineState,
        answer: str,
        grounding: GroundingResult,
        grounding_source: str,
        prompt_version: str | None,
    ) -> tuple[GroundingResult, str, str | None, list[str], str | None, str | None]:
        coverage_missing_terms: list[str] = []
        coverage_source: str | None = None
        coverage_prompt_version: str | None = None

        evidence_for_coverage = state.get("chunks", [])
        (
            insufficient_coverage,
            coverage_missing_terms,
            coverage_source,
            coverage_prompt_version,
        ) = self._reasoner.detect_insufficient_coverage(
            query=state["original_query"],
            answer=answer,
            chunks=evidence_for_coverage,
        )

        if grounding.status == GroundingStatus.SUPPORTED and insufficient_coverage:
            grounding = GroundingResult(
                status=GroundingStatus.UNSUPPORTED,
                reason=(
                    "Evidence does not cover key query terms: " + ", ".join(coverage_missing_terms)
                ),
            )
            grounding_source = "llm-insufficient-coverage"
            prompt_version = None

        return (
            grounding,
            grounding_source,
            prompt_version,
            coverage_missing_terms,
            coverage_source,
            coverage_prompt_version,
        )

