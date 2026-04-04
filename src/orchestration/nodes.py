from __future__ import annotations

import re
from collections import Counter

from src.agent.tools import AgentTools
from src.core.config import Settings
from src.core.models import (
    EvidenceChunk,
    GroundingResult,
    GroundingStatus,
    PipelineTrace,
    QueryRewriteOutput,
    ResponseCategory,
    TraceEvent,
    ValidationResult,
    ValidationStatus,
)
from src.orchestration.graph_state import PipelineState
from src.services.reasoner import QueryReasoner
from src.services.response_policy import ResponsePolicy


class PipelineNodes:
    def __init__(
        self,
        *,
        settings: Settings,
        tools: AgentTools,
        reasoner: QueryReasoner | None,
        response_policy: ResponsePolicy | None,
    ) -> None:
        self._settings = settings
        self._tools = tools
        self._reasoner = reasoner
        self._response_policy = response_policy

    def understand(self, state: PipelineState) -> PipelineState:
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

    def clarify(self, state: PipelineState) -> PipelineState:
        trace = state["trace"]
        answer = self._settings.clarification_message
        response_source = "settings-default"
        prompt_version: str | None = None
        if self._response_policy is not None:
            answer, prompt_version = self._response_policy.render(
                category=ResponseCategory.CLARIFICATION,
                query=state["original_query"],
                reason="Query is too vague and needs clarification.",
                evidence_count=0,
            )
            response_source = "llm-policy"
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
        chunks = self._tools.search_chunks(state["rewritten_query"], self._settings.retrieval_top_k)
        stage = "retrieve_retry" if state["retry_count"] > 0 else "retrieve"
        self._event(trace, stage, {"hits": [c.model_dump() for c in chunks]})
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

    def retry(self, state: PipelineState) -> PipelineState:
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

    def retry_exhausted(self, state: PipelineState) -> PipelineState:
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

    def hydrate(self, state: PipelineState) -> PipelineState:
        trace = state["trace"]
        selected_ids = [chunk.chunk_id for chunk in state["chunks"][:2]]
        self._event(trace, "tool_fetch_chunks_by_ids", {"chunk_ids": selected_ids})
        fetched = self._tools.fetch_chunks_by_ids(selected_ids)
        evidence_chunks = fetched if fetched else state["chunks"][:2]
        self._event(
            trace,
            "hydrate",
            {
                "selected_ids": selected_ids,
                "fetched_ids": [chunk.chunk_id for chunk in fetched],
                "used_fallback": not bool(fetched),
            },
        )
        return {"evidence_chunks": evidence_chunks, "trace": trace}

    def generate(self, state: PipelineState) -> PipelineState:
        trace = state["trace"]
        if state["validation"].status == ValidationStatus.FAIL:
            answer = self._settings.safe_fail_message
            response_category = ResponseCategory.SAFE_FAIL
            response_source = "settings-default"
            prompt_version: str | None = None
            if state["validation"].reason.startswith("Retry budget exhausted"):
                response_category = ResponseCategory.RETRY_EXHAUSTED
            if self._response_policy is not None:
                answer, prompt_version = self._response_policy.render(
                    category=response_category,
                    query=state["original_query"],
                    reason=state["validation"].reason,
                    evidence_count=len(state.get("chunks", [])),
                )
                response_source = "llm-policy"

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

        chunks = state.get("evidence_chunks") or state["chunks"][:2]
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        generation_source = "fallback-no-reasoner"
        prompt_version: str | None = None

        if self._reasoner is not None:
            answer, citation_chunk_ids, generation_source, prompt_version = (
                self._reasoner.synthesize_answer(query=state["original_query"], chunks=chunks)
            )
            citations = [
                f"{chunk_by_id[chunk_id].source}#{chunk_id}"
                for chunk_id in citation_chunk_ids
                if chunk_id in chunk_by_id
            ]
            if not citations:
                citations = [f"{chunk.source}#{chunk.chunk_id}" for chunk in chunks]
        else:
            citations = [f"{chunk.source}#{chunk.chunk_id}" for chunk in chunks]
            answer_lines = [f"- [{chunk.chunk_id}] {chunk.text[:220]}" for chunk in chunks]
            answer = "Question: " + state["original_query"] + "\n\n" + "\n".join(answer_lines)

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

            evidence_for_coverage = state.get("evidence_chunks") or state.get("chunks", [])
            insufficient_coverage, coverage_missing_terms = self._has_insufficient_coverage(
                answer=answer,
                query=state["original_query"],
                chunks=evidence_for_coverage,
            )
            if grounding.status == GroundingStatus.SUPPORTED and insufficient_coverage:
                grounding = GroundingResult(
                    status=GroundingStatus.UNSUPPORTED,
                    reason=(
                        "Evidence does not cover key query terms: "
                        + ", ".join(coverage_missing_terms)
                    ),
                )
                grounding_source = "heuristic-insufficient-coverage"
                prompt_version = None

            reason_source = "model-or-fallback"
            reason_prompt_version: str | None = None
            if self._response_policy is not None:
                natural_reason, reason_prompt_version = self._response_policy.render(
                    category=ResponseCategory.GROUNDING_REASON,
                    query=state["original_query"],
                    reason=grounding.reason,
                    evidence_count=len(evidence),
                )
                grounding.reason = natural_reason
                reason_source = "llm-policy"

            if grounding.status == GroundingStatus.UNSUPPORTED:
                answer = self._settings.safe_fail_message
                citations = []
                safe_fail = True
                if self._response_policy is not None:
                    answer, _ = self._response_policy.render(
                        category=ResponseCategory.SAFE_FAIL,
                        query=state["original_query"],
                        reason=grounding.reason,
                        evidence_count=len(evidence),
                    )

        payload = grounding.model_dump()
        payload["grounding_source"] = grounding_source
        payload["prompt_version"] = prompt_version
        payload["coverage_missing_terms"] = coverage_missing_terms
        payload["reason_source"] = reason_source if "reason_source" in locals() else None
        payload["reason_prompt_version"] = (
            reason_prompt_version if "reason_prompt_version" in locals() else None
        )

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

    @classmethod
    def _has_insufficient_coverage(
        cls,
        *,
        answer: str,
        query: str,
        chunks: list[EvidenceChunk],
    ) -> tuple[bool, list[str]]:
        answer_lower = answer.lower()
        no_evidence_markers = (
            "no information",
            "does not mention",
            "not mentioned",
            "not provided",
            "cannot find",
            "not in the evidence",
        )
        denial_pattern = re.search(
            r"\bno\b.*\b(described|evidence|information|mention|found)\b",
            answer_lower,
        )
        has_denial_signal = any(marker in answer_lower for marker in no_evidence_markers) or bool(
            denial_pattern
        )
        if not has_denial_signal:
            return False, []

        query_terms = cls._key_terms(query)
        if not query_terms:
            return True, []

        evidence_text = " ".join(chunk.text.lower() for chunk in chunks)
        missing = [term for term in query_terms if term not in evidence_text]
        if missing:
            return True, missing
        return True, []

    @staticmethod
    def _key_terms(text: str) -> list[str]:
        stopwords = {
            "what",
            "which",
            "where",
            "when",
            "why",
            "how",
            "does",
            "do",
            "is",
            "are",
            "the",
            "a",
            "an",
            "of",
            "to",
            "in",
            "on",
            "for",
            "and",
            "or",
            "use",
            "uses",
            "used",
            "method",
            "methods",
        }
        terms = re.findall(r"[a-zA-Z]{4,}", text.lower())
        deduped: list[str] = []
        for term in terms:
            if term in stopwords:
                continue
            if term not in deduped:
                deduped.append(term)
        return deduped
