from __future__ import annotations

from src.agent.tools import AgentTools
from src.core.config import Settings
from src.core.models import (
    AgentObservation,
    AgentThought,
    EvidenceChunk,
    GroundingResult,
    GroundingStatus,
    PipelineTrace,
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
        return {
            "conversation_summary": conversation_summary,
            "rewritten_queries": [],
            "retrieval_keys": [],
            "context_summary": "",
            "compress_needed": False,
            "iteration_count": 0,
            "tool_call_count": 0,
            "retrieval_attempted": False,
            "limit_exceeded": False,
        }

    def rewrite_query(self, state: PipelineState) -> PipelineState:
        original_query = " ".join(state["query"].split())
        conversation_summary = state.get("conversation_summary", "")
        clarify_needed = True
        rewritten_query = original_query
        rewritten_queries = [original_query]
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
            rewritten_questions = [q.strip() for q in analysis.questions if isinstance(q, str) and q.strip()]
            rewritten_queries = rewritten_questions or [rewritten_query]
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
                "rewritten_questions": rewritten_queries,
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
            "rewritten_queries": rewritten_queries,
            "clarify_needed": clarify_needed,
            "clarify_message": clarify_message,
            "trace": trace,
        }

    def clarify(self, state: PipelineState) -> PipelineState:
        del state
        return {}

    def detect_query_type(self, state: PipelineState) -> PipelineState:
        """
        Detect if the query is about the conversation itself (meta-query).
        If so, generate answer from conversation history instead of document retrieval.
        """
        trace = state["trace"]
        rewritten_query = state.get("rewritten_query", "")
        conversation_summary = state.get("conversation_summary", "")
        history = state.get("history", [])
        
        is_conversation_query, confidence = self._reasoner.detect_conversation_query(
            query=rewritten_query
        )
        
        self._event(
            trace,
            "detect_query_type",
            {
                "is_conversation_query": is_conversation_query,
                "confidence": confidence,
                "rewritten_query": rewritten_query,
            },
        )
        
        # Threshold: 0.5 (more lenient) - prefer conversation interpretation if uncertain
        if not is_conversation_query or confidence < 0.5:
            return {"is_conversation_query": False, "trace": trace}
        
        # This IS a conversation query - answer from history
        # Even with minimal history, provide the best answer we can
        if not history or len(history) <= 1:
            answer = (
                "This is our first interaction, so I don't have prior conversation history to reference. "
                "Feel free to ask me about the uploaded documents instead!"
            )
            grounding = GroundingResult(
                status=GroundingStatus.SUPPORTED,
                reason="Answered meta-query (first interaction, no prior history).",
            )
            return {
                "is_conversation_query": True,
                "answer": answer,
                "citations": [],
                "safe_fail": False,
                "grounding": grounding,
                "trace": trace,
            }
        
        # Synthesize from available history
        try:
            history_text = "\n".join(
                f"{item.get('role', 'user').title()}: {item.get('content', '')}"
                for item in history[-10:]  # Use last 10 turns
            )
            answer, _, _, _ = self._reasoner.synthesize_answer(
                query=rewritten_query,
                chunks=[
                    EvidenceChunk(
                        chunk_id="conversation",
                        text=history_text,
                        source="conversation_history",
                        score=1.0,
                    )
                ],
            )
            grounding = GroundingResult(
                status=GroundingStatus.SUPPORTED,
                reason="Answered from conversation history.",
            )
        except Exception as exc:
            answer = f"Based on our conversation so far: {history_text[:200]}..."
            grounding = GroundingResult(
                status=GroundingStatus.SUPPORTED,
                reason="Answered from conversation history.",
            )
        
        return {
            "is_conversation_query": True,
            "answer": answer,
            "citations": [],
            "safe_fail": False,
            "grounding": grounding,
            "trace": trace,
        }

    def agent_initialize(self, state: PipelineState) -> PipelineState:
        trace = state["trace"]
        self._event(
            trace,
            "agent_initialize",
            {"mode": "enabled", "max_iterations": self._settings.agent_max_iterations},
        )
        return {
            "agent_iterations": 0,
            "evidence_quality_score": 0.0,
            "agent_thoughts": [],
            "agent_observations": [],
            "selected_action": "search_documents",
            "trace": trace,
        }

    def agent_think(self, state: PipelineState) -> PipelineState:
        trace = state["trace"]
        agent_iterations = state.get("agent_iterations", 0) + 1
        thought = AgentThought(
            reasoning="Gather or improve evidence quality using document search.",
            recommended_action="search_documents",
            confidence=0.6,
        )
        thoughts = [*state.get("agent_thoughts", []), thought]
        self._event(
            trace,
            "agent_think",
            {
                "iteration": agent_iterations,
                "selected_action": thought.recommended_action,
                "confidence": thought.confidence,
            },
        )
        return {
            "agent_iterations": agent_iterations,
            "selected_action": thought.recommended_action,
            "agent_thoughts": thoughts,
            "trace": trace,
        }

    def agent_act(self, state: PipelineState) -> PipelineState:
        trace = state["trace"]
        selected_action = state.get("selected_action", "search_documents")

        if selected_action == "search_documents":
            retrieval_update = self.retrieve(state)
            chunks = retrieval_update.get("chunks", [])
            observation = AgentObservation(
                action=selected_action,
                success=True,
                quality_score=0.0,
                message=f"Retrieved {len(chunks)} chunk(s).",
            )
            observations = [*state.get("agent_observations", []), observation]
            self._event(
                trace,
                "agent_act",
                {"action": selected_action, "chunk_count": len(chunks)},
            )
            return {
                **retrieval_update,
                "agent_observations": observations,
                "trace": trace,
            }

        observation = AgentObservation(
            action=selected_action,
            success=False,
            quality_score=0.0,
            message="Unsupported action.",
        )
        observations = [*state.get("agent_observations", []), observation]
        self._event(trace, "agent_act", {"action": selected_action, "unsupported": True})
        return {"agent_observations": observations, "trace": trace}

    def agent_reflect(self, state: PipelineState) -> PipelineState:
        trace = state["trace"]
        chunks = state.get("chunks", [])
        if not chunks:
            quality_score = 0.0
        else:
            top_score = max(chunk.score for chunk in chunks)
            avg_score = sum(chunk.score for chunk in chunks) / len(chunks)
            quality_score = round((top_score * 0.7) + (avg_score * 0.3), 4)

        self._event(
            trace,
            "agent_reflect",
            {
                "quality_score": quality_score,
                "threshold": self._settings.agent_evidence_quality_threshold,
                "agent_iterations": state.get("agent_iterations", 0),
            },
        )
        return {"evidence_quality_score": quality_score, "trace": trace}

    def retrieve(self, state: PipelineState) -> PipelineState:
        trace = state["trace"]
        rewritten_queries = state.get("rewritten_queries") or [state["rewritten_query"]]

        deduped_queries: list[str] = []
        seen_queries: set[str] = set()
        for query in rewritten_queries:
            key = query.strip().lower()
            if not key or key in seen_queries:
                continue
            seen_queries.add(key)
            deduped_queries.append(query)

        merged_hits: list[EvidenceChunk] = []
        score_by_id: dict[str, float] = {}
        for query in deduped_queries:
            self._event(trace, "tool_search_chunks", {"query": query})
            search_hits = self._tools.search_chunks(query, self._settings.retrieval_top_k)
            for hit in search_hits:
                if hit.chunk_id not in score_by_id:
                    merged_hits.append(hit)
                    score_by_id[hit.chunk_id] = hit.score

        chunk_ids = [chunk.chunk_id for chunk in merged_hits]
        self._event(trace, "tool_fetch_chunks_by_ids", {"chunk_ids": chunk_ids})
        fetched_chunks = self._tools.fetch_chunks_by_ids(chunk_ids)

        if fetched_chunks:
            chunks = [
                chunk.model_copy(update={"score": score_by_id.get(chunk.chunk_id, chunk.score)})
                for chunk in fetched_chunks
            ]
        else:
            chunks = merged_hits

        self._event(trace, "retrieve", {"hits": [c.model_dump() for c in chunks]})
        retrieval_keys = [
            *(f"search::{query}" for query in deduped_queries),
            *(f"chunk::{chunk_id}" for chunk_id in chunk_ids),
        ]
        iteration_count = state.get("iteration_count", 0) + 1
        tool_call_count = state.get("tool_call_count", 0) + len(deduped_queries) + 1
        return {
            "chunks": chunks,
            "trace": trace,
            "retrieval_keys": retrieval_keys,
            "iteration_count": iteration_count,
            "tool_call_count": tool_call_count,
            "retrieval_attempted": True,
        }

    def should_compress_context(self, state: PipelineState) -> PipelineState:
        trace = state["trace"]
        chunks = state.get("chunks", [])
        existing_summary = state.get("context_summary", "")

        # Lightweight token estimate to mirror reference threshold-gate behavior.
        current_chars = sum(len(chunk.text) for chunk in chunks)
        summary_chars = len(existing_summary)
        estimated_tokens = (current_chars + summary_chars) // 4
        max_allowed = self._settings.context_compression_base_threshold + int(
            (summary_chars // 4) * self._settings.context_compression_growth_factor
        )
        compress_needed = estimated_tokens > max_allowed
        iteration_count = state.get("iteration_count", 0)
        tool_call_count = state.get("tool_call_count", 0)
        limit_exceeded = (
            iteration_count >= self._settings.agent_max_iterations
            or tool_call_count > self._settings.agent_max_tool_calls
        )

        self._event(
            trace,
            "should_compress_context",
            {
                "estimated_tokens": estimated_tokens,
                "max_allowed": max_allowed,
                "compress_needed": compress_needed,
                "iteration_count": iteration_count,
                "tool_call_count": tool_call_count,
                "limit_exceeded": limit_exceeded,
            },
        )
        return {
            "compress_needed": compress_needed,
            "limit_exceeded": limit_exceeded,
            "trace": trace,
        }

    def compress_context(self, state: PipelineState) -> PipelineState:
        trace = state["trace"]
        chunks = state.get("chunks", [])
        existing_summary = state.get("context_summary", "")

        summary_input: list[dict[str, str]] = []
        if existing_summary.strip():
            summary_input.append(
                {
                    "role": "assistant",
                    "content": f"Prior compressed context: {existing_summary.strip()}",
                }
            )
        for chunk in chunks[:8]:
            summary_input.append(
                {
                    "role": "assistant",
                    "content": f"[{chunk.source}#{chunk.chunk_id}] {chunk.text[:320]}",
                }
            )

        context_summary = self._reasoner.summarize_conversation(summary_input)
        self._event(
            trace,
            "compress_context",
            {
                "summary_length": len(context_summary),
                "evidence_count": len(chunks),
            },
        )
        return {"context_summary": context_summary, "trace": trace}

    def fallback_response(self, state: PipelineState) -> PipelineState:
        trace = state["trace"]
        answer = self._settings.safe_fail_message
        grounding = GroundingResult(
            status=GroundingStatus.UNSUPPORTED,
            reason="Fallback triggered by agent loop limits.",
        )
        self._event(
            trace,
            "fallback_response",
            {
                "iteration_count": state.get("iteration_count", 0),
                "tool_call_count": state.get("tool_call_count", 0),
            },
        )
        return {
            "answer": answer,
            "citations": [],
            "safe_fail": True,
            "grounding": grounding,
            "trace": trace,
        }

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
            answer, response_source, prompt_version = (
                self._generate_safe_fail(state)
            )
            self._event(
                trace,
                "generate",
                {
                    "safe_fail": True,
                    "answer": answer,
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
            answer, response_source, prompt_version = (
                self._generate_safe_fail(forced_state)
            )
            self._event(
                trace,
                "generate",
                {
                    "safe_fail": True,
                    "answer": answer,
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

        if state["safe_fail"]:
            grounding = GroundingResult(status=GroundingStatus.UNSUPPORTED, reason="No evidence.")
            grounding_source = "pre-gated-safe-fail"
            prompt_version: str | None = None
        else:
            chunks = state.get("chunks", [])
            chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}

            # Grounding should inspect the same evidence used for cited claims.
            cited_chunk_ids = [
                citation.split("#", 1)[1]
                for citation in citations
                if isinstance(citation, str) and "#" in citation
            ]
            cited_chunks = [chunk_by_id[chunk_id] for chunk_id in cited_chunk_ids if chunk_id in chunk_by_id]

            if cited_chunks:
                evidence_chunks = cited_chunks
            else:
                evidence_chunks = chunks[:8]

            evidence = [chunk.text[:500] for chunk in evidence_chunks[:6]]
            grounding, grounding_source, prompt_version = self._reasoner.assess_grounding(
                answer=answer,
                citations=citations,
                evidence=evidence,
            )

            # Revert to strict rejection: reject if unsupported OR refusal
            # Keeps unanswerable questions correctly rejected
            if grounding.status == GroundingStatus.UNSUPPORTED or grounding.is_refusal:
                answer = self._settings.safe_fail_message
                citations = []
                safe_fail = True

        payload = grounding.model_dump()
        payload["grounding_source"] = grounding_source
        payload["prompt_version"] = prompt_version
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
        trace.agent_iterations_used = state.get("agent_iterations", 0)
        trace.agent_thought_count = len(state.get("agent_thoughts", []))
        self._event(trace, "answer", {"safe_fail": state["safe_fail"]})
        return {"trace": trace}

    @staticmethod
    def _event(trace: PipelineTrace, stage: str, payload: dict[str, object]) -> None:
        trace.events.append(TraceEvent(stage=stage, payload=payload))

    def _generate_safe_fail(
        self, state: PipelineState
    ) -> tuple[str, str, str | None]:
        del state
        answer = self._settings.safe_fail_message
        response_source = "settings-default"
        prompt_version: str | None = None
        return answer, response_source, prompt_version

    def _generate_supported(
        self, state: PipelineState
    ) -> tuple[str, list[str], str, str | None]:
        chunks = state.get("chunks", [])
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        query_for_generation = state.get("original_query") or state.get("rewritten_query") or ""
        answer, citation_chunk_ids, generation_source, prompt_version = (
            self._reasoner.synthesize_answer(query=query_for_generation, chunks=chunks)
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


