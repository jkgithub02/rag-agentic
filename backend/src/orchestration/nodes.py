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
    QueryComplexity,
    SubQueryStatus,
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
        progress_callback: object | None = None,
    ) -> None:
        if reasoner is None:
            raise ValueError("PipelineNodes requires a QueryReasoner instance.")
        self._settings = settings
        self._tools = tools
        self._reasoner = reasoner
        self._progress_callback = progress_callback

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

        detect_conv = getattr(self._reasoner, "detect_conversation_query", None)
        if callable(detect_conv):
            try:
                is_conversation_query, confidence = detect_conv(query=rewritten_query)
            except Exception:
                is_conversation_query, confidence = (False, 0.0)
        else:
            is_conversation_query, confidence = (False, 0.0)

        detect_complexity = getattr(self._reasoner, "detect_query_complexity", None)
        if callable(detect_complexity):
            try:
                query_complexity = detect_complexity(
                    query=rewritten_query,
                    conversation_summary=conversation_summary,
                )
            except Exception:
                query_complexity = QueryComplexity.MODERATE
        else:
            query_complexity = QueryComplexity.MODERATE

        complexity_value = (
            query_complexity.value if isinstance(query_complexity, QueryComplexity) else str(query_complexity)
        )
        
        self._event(
            trace,
            "detect_query_type",
            {
                "is_conversation_query": is_conversation_query,
                "confidence": confidence,
                "rewritten_query": rewritten_query,
                "query_complexity": complexity_value,
            },
        )
        
        # Threshold: 0.5 (more lenient) - prefer conversation interpretation if uncertain
        if not is_conversation_query or confidence < 0.5:
            return {
                "is_conversation_query": False,
                "query_complexity": complexity_value,
                "trace": trace,
            }
        
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
            "query_complexity": complexity_value,
            "answer": answer,
            "citations": [],
            "safe_fail": False,
            "grounding": grounding,
            "trace": trace,
        }

    _MULTI_PART_PATTERNS = (
        " and how ", " and what ", " and then ", ", then ",
        " compare ", " differences between ", " trace how ",
        " evolved from ", " contrast ",
    )

    def prepare_decomposition(self, state: PipelineState) -> PipelineState:
        trace = state["trace"]
        rewritten_query = state.get("rewritten_query", "")
        rewritten_queries = state.get("rewritten_queries") or [rewritten_query]
        complexity = str(state.get("query_complexity", "moderate")).lower()

        # Heuristic: force decomposition for multi-part queries
        query_lower = rewritten_query.lower()
        force_decompose = any(p in query_lower for p in self._MULTI_PART_PATTERNS)

        should_decompose = (
            self._settings.enable_query_decomposition
            and (complexity == "complex" or force_decompose)
            and bool(rewritten_query.strip())
        )

        if not should_decompose:
            subquery_statuses = [SubQueryStatus(query=q) for q in rewritten_queries]
            self._event(
                trace,
                "prepare_decomposition",
                {"applied": False, "reason": "disabled-or-not-complex", "query_count": len(rewritten_queries)},
            )
            return {"rewritten_queries": rewritten_queries, "subquery_statuses": subquery_statuses, "trace": trace}

        decompose = getattr(self._reasoner, "decompose_query_lightly", None)
        if not callable(decompose):
            subquery_statuses = [SubQueryStatus(query=q) for q in rewritten_queries]
            self._event(
                trace,
                "prepare_decomposition",
                {"applied": False, "reason": "reasoner-missing-method", "query_count": len(rewritten_queries)},
            )
            return {"rewritten_queries": rewritten_queries, "subquery_statuses": subquery_statuses, "trace": trace}

        try:
            sub_queries = decompose(
                query=rewritten_query,
                conversation_summary=state.get("conversation_summary"),
            )
        except Exception as exc:
            subquery_statuses = [SubQueryStatus(query=q) for q in rewritten_queries]
            self._event(
                trace,
                "prepare_decomposition",
                {
                    "applied": False,
                    "reason": "decomposition-error",
                    "error": str(exc),
                    "query_count": len(rewritten_queries),
                },
            )
            return {"rewritten_queries": rewritten_queries, "subquery_statuses": subquery_statuses, "trace": trace}

        cleaned: list[str] = []
        seen_lower: set[str] = set()
        for item in sub_queries:
            if not isinstance(item, str):
                continue
            candidate = item.strip()
            if not candidate:
                continue
            key = candidate.lower()
            if key in seen_lower:
                continue
            seen_lower.add(key)
            cleaned.append(candidate)

        next_queries = cleaned or rewritten_queries
        subquery_statuses = [
            SubQueryStatus(query=q) for q in next_queries
        ]
        self._event(
            trace,
            "prepare_decomposition",
            {
                "applied": bool(cleaned),
                "original_query": rewritten_query,
                "query_count": len(next_queries),
                "queries": next_queries,
            },
        )
        return {"rewritten_queries": next_queries, "subquery_statuses": subquery_statuses, "trace": trace}

    def agent_initialize(self, state: PipelineState) -> PipelineState:
        trace = state["trace"]
        subquery_statuses = state.get("subquery_statuses") or [
            SubQueryStatus(query=q) for q in (state.get("rewritten_queries") or [state.get("rewritten_query", "")])
        ]
        self._event(
            trace,
            "agent_initialize",
            {
                "mode": "enabled",
                "max_iterations": self._settings.agent_max_iterations,
                "subquery_count": len(subquery_statuses),
            },
        )
        return {
            "agent_iterations": 0,
            "evidence_quality_score": 0.0,
            "agent_thoughts": [],
            "agent_observations": [],
            "selected_action": "search_documents",
            "subquery_statuses": subquery_statuses,
            "target_subquery_index": 0,
            "trace": trace,
        }

    def agent_think(self, state: PipelineState) -> PipelineState:
        trace = state["trace"]
        agent_iterations = state.get("agent_iterations", 0) + 1
        subquery_statuses = state.get("subquery_statuses") or []
        sq_dicts = [sq.model_dump() for sq in subquery_statuses] if subquery_statuses else None

        planner = getattr(self._reasoner, "plan_agent_step", None)
        if callable(planner):
            try:
                last_observation = None
                observations = state.get("agent_observations", [])
                if observations:
                    last_observation = observations[-1].message
                thought = planner(
                    query=state.get("original_query") or state.get("rewritten_query") or "",
                    conversation_summary=state.get("conversation_summary"),
                    rewritten_queries=state.get("rewritten_queries") or [],
                    evidence_quality_score=state.get("evidence_quality_score", 0.0),
                    chunk_count=len(state.get("chunks", [])),
                    agent_iterations=agent_iterations,
                    max_iterations=self._settings.agent_max_iterations,
                    last_observation=last_observation,
                    subquery_statuses=sq_dicts,
                )
            except Exception:
                thought = AgentThought(
                    reasoning="Planner failed; defaulting to document search.",
                    recommended_action="search_documents",
                    confidence=0.0,
                )
        else:
            # Fallback: target the first pending/weakest subquery
            target_idx = self._pick_weakest_subquery(subquery_statuses)
            thought = AgentThought(
                reasoning="Gather or improve evidence quality using document search.",
                recommended_action="search_documents",
                confidence=0.6,
                target_subquery_index=target_idx,
            )

        # Resolve target subquery index
        target_index = thought.target_subquery_index
        if target_index is None and subquery_statuses:
            target_index = self._pick_weakest_subquery(subquery_statuses)

        thoughts = [*state.get("agent_thoughts", []), thought]
        self._event(
            trace,
            "agent_think",
            {
                "iteration": agent_iterations,
                "reasoning": thought.reasoning,
                "selected_action": thought.recommended_action,
                "confidence": thought.confidence,
                "target_subquery_index": target_index,
            },
        )
        return {
            "agent_iterations": agent_iterations,
            "selected_action": thought.recommended_action,
            "target_subquery_index": target_index if target_index is not None else 0,
            "agent_thoughts": thoughts,
            "trace": trace,
        }

    def agent_act(self, state: PipelineState) -> PipelineState:
        trace = state["trace"]
        selected_action = state.get("selected_action", "search_documents")
        subquery_statuses = list(state.get("subquery_statuses") or [])
        target_idx = state.get("target_subquery_index", 0)

        if selected_action == "finalize":
            observation = AgentObservation(
                action=selected_action,
                success=True,
                quality_score=state.get("evidence_quality_score", 0.0),
                message="Planner decided evidence is sufficient to finalize.",
            )
            observations = [*state.get("agent_observations", []), observation]
            self._event(trace, "agent_act", {"action": selected_action, "finalize": True})
            return {
                "agent_observations": observations,
                "chunks": state.get("chunks", []),
                "subquery_statuses": subquery_statuses,
                "trace": trace,
            }

        if selected_action == "search_documents":
            # Use targeted subquery if available, otherwise fall back to full retrieval
            targeted_query = None
            if subquery_statuses and 0 <= target_idx < len(subquery_statuses):
                targeted_query = subquery_statuses[target_idx].query

            if targeted_query:
                # Targeted single-subquery retrieval
                targeted_state = dict(state)
                targeted_state["rewritten_queries"] = [targeted_query]
                retrieval_update = self.retrieve(targeted_state)
            else:
                retrieval_update = self.retrieve(state)

            new_chunks = retrieval_update.get("chunks", [])

            # Merge new chunks into existing chunks, deduplicating by chunk_id
            existing_chunks = list(state.get("chunks") or [])
            existing_ids = {c.chunk_id for c in existing_chunks}
            for chunk in new_chunks:
                if chunk.chunk_id not in existing_ids:
                    existing_chunks.append(chunk)
                    existing_ids.add(chunk.chunk_id)

            # Update targeted subquery status
            if subquery_statuses and 0 <= target_idx < len(subquery_statuses):
                sq = subquery_statuses[target_idx]
                new_chunk_ids = [c.chunk_id for c in new_chunks if c.chunk_id not in set(sq.chunk_ids)]
                subquery_statuses[target_idx] = sq.model_copy(update={
                    "status": "retrieved",
                    "chunk_ids": [*sq.chunk_ids, *new_chunk_ids],
                })

            observation = AgentObservation(
                action=selected_action,
                success=True,
                quality_score=0.0,
                message=f"Retrieved {len(new_chunks)} chunk(s) for subquery[{target_idx}].",
            )
            observations = [*state.get("agent_observations", []), observation]
            self._event(
                trace,
                "agent_act",
                {
                    "action": selected_action,
                    "chunk_count": len(new_chunks),
                    "total_chunks": len(existing_chunks),
                    "target_subquery_index": target_idx,
                },
            )
            return {
                **retrieval_update,
                "chunks": existing_chunks,
                "agent_observations": observations,
                "subquery_statuses": subquery_statuses,
                "trace": trace,
            }

        if selected_action == "web_search":
            if not self._settings.web_search_enabled:
                observation = AgentObservation(
                    action=selected_action,
                    success=False,
                    quality_score=0.0,
                    message="Web search is disabled.",
                )
                observations = [*state.get("agent_observations", []), observation]
                self._event(trace, "agent_act", {"action": selected_action, "disabled": True})
                return {"agent_observations": observations, "subquery_statuses": subquery_statuses, "trace": trace}

            # Gate: only allow web search when local evidence already exists
            existing_chunks = list(state.get("chunks") or [])
            local_chunks = [c for c in existing_chunks if getattr(c, "provenance", "local") != "web"]
            if self._settings.web_search_requires_local_evidence and len(local_chunks) == 0:
                observation = AgentObservation(
                    action=selected_action,
                    success=False,
                    quality_score=0.0,
                    message="Web search blocked: no local document evidence to augment.",
                )
                observations = [*state.get("agent_observations", []), observation]
                self._event(trace, "agent_act", {"action": selected_action, "blocked": True, "reason": "no_local_evidence"})
                return {"agent_observations": observations, "subquery_statuses": subquery_statuses, "trace": trace}

            targeted_query = None
            if subquery_statuses and 0 <= target_idx < len(subquery_statuses):
                targeted_query = subquery_statuses[target_idx].query
            search_query = targeted_query or state.get("rewritten_query") or state.get("query", "")

            self._event(trace, "tool_web_search", {"query": search_query})
            web_chunks = self._tools.web_search(search_query, settings=self._settings)
            self._event(trace, "retrieve", {"hits": [c.model_dump() for c in web_chunks]})

            existing_chunks = list(state.get("chunks") or [])
            existing_ids = {c.chunk_id for c in existing_chunks}
            for chunk in web_chunks:
                if chunk.chunk_id not in existing_ids:
                    existing_chunks.append(chunk)
                    existing_ids.add(chunk.chunk_id)

            if subquery_statuses and 0 <= target_idx < len(subquery_statuses):
                sq = subquery_statuses[target_idx]
                new_chunk_ids = [c.chunk_id for c in web_chunks if c.chunk_id not in set(sq.chunk_ids)]
                subquery_statuses[target_idx] = sq.model_copy(update={
                    "status": "retrieved",
                    "chunk_ids": [*sq.chunk_ids, *new_chunk_ids],
                })

            tool_call_count = state.get("tool_call_count", 0) + 1
            observation = AgentObservation(
                action=selected_action,
                success=True,
                quality_score=0.0,
                message=f"Web search returned {len(web_chunks)} result(s) for subquery[{target_idx}].",
            )
            observations = [*state.get("agent_observations", []), observation]
            self._event(
                trace,
                "agent_act",
                {
                    "action": selected_action,
                    "chunk_count": len(web_chunks),
                    "total_chunks": len(existing_chunks),
                    "target_subquery_index": target_idx,
                },
            )
            return {
                "chunks": existing_chunks,
                "tool_call_count": tool_call_count,
                "agent_observations": observations,
                "subquery_statuses": subquery_statuses,
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
        return {"agent_observations": observations, "subquery_statuses": subquery_statuses, "trace": trace}

    def agent_reflect(self, state: PipelineState) -> PipelineState:
        trace = state["trace"]
        chunks = state.get("chunks", [])
        chunk_by_id = {c.chunk_id: c for c in chunks}

        if not chunks:
            quality_score = 0.0
        else:
            top_score = max(chunk.score for chunk in chunks)
            avg_score = sum(chunk.score for chunk in chunks) / len(chunks)
            quality_score = round((top_score * 0.7) + (avg_score * 0.3), 4)

        # Per-subquery quality assessment
        subquery_statuses = list(state.get("subquery_statuses") or [])
        threshold = self._settings.agent_evidence_quality_threshold
        for i, sq in enumerate(subquery_statuses):
            sq_chunks = [chunk_by_id[cid] for cid in sq.chunk_ids if cid in chunk_by_id]
            if not sq_chunks:
                sq_quality = 0.0
            else:
                sq_top = max(c.score for c in sq_chunks)
                sq_avg = sum(c.score for c in sq_chunks) / len(sq_chunks)
                sq_quality = round((sq_top * 0.7) + (sq_avg * 0.3), 4)
            new_status = "sufficient" if sq_quality >= threshold else sq.status
            subquery_statuses[i] = sq.model_copy(update={
                "quality_score": sq_quality,
                "status": new_status,
            })

        self._event(
            trace,
            "agent_reflect",
            {
                "quality_score": quality_score,
                "threshold": threshold,
                "agent_iterations": state.get("agent_iterations", 0),
                "subquery_quality": [
                    {"index": i, "status": sq.status, "quality": sq.quality_score}
                    for i, sq in enumerate(subquery_statuses)
                ],
            },
        )
        return {
            "evidence_quality_score": quality_score,
            "subquery_statuses": subquery_statuses,
            "trace": trace,
        }

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

        # Limit check: tool_call_count exceeding the configured max is the
        # canonical signal that the pipeline has exhausted its budget.  The
        # iteration_count mirrors the agent_max_iterations guard but is also
        # used on the non-agent path.
        iteration_count = state.get("iteration_count", 0)
        tool_call_count = state.get("tool_call_count", 0)
        limit_exceeded = tool_call_count > self._settings.agent_max_tool_calls

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

        # Detect refusal on first attempt — retry with force if evidence is strong
        chunks = state.get("chunks", [])
        top_score = max((c.score for c in chunks), default=0.0)
        safe_fail_msg = self._settings.safe_fail_message.lower()
        answer_lower = answer.lower()
        is_refusal = (
            safe_fail_msg in answer_lower
            or "do not have sufficient evidence" in answer_lower
            or "cannot answer" in answer_lower
            or "i don't have" in answer_lower
        )
        if top_score >= 0.7 and is_refusal:
            try:
                answer, citations, generation_source, prompt_version = self._generate_supported(
                    state, force_answer=True
                )
            except Exception:
                pass  # Keep first attempt

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
    def _pick_weakest_subquery(subquery_statuses: list[SubQueryStatus]) -> int:
        if not subquery_statuses:
            return 0
        # Prefer pending, then lowest quality
        best_idx = 0
        best_priority = (1, subquery_statuses[0].quality_score)  # (0=pending, 1=other), quality
        for i, sq in enumerate(subquery_statuses):
            is_pending = 0 if sq.status == "pending" else 1
            priority = (is_pending, sq.quality_score)
            if priority < best_priority:
                best_priority = priority
                best_idx = i
        return best_idx

    def _event(self, trace: PipelineTrace, stage: str, payload: dict[str, object]) -> None:
        trace.events.append(TraceEvent(stage=stage, payload=payload))
        if self._progress_callback is not None:
            try:
                summary = self._summarize_stage(stage, payload)
                self._progress_callback(stage, summary)
            except Exception:
                pass

    @staticmethod
    def _summarize_stage(stage: str, payload: dict[str, object]) -> dict[str, object]:
        if stage == "rewrite_query":
            return {"rewritten": payload.get("rewritten", ""), "clarify_needed": payload.get("clarify_needed")}
        if stage == "detect_query_type":
            return {"complexity": payload.get("query_complexity", ""), "is_conversation": payload.get("is_conversation_query")}
        if stage == "prepare_decomposition":
            return {"applied": payload.get("applied"), "query_count": payload.get("query_count")}
        if stage == "agent_think":
            return {"reasoning": str(payload.get("reasoning", ""))[:120], "action": payload.get("recommended_action")}
        if stage in ("retrieve", "agent_act"):
            return {"chunks_found": payload.get("chunk_count", len(payload.get("hits", []))), "action": payload.get("action")}
        if stage == "agent_reflect":
            return {"quality_score": payload.get("quality_score"), "threshold": payload.get("threshold")}
        if stage == "validate":
            return {"status": payload.get("status"), "confidence": payload.get("confidence")}
        if stage == "generate":
            return {"safe_fail": payload.get("safe_fail")}
        if stage == "verify_grounding":
            return {"status": payload.get("status"), "is_refusal": payload.get("is_refusal")}
        return {}

    def _generate_safe_fail(
        self, state: PipelineState
    ) -> tuple[str, str, str | None]:
        del state
        answer = self._settings.safe_fail_message
        response_source = "settings-default"
        prompt_version: str | None = None
        return answer, response_source, prompt_version

    def _generate_supported(
        self, state: PipelineState, force_answer: bool = False
    ) -> tuple[str, list[str], str, str | None]:
        chunks = state.get("chunks", [])
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        query_for_generation = state.get("original_query") or state.get("rewritten_query") or ""

        # Pass subquery context if decomposition was used
        subquery_statuses = state.get("subquery_statuses", [])
        subquery_texts = [sq.query for sq in subquery_statuses] if subquery_statuses else None

        answer, citation_chunk_ids, generation_source, prompt_version = (
            self._reasoner.synthesize_answer(
                query=query_for_generation,
                chunks=chunks,
                subqueries=subquery_texts,
                force_answer=force_answer,
            )
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


