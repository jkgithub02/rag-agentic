from __future__ import annotations

from src.core.config import Settings
from src.orchestration.graph_state import PipelineState


class PipelineEdges:
    def __init__(self, *, settings: Settings | None = None) -> None:
        self._settings = settings

    def route_after_rewrite(self, state: PipelineState) -> str:
        del state
        return "detect_query_type"

    def route_after_detect_query_type(self, state: PipelineState) -> str:
        """Route based on whether this is a conversation meta-query."""
        if state.get("is_conversation_query"):
            return "finish"
        return "prepare_decomposition"

    def route_after_prepare_decomposition(self, state: PipelineState) -> str:
        """Always route through the agent loop.

        The agent path is now the single execution path for all query
        complexities.  Simple queries will search once and finalize in a
        single iteration.
        """
        del state
        return "agent_initialize"

    def evaluate_sufficiency(self, state: PipelineState) -> tuple[bool, str]:
        """Single policy for stop/continue decisions.

        Returns (should_stop, reason).
        Checks in priority order: planner finalize, hard limits,
        subquery coverage + quality.
        """
        if self._settings is None:
            return True, "no-settings"

        if state.get("selected_action") == "finalize":
            return True, "planner-finalize"

        # Hard limits always stop the loop regardless of coverage.
        iterations = state.get("agent_iterations", 0)
        if iterations >= self._settings.agent_max_iterations:
            return True, "iteration-limit"

        tool_calls = state.get("tool_call_count", 0)
        if tool_calls >= self._settings.agent_max_tool_calls:
            return True, "tool-call-limit"

        # Subquery-aware sufficiency: don't stop if any subquery is still
        # pending (never searched).  Once all have been searched at least
        # once, stop when overall quality is high enough or every subquery
        # individually meets the threshold.
        subquery_statuses = state.get("subquery_statuses") or []
        has_pending = any(sq.status == "pending" for sq in subquery_statuses)

        if has_pending:
            return False, "pending-subqueries"

        if subquery_statuses and all(sq.status == "sufficient" for sq in subquery_statuses):
            return True, "all-subqueries-sufficient"

        quality = state.get("evidence_quality_score", 0.0)
        if quality >= self._settings.agent_evidence_quality_threshold:
            return True, "quality-threshold"

        return False, "continue"

    def route_agent_loop(self, state: PipelineState) -> str:
        """Continue or exit the agent loop using the unified sufficiency policy."""
        should_stop, _ = self.evaluate_sufficiency(state)
        return "should_compress_context" if should_stop else "agent_think"

    def route_after_should_compress(self, state: PipelineState) -> str:
        if state.get("limit_exceeded"):
            return "fallback_response"
        if state.get("compress_needed"):
            return "compress_context"
        return "validate"
