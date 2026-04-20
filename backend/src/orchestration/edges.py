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
        """Route after optional decomposition based on complexity and agent mode."""
        if self._settings and self._settings.enable_agent_mode:
            complexity = str(state.get("query_complexity", "moderate")).lower()
            if complexity in {"moderate", "complex"}:
                return "agent_initialize"
        return "retrieve"

    def route_agent_loop(self, state: PipelineState) -> str:
        """Continue or exit the agent loop based on quality and loop limits."""
        if self._settings is None:
            return "should_compress_context"

        quality = state.get("evidence_quality_score", 0.0)
        iterations = state.get("agent_iterations", 0)
        tool_calls = state.get("tool_call_count", 0)

        if quality >= self._settings.agent_evidence_quality_threshold:
            return "should_compress_context"
        if iterations >= self._settings.agent_max_iterations:
            return "should_compress_context"
        if tool_calls >= self._settings.agent_max_tool_calls:
            return "should_compress_context"
        return "agent_think"

    def route_after_should_compress(self, state: PipelineState) -> str:
        if state.get("limit_exceeded"):
            return "fallback_response"
        if state.get("compress_needed"):
            return "compress_context"
        return "validate"
