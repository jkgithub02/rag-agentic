from __future__ import annotations

from src.orchestration.graph_state import PipelineState


class PipelineEdges:
    def __init__(self) -> None:
        pass

    def route_after_rewrite(self, state: PipelineState) -> str:
        if state.get("clarify_needed"):
            return "clarify"
        return "detect_query_type"

    def route_after_detect_query_type(self, state: PipelineState) -> str:
        """Route based on whether this is a conversation meta-query."""
        if state.get("is_conversation_query"):
            return "finish"
        return "retrieve"

    def route_after_should_compress(self, state: PipelineState) -> str:
        if state.get("limit_exceeded"):
            return "fallback_response"
        if state.get("compress_needed"):
            return "compress_context"
        return "validate"
