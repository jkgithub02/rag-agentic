from __future__ import annotations

from src.orchestration.graph_state import PipelineState


class PipelineEdges:
    def __init__(self) -> None:
        pass

    def route_after_rewrite(self, state: PipelineState) -> str:
        return "retrieve"

    def route_after_should_compress(self, state: PipelineState) -> str:
        if state.get("limit_exceeded"):
            return "fallback_response"
        if state.get("compress_needed"):
            return "compress_context"
        return "validate"
