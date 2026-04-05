from __future__ import annotations

from src.orchestration.graph_state import PipelineState


class PipelineEdges:
    def __init__(self) -> None:
        pass

    def route_after_rewrite(self, state: PipelineState) -> str:
        if state.get("clarify_needed"):
            return "clarify"
        return "retrieve"

    def route_after_validate(self, state: PipelineState) -> str:
        del state
        return "generate"
