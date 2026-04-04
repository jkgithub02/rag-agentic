from __future__ import annotations

from src.core.config import Settings
from src.core.models import ValidationStatus
from src.orchestration.graph_state import PipelineState


class PipelineEdges:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def route_after_understand(self, state: PipelineState) -> str:
        if state.get("clarify_needed"):
            return "clarify"
        return "retrieve"

    def route_after_validate(self, state: PipelineState) -> str:
        if (
            state["validation"].status == ValidationStatus.RETRY
            and state["retry_count"] < self._settings.max_retry_count
        ):
            return "retry"
        if state["validation"].status == ValidationStatus.RETRY:
            return "retry_exhausted"
        return "hydrate"
