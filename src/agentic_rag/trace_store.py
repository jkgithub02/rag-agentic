from __future__ import annotations

from threading import Lock

from agentic_rag.models import PipelineTrace


class TraceStore:
    """Thread-safe in-memory store for pipeline traces."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._traces: dict[str, PipelineTrace] = {}

    def save(self, trace: PipelineTrace) -> None:
        with self._lock:
            self._traces[trace.trace_id] = trace

    def get(self, trace_id: str) -> PipelineTrace | None:
        with self._lock:
            return self._traces.get(trace_id)
