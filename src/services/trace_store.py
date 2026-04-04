from __future__ import annotations

from pathlib import Path
from threading import Lock

from src.core.models import PipelineTrace


class TraceStore:
    """Thread-safe trace store with in-memory cache and durable disk persistence."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        self._lock = Lock()
        self._traces: dict[str, PipelineTrace] = {}
        self._storage_dir = storage_dir or Path(".data/traces")
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def save(self, trace: PipelineTrace) -> None:
        with self._lock:
            self._traces[trace.trace_id] = trace
            self._trace_file(trace.trace_id).write_text(
                trace.model_dump_json(indent=2),
                encoding="utf-8",
            )

    def get(self, trace_id: str) -> PipelineTrace | None:
        with self._lock:
            cached = self._traces.get(trace_id)
            if cached is not None:
                return cached

            trace_file = self._trace_file(trace_id)
            if not trace_file.exists():
                return None

            trace = PipelineTrace.model_validate_json(trace_file.read_text(encoding="utf-8"))
            self._traces[trace_id] = trace
            return trace

    def list_recent(self, limit: int = 20) -> list[PipelineTrace]:
        safe_limit = max(1, limit)
        with self._lock:
            files = sorted(
                self._storage_dir.glob("*.json"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )

            traces: list[PipelineTrace] = []
            for trace_file in files[:safe_limit]:
                trace_id = trace_file.stem
                trace = self._traces.get(trace_id)
                if trace is None:
                    trace = PipelineTrace.model_validate_json(
                        trace_file.read_text(encoding="utf-8")
                    )
                    self._traces[trace_id] = trace
                traces.append(trace)

            return traces

    def _trace_file(self, trace_id: str) -> Path:
        return self._storage_dir / f"{trace_id}.json"
