from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app
from src.bootstrap import get_pipeline, get_trace_store
from src.core.models import AskResponse, GroundingStatus, PipelineTrace


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def ask(self, query: str, *, thread_id: str | None = None) -> AskResponse:
        self.calls.append((query, thread_id))
        return AskResponse(
            answer="ok",
            citations=[],
            safe_fail=False,
            trace_id="trace-1",
        )


class FakeTraceStore:
    def __init__(self) -> None:
        self.traces = [
            PipelineTrace(
                trace_id="t-1",
                original_query="q1",
                rewritten_query="r1",
                final_grounding_status=GroundingStatus.SUPPORTED,
            ),
            PipelineTrace(
                trace_id="t-2",
                original_query="q2",
                rewritten_query="r2",
                final_grounding_status=GroundingStatus.UNSUPPORTED,
            ),
        ]
        self.last_limit: int | None = None

    def get(self, trace_id: str) -> PipelineTrace | None:
        for trace in self.traces:
            if trace.trace_id == trace_id:
                return trace
        return None

    def list_recent(self, limit: int = 20) -> list[PipelineTrace]:
        self.last_limit = limit
        return self.traces[:limit]


def test_ask_endpoint_forwards_optional_thread_id() -> None:
    fake_pipeline = FakePipeline()
    app.dependency_overrides[get_pipeline] = lambda: fake_pipeline

    try:
        client = TestClient(app)
        response = client.post("/ask", json={"query": "What is BERT?", "thread_id": "thread-a"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert fake_pipeline.calls == [("What is BERT?", "thread-a")]


def test_traces_endpoint_returns_recent_traces_and_limit() -> None:
    fake_store = FakeTraceStore()
    app.dependency_overrides[get_trace_store] = lambda: fake_store

    try:
        client = TestClient(app)
        response = client.get("/traces?limit=1")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["trace_id"] == "t-1"
    assert fake_store.last_limit == 1
