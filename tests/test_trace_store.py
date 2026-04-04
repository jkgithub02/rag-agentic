from __future__ import annotations

import os

from src.core.models import PipelineTrace
from src.services.trace_store import TraceStore


def test_trace_store_persists_and_reads_from_disk(tmp_path) -> None:
    store = TraceStore(storage_dir=tmp_path)
    trace = PipelineTrace(original_query="q", rewritten_query="rq")

    store.save(trace)

    trace_file = tmp_path / f"{trace.trace_id}.json"
    assert trace_file.exists()

    reloaded_store = TraceStore(storage_dir=tmp_path)
    loaded = reloaded_store.get(trace.trace_id)

    assert loaded is not None
    assert loaded.trace_id == trace.trace_id
    assert loaded.original_query == "q"


def test_trace_store_returns_none_for_unknown_trace(tmp_path) -> None:
    store = TraceStore(storage_dir=tmp_path)

    assert store.get("missing-trace-id") is None


def test_trace_store_list_recent_returns_newest_first(tmp_path) -> None:
    store = TraceStore(storage_dir=tmp_path)
    t1 = PipelineTrace(original_query="q1", rewritten_query="r1")
    t2 = PipelineTrace(original_query="q2", rewritten_query="r2")
    t3 = PipelineTrace(original_query="q3", rewritten_query="r3")

    store.save(t1)
    store.save(t2)
    store.save(t3)

    os.utime(tmp_path / f"{t1.trace_id}.json", (1_000, 1_000))
    os.utime(tmp_path / f"{t2.trace_id}.json", (2_000, 2_000))
    os.utime(tmp_path / f"{t3.trace_id}.json", (3_000, 3_000))

    recent = store.list_recent(limit=2)

    assert [trace.trace_id for trace in recent] == [t3.trace_id, t2.trace_id]
