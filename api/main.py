from __future__ import annotations

from fastapi import FastAPI, HTTPException

from agentic_rag.bootstrap import get_pipeline, get_trace_store
from agentic_rag.models import AskRequest, AskResponse, PipelineTrace

app = FastAPI(title="Agentic RAG API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    pipeline = get_pipeline()
    return pipeline.ask(payload.query)


@app.get("/trace/{trace_id}", response_model=PipelineTrace)
def get_trace(trace_id: str) -> PipelineTrace:
    trace = get_trace_store().get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace
