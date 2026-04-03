from __future__ import annotations

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile

from src.bootstrap import get_pipeline, get_trace_store, get_upload_service
from src.models import AskRequest, AskResponse, ConflictPolicy, PipelineTrace, UploadResponse
from src.upload_service import UploadService, UploadValidationError

app = FastAPI(title="Agentic RAG API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    return get_pipeline().ask(payload.query)


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),  # noqa: B008
    conflict_policy: ConflictPolicy = Form(default=ConflictPolicy.ASK),  # noqa: B008
    upload_service: UploadService = Depends(get_upload_service),  # noqa: B008
) -> UploadResponse:
    filename = file.filename or "uploaded-file"
    content = await file.read()

    try:
        return upload_service.upload_bytes(
            filename=filename,
            content=content,
            conflict_policy=conflict_policy,
        )
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/trace/{trace_id}", response_model=PipelineTrace)
def get_trace(trace_id: str) -> PipelineTrace:
    trace = get_trace_store().get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace
