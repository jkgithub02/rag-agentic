from __future__ import annotations

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.bootstrap import get_pipeline, get_settings, get_trace_store, get_upload_service
from src.core.models import AskRequest, AskResponse, ConflictPolicy, PipelineTrace, UploadResponse
from src.orchestration.pipeline import AgenticPipeline
from src.services.trace_store import TraceStore
from src.services.upload_service import UploadService, UploadValidationError

app = FastAPI(title="Agentic RAG API", version="0.1.0")

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_settings.cors_allowed_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(
    payload: AskRequest,
    pipeline: AgenticPipeline = Depends(get_pipeline),  # noqa: B008
) -> AskResponse:
    return pipeline.ask(payload.query, thread_id=payload.thread_id)


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


@app.get("/documents")
def list_documents(
    upload_service: UploadService = Depends(get_upload_service),  # noqa: B008
) -> dict[str, object]:
    return {"documents": upload_service.list_documents()}


@app.delete("/documents/{source_name}")
def delete_document(
    source_name: str,
    upload_service: UploadService = Depends(get_upload_service),  # noqa: B008
) -> dict[str, object]:
    deleted = upload_service.delete_document(source_name)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": True, "source_name": source_name}


@app.delete("/documents")
def delete_all_documents(
    upload_service: UploadService = Depends(get_upload_service),  # noqa: B008
) -> dict[str, object]:
    deleted_count = upload_service.delete_all_documents()
    return {"deleted": True, "deleted_count": deleted_count}


@app.get("/trace/{trace_id}", response_model=PipelineTrace)
def get_trace(
    trace_id: str,
    trace_store: TraceStore = Depends(get_trace_store),  # noqa: B008
) -> PipelineTrace:
    trace = trace_store.get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace


@app.get("/traces", response_model=list[PipelineTrace])
def list_traces(
    limit: int = 20,
    trace_store: TraceStore = Depends(get_trace_store),  # noqa: B008
) -> list[PipelineTrace]:
    return trace_store.list_recent(limit=limit)
