from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ValidationStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"


class GroundingStatus(StrEnum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


class ResponseCategory(StrEnum):
    CLARIFICATION = "clarification"
    SAFE_FAIL = "safe_fail"


class EvidenceChunk(BaseModel):
    chunk_id: str
    source: str
    text: str
    score: float = 0.0


class ValidationResult(BaseModel):
    status: ValidationStatus
    reason: str
    confidence: float


class GroundingResult(BaseModel):
    status: GroundingStatus
    reason: str


class TraceEvent(BaseModel):
    stage: str
    payload: dict[str, Any]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PipelineTrace(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    original_query: str
    rewritten_query: str
    final_grounding_status: GroundingStatus = GroundingStatus.UNSUPPORTED
    events: list[TraceEvent] = Field(default_factory=list)


class AskRequest(BaseModel):
    query: str = Field(min_length=2)
    thread_id: str | None = Field(default=None, min_length=1)


class AskResponse(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    safe_fail: bool = False
    trace_id: str


class QueryAnalysisOutput(BaseModel):
    is_clear: bool
    questions: list[str] = Field(default_factory=list)
    rewritten_query: str | None = None
    clarification_needed: str | None = None
    prompt_version: str | None = None


class GroundingCheckOutput(BaseModel):
    status: GroundingStatus
    reason: str
    prompt_version: str | None = None


class AnswerSynthesisOutput(BaseModel):
    answer: str
    citation_chunk_ids: list[str] = Field(default_factory=list)
    prompt_version: str | None = None


class ConflictPolicy(StrEnum):
    ASK = "ask"
    REPLACE = "replace"
    KEEP_BOTH = "keep_both"


class UploadStatus(StrEnum):
    SUCCESS = "success"
    CONFLICT = "conflict"


class UploadResponse(BaseModel):
    status: UploadStatus
    message: str
    original_filename: str
    stored_filename: str | None = None
    chunks_added: int | None = None
    existing_filename: str | None = None
    suggested_filename: str | None = None
    conflict_options: list[ConflictPolicy] = Field(default_factory=list)
