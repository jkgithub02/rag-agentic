from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ValidationStatus(StrEnum):
    """Outcome of validating retrieved evidence against the user query."""
    PASS = "pass"
    FAIL = "fail"


class GroundingStatus(StrEnum):
    """Indicates whether the final generated answer is fully supported by the retrieved evidence."""
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


class ResponseCategory(StrEnum):
    """Categorizes the type of response returned to the user."""
    CLARIFICATION = "clarification"
    SAFE_FAIL = "safe_fail"


class QueryComplexity(StrEnum):
    """Classifies the complexity of a user query, used to route to simple vs. agentic loops."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class EvidenceChunk(BaseModel):
    """A single piece of text evidence retrieved from a local document or web search."""
    chunk_id: str
    source: str
    text: str
    score: float = 0.0
    provenance: str = "local"  # "local" | "web"


class ValidationResult(BaseModel):
    """The result of the LLM validating retrieved chunks against a query."""
    status: ValidationStatus
    reason: str
    confidence: float


class GroundingResult(BaseModel):
    """The result of the LLM checking if the final synthesized answer is grounded in evidence."""
    status: GroundingStatus
    reason: str
    is_refusal: bool = False


class TraceEvent(BaseModel):
    """A single discrete step or state change recorded during pipeline execution."""
    stage: str
    payload: dict[str, Any]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SubQueryStatus(BaseModel):
    """Tracks the progress and quality of evidence retrieval for a decomposed sub-query."""
    query: str
    status: str = "pending"  # pending | retrieved | sufficient
    chunk_ids: list[str] = Field(default_factory=list)
    quality_score: float = 0.0


class AgentThought(BaseModel):
    """Represents the LLM agent's internal reasoning and planned next action."""
    reasoning: str
    recommended_action: str
    confidence: float = 0.0
    target_subquery_index: int | None = None


class AgentObservation(BaseModel):
    """Represents the outcome of a tool execution during the agent loop."""
    action: str
    success: bool
    quality_score: float = 0.0
    message: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineTrace(BaseModel):
    """A complete chronological log of a single request flowing through the agentic pipeline."""
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    original_query: str
    rewritten_query: str
    final_grounding_status: GroundingStatus = GroundingStatus.UNSUPPORTED
    agent_iterations_used: int = 0
    agent_thought_count: int = 0
    events: list[TraceEvent] = Field(default_factory=list)


class AskRequest(BaseModel):
    """Incoming request payload for the /ask API endpoint."""
    query: str = Field(min_length=2)
    thread_id: str | None = Field(default=None, min_length=1)


class AskResponse(BaseModel):
    """Outgoing response payload for the /ask API endpoint."""
    answer: str
    citations: list[str] = Field(default_factory=list)
    safe_fail: bool = False
    trace_id: str


class QueryAnalysisOutput(BaseModel):
    """Structured output from the LLM's initial query analysis and rewrite stage."""
    is_clear: bool
    questions: list[str] = Field(default_factory=list)
    rewritten_query: str | None = None
    clarification_needed: str | None = None
    prompt_version: str | None = None


class GroundingCheckOutput(BaseModel):
    """Structured output from the LLM when verifying if the answer matches the evidence."""
    status: GroundingStatus
    reason: str
    is_refusal: bool = False
    prompt_version: str | None = None


class AnswerSynthesisOutput(BaseModel):
    """Structured output from the LLM generating the final text response to the user."""
    answer: str
    citation_chunk_ids: list[str] = Field(default_factory=list)
    prompt_version: str | None = None


class ConflictPolicy(StrEnum):
    """Strategy for handling file name collisions during document upload."""
    ASK = "ask"
    REPLACE = "replace"
    KEEP_BOTH = "keep_both"


class UploadStatus(StrEnum):
    """Outcome of a document upload attempt."""
    SUCCESS = "success"
    CONFLICT = "conflict"


class UploadResponse(BaseModel):
    """Outgoing response payload for the /upload API endpoint."""
    status: UploadStatus
    message: str
    original_filename: str
    stored_filename: str | None = None
    chunks_added: int | None = None
    existing_filename: str | None = None
    suggested_filename: str | None = None
    conflict_options: list[ConflictPolicy] = Field(default_factory=list)

