from __future__ import annotations

from typing import TypedDict

from src.core.models import (
    AgentObservation,
    AgentThought,
    EvidenceChunk,
    GroundingResult,
    PipelineTrace,
    ValidationResult,
)


class PipelineState(TypedDict, total=False):
    query: str
    history: list[dict[str, str]]
    conversation_summary: str
    original_query: str
    rewritten_query: str
    rewritten_queries: list[str]
    retrieval_keys: list[str]
    context_summary: str
    compress_needed: bool
    iteration_count: int
    tool_call_count: int
    retrieval_attempted: bool
    limit_exceeded: bool
    clarify_needed: bool
    clarify_message: str
    validation: ValidationResult
    answer: str
    citations: list[str]
    safe_fail: bool
    grounding: GroundingResult
    trace: PipelineTrace
    chunks: list[EvidenceChunk]
    agent_iterations: int
    evidence_quality_score: float
    agent_thoughts: list[AgentThought]
    agent_observations: list[AgentObservation]
    selected_action: str
