from __future__ import annotations

from typing import TypedDict

from src.core.models import EvidenceChunk, GroundingResult, PipelineTrace, ValidationResult


class PipelineState(TypedDict, total=False):
    query: str
    original_query: str
    rewritten_query: str
    clarify_needed: bool
    retry_count: int
    validation: ValidationResult
    answer: str
    citations: list[str]
    safe_fail: bool
    grounding: GroundingResult
    trace: PipelineTrace
    chunks: list[EvidenceChunk]
