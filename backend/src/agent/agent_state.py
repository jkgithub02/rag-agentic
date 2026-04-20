from __future__ import annotations

from typing import TypedDict

from src.core.models import AgentObservation, AgentThought, EvidenceChunk


class AgentRuntimeState(TypedDict, total=False):
    """Focused runtime state fragment used by agent loop nodes.

    This intentionally mirrors only agent-specific fields so orchestration code can
    evolve incrementally without coupling to the full pipeline state.
    """

    agent_iterations: int
    evidence_quality_score: float
    selected_action: str
    chunks: list[EvidenceChunk]
    agent_thoughts: list[AgentThought]
    agent_observations: list[AgentObservation]
