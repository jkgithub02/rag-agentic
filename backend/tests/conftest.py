from __future__ import annotations

import sys
from pathlib import Path

from src.core.models import (
    EvidenceChunk,
    GroundingResult,
    GroundingStatus,
    QueryAnalysisOutput,
)
from src.services.llm_client import LLMInvocationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ====== Shared Test Fixtures & Fakes ======

class FakeTools:
    """Generic tools mock that returns deterministic BERT-related chunks.
    
    Useful for testing pipeline orchestration and question answering flow
    without depending on actual LLM or vector database.
    """
    def search_chunks(self, query: str, top_k: int) -> list[EvidenceChunk]:
        del top_k
        if "quantum" in query.lower():
            return []
        return [
            EvidenceChunk(
                chunk_id="bert-0001",
                source="bert.pdf",
                text="BERT uses masked language modeling.",
                score=0.7,
            )
        ]

    def fetch_chunks_by_ids(self, chunk_ids: list[str]) -> list[EvidenceChunk]:
        if "bert-0001" in chunk_ids:
            return [
                EvidenceChunk(
                    chunk_id="bert-0001",
                    source="bert.pdf",
                    text="BERT uses masked language modeling.",
                    score=0.9,
                )
            ]
        return []


class FakeReasoner:
    """Base reasoner mock with configurable grounding and synthesis responses.
    
    Allows tests to control LLM behavior without external API calls. Subclass
    for specialized behaviors (e.g., conversation awareness, error simulation).
    """
    def __init__(
        self,
        *,
        grounding_status: GroundingStatus = GroundingStatus.SUPPORTED,
        grounding_reason: str = "Grounded in evidence.",
        synthesis_answer: str = "BERT pretraining uses masked language modeling.",
        synthesis_chunk_ids: list[str] | None = None,
    ) -> None:
        self._grounding_status = grounding_status
        self._grounding_reason = grounding_reason
        self._synthesis_answer = synthesis_answer
        self._synthesis_chunk_ids = synthesis_chunk_ids or ["bert-0001"]

    def summarize_conversation(self, history: list[dict[str, str]]) -> str:
        if not history:
            return ""
        last_user = next((item["content"] for item in reversed(history) if item["role"] == "user"), "")
        return f"Conversation summary: prior user intent around '{last_user}'."

    def assess_grounding(
        self,
        *,
        answer: str,
        citations: list[str],
        evidence: list[str],
    ) -> tuple[GroundingResult, str, str | None]:
        del answer, citations, evidence
        return (
            GroundingResult(status=self._grounding_status, reason=self._grounding_reason),
            "llm",
            "v1.0.0",
        )

    def synthesize_answer(
        self,
        *,
        query: str,
        chunks: list[EvidenceChunk],
    ) -> tuple[str, list[str], str, str | None]:
        del query, chunks
        return self._synthesis_answer, self._synthesis_chunk_ids, "llm", "v1.0.0"

    def assess_query_clarity(
        self,
        *,
        query: str,
        conversation_summary: str | None = None,
    ) -> tuple[bool, str, str | None]:
        del conversation_summary
        return (query.strip().lower() in {"hi", "hello", "hey"}, "clarity check", "v1.0.0")

    def analyze_query(
        self,
        *,
        query: str,
        conversation_summary: str | None = None,
    ) -> tuple[QueryAnalysisOutput, str]:
        rewritten_query = f"{query} in detail?"
        clarify_needed, reason, prompt_version = self.assess_query_clarity(
            query=query,
            conversation_summary=conversation_summary,
        )
        if clarify_needed:
            return (
                QueryAnalysisOutput(
                    is_clear=False,
                    rewritten_query=None,
                    clarification_needed=reason,
                    prompt_version=prompt_version,
                ),
                "llm",
            )
        return (
            QueryAnalysisOutput(
                is_clear=True,
                rewritten_query=rewritten_query,
                clarification_needed=None,
                prompt_version="v1.0.0",
            ),
            "llm",
        )


class BrokenSynthesisReasoner(FakeReasoner):
    """Reasoner that fails on answer synthesis (simulates LLM error)."""
    def synthesize_answer(
        self,
        *,
        query: str,
        chunks: list[EvidenceChunk],
    ) -> tuple[str, list[str], str, str | None]:
        del query, chunks
        raise LLMInvocationError("synthesize_answer produced invalid citation ids.")


class BrokenAnalyzeReasoner(FakeReasoner):
    """Reasoner that fails on query analysis (simulates LLM error)."""
    def analyze_query(
        self,
        *,
        query: str,
        conversation_summary: str | None = None,
    ) -> tuple[QueryAnalysisOutput, str]:
        del query, conversation_summary
        raise LLMInvocationError("analyze_query returned non-JSON payload")
