from __future__ import annotations

from src.core.config import Settings
from src.core.models import EvidenceChunk, GroundingStatus
from src.services.reasoner import QueryReasoner


class _StubLLMClient:
    def __init__(self, response: str) -> None:
        self._response = response

    def invoke_text(self, prompt: str) -> str:
        del prompt
        return self._response


def _settings() -> Settings:
    return Settings(reasoning_enabled=True)


def test_assess_grounding_normalizes_grounded_to_supported() -> None:
    reasoner = QueryReasoner(
        settings=_settings(),
        llm_client=_StubLLMClient('{"status":"grounded","reason":"Backed by evidence."}'),
    )

    result, source, prompt_version = reasoner.assess_grounding(
        answer="BERT uses masked language modeling.",
        citations=["bert.pdf#bert-0001"],
        evidence=["BERT is trained with masked language modeling objective."],
    )

    assert result.status == GroundingStatus.SUPPORTED
    assert source == "llm"
    assert prompt_version is None


def test_assess_grounding_normalizes_ungrounded_to_unsupported() -> None:
    reasoner = QueryReasoner(
        settings=_settings(),
        llm_client=_StubLLMClient('{"status":"ungrounded","reason":"Not found in evidence."}'),
    )

    result, _, _ = reasoner.assess_grounding(
        answer="BERT was invented in 2024.",
        citations=["bert.pdf#bert-0001"],
        evidence=["BERT was introduced in 2018."],
    )

    assert result.status == GroundingStatus.UNSUPPORTED


def test_synthesize_answer_falls_back_to_top_chunk_ids_when_invalid_ids_returned() -> None:
    reasoner = QueryReasoner(
        settings=_settings(),
        llm_client=_StubLLMClient(
            '{"answer":"Jason has AI and cloud experience.","citation_chunk_ids":["missing-id"],"prompt_version":"v1.0.0"}'
        ),
    )

    answer, citation_ids, _, _ = reasoner.synthesize_answer(
        query="Who is Jason?",
        chunks=[
            EvidenceChunk(chunk_id="resume-0001", source="resume.pdf", text="AI experience", score=0.8),
            EvidenceChunk(chunk_id="resume-0002", source="resume.pdf", text="Cloud experience", score=0.7),
        ],
    )

    assert "Jason" in answer
    assert citation_ids == ["resume-0001", "resume-0002"]
