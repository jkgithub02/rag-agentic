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


class _SequentialStubLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls = 0

    def invoke_text(self, prompt: str) -> str:
        del prompt
        if self.calls >= len(self._responses):
            return self._responses[-1]
        value = self._responses[self.calls]
        self.calls += 1
        return value


class _CaptureStubLLMClient:
    def __init__(self, response: str) -> None:
        self._response = response
        self.prompts: list[str] = []

    def invoke_text(self, prompt: str) -> str:
        self.prompts.append(prompt)
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


def test_analyze_query_normalizes_non_string_clarification_needed() -> None:
    reasoner = QueryReasoner(
        settings=_settings(),
        llm_client=_StubLLMClient(
            '{"is_clear": false, "questions": [], "rewritten_query": null, "clarification_needed": false, "prompt_version": "v1.0.0"}'
        ),
    )

    output, source = reasoner.analyze_query(query="what about Jason Kong", conversation_summary="")

    assert output.is_clear is False
    assert isinstance(output.clarification_needed, str)
    assert len(output.clarification_needed) > 0
    assert source == "llm"


def test_synthesize_answer_retries_when_first_response_is_not_json() -> None:
    reasoner = QueryReasoner(
        settings=_settings(),
        llm_client=_SequentialStubLLMClient(
            [
                "The encoder maps input tokens to contextual vectors.",
                '{"answer":"The encoder creates contextual token representations.","citation_chunk_ids":["bert-0001"],"prompt_version":"v1.0.0"}',
            ]
        ),
    )

    answer, citation_ids, _, _ = reasoner.synthesize_answer(
        query="What does the encoder do?",
        chunks=[EvidenceChunk(chunk_id="bert-0001", source="bert.pdf", text="Encoder contextualizes tokens.", score=0.8)],
    )

    assert "encoder" in answer.lower()
    assert citation_ids == ["bert-0001"]


def test_analyze_query_retries_when_first_response_is_not_json() -> None:
    reasoner = QueryReasoner(
        settings=_settings(),
        llm_client=_SequentialStubLLMClient(
            [
                "I need clarification.",
                '{"is_clear": true, "questions": ["Find encoder role in RAG, BERT, and Transformer?"], "rewritten_query": "Find encoder role in RAG, BERT, and Transformer", "clarification_needed": null, "prompt_version": "v1.0.0"}',
            ]
        ),
    )

    output, source = reasoner.analyze_query(
        query="encoder role across papers",
        conversation_summary="",
    )

    assert output.is_clear is True
    assert output.rewritten_query is not None
    assert source == "llm"


def test_synthesize_answer_uses_source_diverse_evidence_selection() -> None:
    llm = _CaptureStubLLMClient(
        '{"answer":"The encoder role differs across papers.","citation_chunk_ids":["rag-0001"],"prompt_version":"v1.0.0"}'
    )
    reasoner = QueryReasoner(settings=_settings(), llm_client=llm)

    chunks = [
        EvidenceChunk(chunk_id="attn-0001", source="attention.pdf", text="Attention chunk 1", score=0.9),
        EvidenceChunk(chunk_id="attn-0002", source="attention.pdf", text="Attention chunk 2", score=0.89),
        EvidenceChunk(chunk_id="attn-0003", source="attention.pdf", text="Attention chunk 3", score=0.88),
        EvidenceChunk(chunk_id="attn-0004", source="attention.pdf", text="Attention chunk 4", score=0.87),
        EvidenceChunk(chunk_id="rag-0001", source="rag.pdf", text="RAG encoder chunk", score=0.86),
        EvidenceChunk(chunk_id="bert-0001", source="bert.pdf", text="BERT encoder chunk", score=0.85),
    ]

    answer, citation_ids, _, _ = reasoner.synthesize_answer(
        query="encoder role across papers",
        chunks=chunks,
    )

    assert "encoder" in answer.lower()
    assert citation_ids == ["rag-0001"]
    assert len(llm.prompts) >= 1
    synthesis_prompt = llm.prompts[-1]
    assert "[rag-0001]" in synthesis_prompt
    assert "[bert-0001]" in synthesis_prompt
