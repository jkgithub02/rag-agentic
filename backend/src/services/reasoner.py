from __future__ import annotations

import json
from typing import TypeVar

from src.core.config import Settings
from src.core.models import (
    AnswerSynthesisOutput,
    CoverageCheckOutput,
    EvidenceChunk,
    GroundingCheckOutput,
    GroundingResult,
    QueryAnalysisOutput,
)
from src.core.prompts import (
    answer_prompt,
    conversation_summary_prompt,
    coverage_check_prompt,
    grounding_prompt,
    query_analysis_prompt,
)
from src.services.llm_client import BedrockChatClient, LLMInvocationError

SchemaModelT = TypeVar("SchemaModelT")


class QueryReasoner:
    """Prompt-driven reasoner with strict LLM-only behavior."""

    def __init__(self, settings: Settings, llm_client: BedrockChatClient) -> None:
        self._settings = settings
        self._llm_client = llm_client

    def summarize_conversation(self, history: list[dict[str, str]]) -> str:
        if not history:
            return ""

        turns: list[str] = []
        for item in history[-8:]:
            role = item.get("role", "user").strip().lower()
            label = "User" if role == "user" else "Assistant"
            content = item.get("content", "").strip()
            if content:
                turns.append(f"{label}: {content}")

        if not turns:
            return ""

        prompt = conversation_summary_prompt(history="\n".join(turns))
        return self._llm_client.invoke_text(prompt).strip()

    def analyze_query(
        self,
        *,
        query: str,
        conversation_summary: str | None = None,
    ) -> tuple[QueryAnalysisOutput, str]:
        if not self._settings.reasoning_enabled:
            raise LLMInvocationError("Reasoning is disabled but analyze_query was invoked.")

        prompt = query_analysis_prompt(query=query, conversation_summary=conversation_summary)
        output = self._invoke_structured(prompt, QueryAnalysisOutput)

        if output.is_clear:
            rewritten = (output.rewritten_query or query).strip()
            output.rewritten_query = self._normalize_rewritten_query(
                rewritten,
                operation="analyze_query",
            )
            output.clarification_needed = None
        else:
            output.rewritten_query = None
            output.clarification_needed = (
                output.clarification_needed.strip()
                if isinstance(output.clarification_needed, str) and output.clarification_needed.strip()
                else "Please clarify what you want to know from the uploaded documents."
            )

        return output, "llm"

    def assess_grounding(
        self,
        *,
        answer: str,
        citations: list[str],
        evidence: list[str],
    ) -> tuple[GroundingResult, str, str | None]:
        if not self._settings.reasoning_enabled:
            raise LLMInvocationError("Reasoning is disabled but assess_grounding was invoked.")

        evidence_block = "\n".join(f"- {item}" for item in evidence if item.strip())
        if not evidence_block:
            evidence_block = "- No evidence snippets available."

        prompt = grounding_prompt(answer=answer, citations=citations, evidence=evidence_block)

        parsed = self._invoke_structured_raw(prompt)
        parsed["status"] = self._normalize_grounding_status(parsed.get("status"))
        output = GroundingCheckOutput.model_validate(parsed)
        reason = output.reason.strip() or "Reasoner grounding evaluation complete."
        return (
            GroundingResult(status=output.status, reason=reason),
            "llm",
            output.prompt_version,
        )

    def synthesize_answer(
        self,
        *,
        query: str,
        chunks: list[EvidenceChunk],
    ) -> tuple[str, list[str], str, str | None]:
        if not self._settings.reasoning_enabled:
            raise LLMInvocationError("Reasoning is disabled but synthesize_answer was invoked.")

        evidence_block = "\n".join(
            f"- [{chunk.chunk_id}] ({chunk.source}) {chunk.text[:300]}" for chunk in chunks[:4]
        )
        if not evidence_block:
            evidence_block = "- No evidence snippets available."

        prompt = answer_prompt(query=query, evidence=evidence_block)

        output = self._invoke_structured(prompt, AnswerSynthesisOutput)
        answer = output.answer.strip()
        if not answer:
            raise LLMInvocationError("synthesize_answer produced empty answer.")

        allowed_ids = {chunk.chunk_id for chunk in chunks}
        citation_chunk_ids = [
            chunk_id for chunk_id in output.citation_chunk_ids if chunk_id in allowed_ids
        ]
        if not citation_chunk_ids:
            citation_chunk_ids = [chunk.chunk_id for chunk in chunks[:2] if chunk.chunk_id]
        if not citation_chunk_ids:
            raise LLMInvocationError("synthesize_answer produced invalid citation ids.")

        return answer, citation_chunk_ids, "llm", output.prompt_version

    def detect_insufficient_coverage(
        self,
        *,
        query: str,
        answer: str,
        chunks: list[EvidenceChunk],
    ) -> tuple[bool, list[str], str, str | None]:
        if not self._settings.reasoning_enabled:
            raise LLMInvocationError(
                "Reasoning is disabled but detect_insufficient_coverage was invoked."
            )

        evidence_block = "\n".join(
            f"- [{chunk.chunk_id}] ({chunk.source}) {chunk.text[:300]}" for chunk in chunks[:4]
        )
        if not evidence_block:
            evidence_block = "- No evidence snippets available."

        prompt = coverage_check_prompt(query=query, answer=answer, evidence=evidence_block)

        output = self._invoke_structured(prompt, CoverageCheckOutput)
        missing_terms = [term.strip().lower() for term in output.missing_terms if term.strip()]
        return output.unsupported, missing_terms, "llm", output.prompt_version

    def _invoke_structured(self, prompt: str, model_type: type[SchemaModelT]) -> SchemaModelT:
        parsed = self._invoke_structured_raw(prompt)
        return model_type.model_validate(parsed)

    def _invoke_structured_raw(self, prompt: str) -> dict[str, object]:
        llm_text = self._llm_client.invoke_text(prompt)
        return self._parse_json_payload(llm_text)

    @staticmethod
    def _normalize_rewritten_query(raw_query: str, *, operation: str) -> str:
        rewritten = raw_query.strip()
        if len(rewritten) < 2:
            raise LLMInvocationError(f"{operation} produced invalid output.")
        if not rewritten.endswith("?"):
            rewritten = rewritten + "?"
        return rewritten

    @staticmethod
    def _parse_json_payload(text: str) -> dict[str, object]:
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return json.loads(stripped)

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in LLM response.")
        return json.loads(stripped[start : end + 1])

    @staticmethod
    def _normalize_grounding_status(value: object) -> object:
        if not isinstance(value, str):
            return value

        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        synonyms = {
            "grounded": "supported",
            "supported": "supported",
            "partially_grounded": "partial",
            "partially_supported": "partial",
            "partial": "partial",
            "ungrounded": "unsupported",
            "not_grounded": "unsupported",
            "unsupported": "unsupported",
        }
        return synonyms.get(normalized, normalized)
