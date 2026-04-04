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
    QueryClarityOutput,
    QueryRewriteOutput,
)
from src.core.prompts import (
    answer_prompt,
    coverage_check_prompt,
    grounding_prompt,
    query_clarity_prompt,
    retry_rewrite_prompt,
    rewrite_query_prompt,
)
from src.services.llm_client import BedrockChatClient, LLMInvocationError

SchemaModelT = TypeVar("SchemaModelT")


class QueryReasoner:
    """Prompt-driven reasoner with strict LLM-only behavior."""

    def __init__(self, settings: Settings, llm_client: BedrockChatClient) -> None:
        self._settings = settings
        self._llm_client = llm_client

    def rewrite_query(self, query: str) -> tuple[QueryRewriteOutput, str]:
        if not self._settings.reasoning_enabled:
            raise LLMInvocationError("Reasoning is disabled but rewrite_query was invoked.")

        prompt = rewrite_query_prompt(query)
        output = self._invoke_structured(prompt, QueryRewriteOutput)
        output.rewritten_query = self._normalize_rewritten_query(
            output.rewritten_query,
            operation="rewrite_query",
        )
        return output, "llm"

    def assess_query_clarity(self, *, query: str) -> tuple[bool, str, str | None]:
        if not self._settings.reasoning_enabled:
            raise LLMInvocationError("Reasoning is disabled but assess_query_clarity was invoked.")

        prompt = query_clarity_prompt(query=query)
        output = self._invoke_structured(prompt, QueryClarityOutput)
        return output.clarify_needed, output.reason.strip(), output.prompt_version

    def rewrite_for_retry(
        self,
        *,
        original_query: str,
        retry_reason: str,
        evidence: list[str],
    ) -> tuple[QueryRewriteOutput, str]:
        if not self._settings.reasoning_enabled:
            raise LLMInvocationError("Reasoning is disabled but rewrite_for_retry was invoked.")

        evidence_block = "\n".join(f"- {item}" for item in evidence if item.strip())
        if not evidence_block:
            evidence_block = "- No evidence snippets available."

        prompt = retry_rewrite_prompt(
            original_query=original_query,
            retry_reason=retry_reason,
            evidence=evidence_block,
        )

        output = self._invoke_structured(prompt, QueryRewriteOutput)
        output.rewritten_query = self._normalize_rewritten_query(
            output.rewritten_query,
            operation="rewrite_for_retry",
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
        if isinstance(parsed.get("status"), str):
            parsed["status"] = parsed["status"].lower()
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
