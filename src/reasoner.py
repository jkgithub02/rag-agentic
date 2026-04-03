from __future__ import annotations

import json

from src.config import Settings
from src.llm_client import BedrockChatClient, LLMInvocationError
from src.models import (
    GroundingCheckOutput,
    GroundingResult,
    GroundingStatus,
    QueryRewriteOutput,
)
from src.prompts import grounding_prompt, retry_rewrite_prompt, rewrite_query_prompt


class QueryReasoner:
    """Prompt-driven reasoner with deterministic fallbacks for stability."""

    def __init__(self, settings: Settings, llm_client: BedrockChatClient) -> None:
        self._settings = settings
        self._llm_client = llm_client

    def rewrite_query(self, query: str) -> tuple[QueryRewriteOutput, str]:
        fallback = QueryRewriteOutput(rewritten_query=query.rstrip("?") + "?", prompt_version=None)
        if not self._settings.reasoning_enabled:
            return fallback, "fallback-disabled"

        prompt = rewrite_query_prompt(query)
        try:
            llm_text = self._llm_client.invoke_text(prompt)
            parsed = self._parse_json_payload(llm_text)
            output = QueryRewriteOutput.model_validate(parsed)
            rewritten = output.rewritten_query.strip()
            if len(rewritten) < 2:
                return fallback, "fallback-invalid-output"
            if not rewritten.endswith("?"):
                rewritten = rewritten + "?"
            output.rewritten_query = rewritten
            return output, "llm"
        except (LLMInvocationError, ValueError, json.JSONDecodeError):
            return fallback, "fallback-llm-error"

    def rewrite_for_retry(
        self,
        *,
        original_query: str,
        retry_reason: str,
        evidence: list[str],
    ) -> tuple[QueryRewriteOutput, str]:
        fallback = QueryRewriteOutput(
            rewritten_query=f"{original_query.rstrip('?')}? Focus on one concrete source.",
            prompt_version=None,
        )
        if not self._settings.reasoning_enabled:
            return fallback, "fallback-disabled"

        evidence_block = "\n".join(f"- {item}" for item in evidence if item.strip())
        if not evidence_block:
            evidence_block = "- No evidence snippets available."

        prompt = retry_rewrite_prompt(
            original_query=original_query,
            retry_reason=retry_reason,
            evidence=evidence_block,
        )

        try:
            llm_text = self._llm_client.invoke_text(prompt)
            parsed = self._parse_json_payload(llm_text)
            output = QueryRewriteOutput.model_validate(parsed)
            rewritten = output.rewritten_query.strip()
            if len(rewritten) < 2:
                return fallback, "fallback-invalid-output"
            if not rewritten.endswith("?"):
                rewritten = rewritten + "?"
            output.rewritten_query = rewritten
            return output, "llm"
        except (LLMInvocationError, ValueError, json.JSONDecodeError):
            return fallback, "fallback-llm-error"

    def assess_grounding(
        self,
        *,
        answer: str,
        citations: list[str],
        evidence: list[str],
    ) -> tuple[GroundingResult, str, str | None]:
        fallback = self._heuristic_grounding(answer=answer, citations=citations, evidence=evidence)
        if not self._settings.reasoning_enabled:
            return fallback, "fallback-disabled", None

        evidence_block = "\n".join(f"- {item}" for item in evidence if item.strip())
        if not evidence_block:
            evidence_block = "- No evidence snippets available."

        prompt = grounding_prompt(answer=answer, citations=citations, evidence=evidence_block)

        try:
            llm_text = self._llm_client.invoke_text(prompt)
            parsed = self._parse_json_payload(llm_text)
            if isinstance(parsed.get("status"), str):
                parsed["status"] = parsed["status"].lower()
            output = GroundingCheckOutput.model_validate(parsed)
            reason = output.reason.strip() or "Reasoner grounding evaluation complete."
            return (
                GroundingResult(status=output.status, reason=reason),
                "llm",
                output.prompt_version,
            )
        except (LLMInvocationError, ValueError, json.JSONDecodeError):
            return fallback, "fallback-llm-error", None

    @staticmethod
    def _heuristic_grounding(
        *, answer: str, citations: list[str], evidence: list[str]
    ) -> GroundingResult:
        if not answer.strip():
            return GroundingResult(status=GroundingStatus.UNSUPPORTED, reason="No answer text.")
        if citations:
            return GroundingResult(status=GroundingStatus.SUPPORTED, reason="Citations present.")
        if evidence:
            return GroundingResult(status=GroundingStatus.PARTIAL, reason="No citations.")
        return GroundingResult(status=GroundingStatus.UNSUPPORTED, reason="No evidence.")

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
