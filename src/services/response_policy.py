from __future__ import annotations

from src.core.models import ResponseCategory
from src.core.prompts import PROMPT_VERSION, naturalize_response_prompt
from src.services.llm_client import BedrockChatClient, LLMInvocationError


class ResponsePolicy:
    """Generates natural user-facing text for selected response categories."""

    def __init__(self, llm_client: BedrockChatClient) -> None:
        self._llm_client = llm_client

    def render(
        self,
        *,
        category: ResponseCategory,
        query: str,
        reason: str | None = None,
        evidence_count: int = 0,
    ) -> tuple[str, str]:
        prompt = naturalize_response_prompt(
            category=category.value,
            query=query,
            reason=reason,
            evidence_count=evidence_count,
        )
        try:
            text = self._llm_client.invoke_text(prompt).strip()
        except LLMInvocationError:
            text = self._fallback_text(category=category, query=query, reason=reason)
        if not text:
            text = self._fallback_text(category=category, query=query, reason=reason)
        return text, PROMPT_VERSION

    @staticmethod
    def _fallback_text(*, category: ResponseCategory, query: str, reason: str | None) -> str:
        if category == ResponseCategory.CLARIFICATION:
            return "Could you share a bit more detail about what you want to know?"
        if category == ResponseCategory.RETRY_EXHAUSTED:
            return "I could not confidently disambiguate that request from the available evidence."
        if category == ResponseCategory.SAFE_FAIL:
            return "I do not have enough evidence in the indexed documents to answer confidently."
        if category == ResponseCategory.GROUNDING_REASON:
            return reason or "The answer could not be fully grounded in the available evidence."
        return reason or query
