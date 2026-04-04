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
        text = self._llm_client.invoke_text(prompt).strip()
        if not text:
            raise LLMInvocationError("Response policy returned empty text.")
        return text, PROMPT_VERSION
