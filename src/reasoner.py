from __future__ import annotations

import json

from src.config import Settings
from src.llm_client import BedrockChatClient, LLMInvocationError
from src.models import QueryRewriteOutput
from src.prompts import rewrite_query_prompt


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
