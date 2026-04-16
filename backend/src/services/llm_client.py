from __future__ import annotations

import time
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError
from langchain_aws import ChatBedrock

from src.core.config import Settings


class LLMInvocationError(RuntimeError):
    """Raised when the Bedrock chat model invocation fails."""


class BedrockChatClient:
    """Thin wrapper around Bedrock Runtime for Claude-style JSON responses."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = ChatBedrock(
            region_name=settings.aws_region,
            model_id=settings.bedrock_chat_model_id,
            model_kwargs={"temperature": settings.reasoning_temperature},
        )

    @staticmethod
    def _message_text(message: Any) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts).strip()
        return str(content)

    def invoke_text(self, prompt: str) -> str:
        attempts = max(1, self._settings.reasoning_retry_attempts)
        backoff = max(0.0, self._settings.reasoning_retry_backoff_seconds)
        last_exc: Exception | None = None
        response = None

        for attempt in range(1, attempts + 1):
            try:
                response = self._client.invoke(prompt)
                break
            except (ClientError, BotoCoreError, Exception) as exc:
                last_exc = exc
                if attempt == attempts:
                    break
                if backoff > 0:
                    time.sleep(backoff * attempt)

        if response is None:
            if last_exc is not None:
                raise LLMInvocationError(str(last_exc)) from last_exc
            raise LLMInvocationError("Bedrock invocation failed without exception detail.")

        text = self._message_text(response)
        if not text:
            raise LLMInvocationError("No text content returned by Bedrock model.")

        return text.strip()
