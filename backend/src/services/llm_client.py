from __future__ import annotations

import json
import time

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from src.core.config import Settings


class LLMInvocationError(RuntimeError):
    """Raised when the Bedrock chat model invocation fails."""


class BedrockChatClient:
    """Thin wrapper around Bedrock Runtime for Claude-style JSON responses."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = boto3.client("bedrock-runtime", region_name=settings.aws_region)

    def invoke_text(self, prompt: str) -> str:
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "temperature": self._settings.reasoning_temperature,
            "max_tokens": self._settings.reasoning_max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
        }

        attempts = max(1, self._settings.reasoning_retry_attempts)
        backoff = max(0.0, self._settings.reasoning_retry_backoff_seconds)
        last_exc: Exception | None = None
        response = None

        for attempt in range(1, attempts + 1):
            try:
                response = self._client.invoke_model(
                    modelId=self._settings.bedrock_chat_model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(payload).encode("utf-8"),
                )
                break
            except (ClientError, BotoCoreError) as exc:
                last_exc = exc
                if attempt == attempts:
                    break
                if backoff > 0:
                    time.sleep(backoff * attempt)

        if response is None:
            if last_exc is not None:
                raise LLMInvocationError(str(last_exc)) from last_exc
            raise LLMInvocationError("Bedrock invocation failed without exception detail.")

        raw = response.get("body")
        if raw is None:
            raise LLMInvocationError("Missing response body from Bedrock invocation.")

        decoded = json.loads(raw.read())
        content = decoded.get("content", [])
        if not isinstance(content, list):
            raise LLMInvocationError("Unexpected Bedrock response format.")

        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    text_parts.append(text)

        if not text_parts:
            raise LLMInvocationError("No text content returned by Bedrock model.")

        return "\n".join(text_parts).strip()
