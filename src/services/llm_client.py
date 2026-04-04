from __future__ import annotations

import json

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

        try:
            response = self._client.invoke_model(
                modelId=self._settings.bedrock_chat_model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload).encode("utf-8"),
            )
        except (ClientError, BotoCoreError) as exc:
            raise LLMInvocationError(str(exc)) from exc

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
