from __future__ import annotations

import json

import pytest
from botocore.exceptions import ClientError

from src.core.config import Settings
from src.services.llm_client import BedrockChatClient, LLMInvocationError


class _Body:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> str:
        return json.dumps(self._payload)


class _FlakyClient:
    def __init__(self) -> None:
        self.calls = 0

    def invoke_model(self, **_: object) -> dict:
        self.calls += 1
        if self.calls == 1:
            raise ClientError(
                error_response={"Error": {"Code": "ThrottlingException", "Message": "retry"}},
                operation_name="InvokeModel",
            )
        return {"body": _Body({"content": [{"type": "text", "text": "ok"}]})}


class _FailingClient:
    def invoke_model(self, **_: object) -> dict:
        raise ClientError(
            error_response={"Error": {"Code": "ServiceUnavailable", "Message": "down"}},
            operation_name="InvokeModel",
        )


def _settings() -> Settings:
    return Settings(reasoning_retry_attempts=2, reasoning_retry_backoff_seconds=0)


def test_invoke_text_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FlakyClient()
    monkeypatch.setattr("src.services.llm_client.boto3.client", lambda *args, **kwargs: client)

    bedrock = BedrockChatClient(_settings())
    text = bedrock.invoke_text("hello")

    assert text == "ok"
    assert client.calls == 2


def test_invoke_text_raises_after_retry_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.services.llm_client.boto3.client",
        lambda *args, **kwargs: _FailingClient(),
    )

    bedrock = BedrockChatClient(_settings())

    with pytest.raises(LLMInvocationError):
        bedrock.invoke_text("hello")
