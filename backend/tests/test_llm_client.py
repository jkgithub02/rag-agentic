from __future__ import annotations

import json

import pytest

from src.core.config import Settings
from src.services.llm_client import BedrockChatClient, LLMInvocationError


class _Body:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> str:
        return json.dumps(self._payload)


class _FlakyChatBedrock:
    def __init__(self, *_: object, **__: object) -> None:
        self.calls = 0

    def invoke(self, *_: object, **__: object) -> object:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("retry")
        return type("_Message", (), {"content": "ok"})()


class _FailingChatBedrock:
    def __init__(self, *_: object, **__: object) -> None:
        pass

    def invoke(self, *_: object, **__: object) -> object:
        raise RuntimeError("down")


def _settings() -> Settings:
    return Settings(reasoning_retry_attempts=2, reasoning_retry_backoff_seconds=0)


def test_invoke_text_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FlakyChatBedrock()
    monkeypatch.setattr("src.services.llm_client.ChatBedrock", lambda *args, **kwargs: client)

    bedrock = BedrockChatClient(_settings())
    text = bedrock.invoke_text("hello")

    assert text == "ok"
    assert client.calls == 2


def test_invoke_text_raises_after_retry_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.services.llm_client.ChatBedrock",
        lambda *args, **kwargs: _FailingChatBedrock(),
    )

    bedrock = BedrockChatClient(_settings())

    with pytest.raises(LLMInvocationError):
        bedrock.invoke_text("hello")
