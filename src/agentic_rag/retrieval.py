from __future__ import annotations

import re
from collections import Counter
from typing import Protocol

from agentic_rag.models import EvidenceChunk

_WORD_REGEX = re.compile(r"[a-z0-9]+")


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[EvidenceChunk]: ...


def _tokenize(text: str) -> list[str]:
    return _WORD_REGEX.findall(text.lower())


def _overlap_score(query_tokens: list[str], chunk_tokens: list[str]) -> float:
    if not query_tokens or not chunk_tokens:
        return 0.0

    query_counter = Counter(query_tokens)
    chunk_counter = Counter(chunk_tokens)
    intersection = sum(min(query_counter[word], chunk_counter[word]) for word in query_counter)
    denominator = max(len(query_tokens), len(chunk_tokens))
    return intersection / denominator


class KeywordRetriever:
    """Simple lexical retriever used for deterministic baseline behavior."""

    def __init__(self, chunks: list[EvidenceChunk]) -> None:
        self._chunks = chunks

    def retrieve(self, query: str, top_k: int) -> list[EvidenceChunk]:
        query_tokens = _tokenize(query)
        scored: list[EvidenceChunk] = []

        for chunk in self._chunks:
            score = _overlap_score(query_tokens, _tokenize(chunk.text))
            if score <= 0:
                continue
            scored.append(
                EvidenceChunk(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    text=chunk.text,
                    score=score,
                )
            )

        ranked = sorted(scored, key=lambda item: item.score, reverse=True)
        return ranked[:top_k]
