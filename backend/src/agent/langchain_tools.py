from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, tool

from src.core.config import Settings
from src.core.models import EvidenceChunk
from src.db.vector_db import VectorDbManager
from src.services.reasoner import QueryReasoner


def _serialize_chunks(chunks: list[EvidenceChunk]) -> list[dict[str, Any]]:
    return [chunk.model_dump() for chunk in chunks]


def _parse_json_list(payload: str) -> list[Any]:
    if not payload.strip():
        return []
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return parsed


def _parse_chunks(chunks_json: str) -> list[EvidenceChunk]:
    return [
        EvidenceChunk.model_validate(item)
        for item in _parse_json_list(chunks_json)
        if isinstance(item, dict)
    ]


def _parse_chunk_ids(chunk_ids_json: str) -> list[str]:
    return [
        item
        for item in _parse_json_list(chunk_ids_json)
        if isinstance(item, str) and item.strip()
    ]


def create_langchain_tools(
    *,
    settings: Settings,
    vector_db: VectorDbManager,
    reasoner: QueryReasoner,
) -> list[BaseTool]:
    """Create LangChain tool wrappers around retrieval and reasoning services."""

    @tool
    def search_documents(query: str, strategy: str = "hybrid", top_k: int = 0) -> str:
        """Search indexed documents and return JSON list of evidence chunks.

        Args:
            query: Retrieval query.
            strategy: Retrieval strategy hint for future routing.
            top_k: Maximum chunk count; uses configured retrieval top-k when <= 0.
        """

        _ = strategy.strip().lower()
        effective_top_k = top_k if top_k > 0 else settings.retrieval_top_k
        chunks = vector_db.search(query, effective_top_k)
        return json.dumps(_serialize_chunks(chunks))

    @tool
    def fetch_chunks_by_ids(chunk_ids_json: str) -> str:
        """Hydrate chunk ids and return JSON list of evidence chunks.

        Args:
            chunk_ids_json: JSON array of chunk ids.
        """

        normalized = _parse_chunk_ids(chunk_ids_json)
        chunks = vector_db.fetch_by_ids(normalized)
        return json.dumps(_serialize_chunks(chunks))

    @tool
    def compress_evidence(chunks_json: str) -> str:
        """Summarize evidence snippets into compressed context text.

        Args:
            chunks_json: JSON array of evidence chunks.
        """

        chunks = _parse_chunks(chunks_json)
        if not chunks:
            return json.dumps({"summary": "", "chunk_count": 0})

        summary_input = [
            {
                "role": "assistant",
                "content": f"[{chunk.source}#{chunk.chunk_id}] {chunk.text[:320]}",
            }
            for chunk in chunks[:8]
        ]
        summary = reasoner.summarize_conversation(summary_input)
        return json.dumps({"summary": summary, "chunk_count": len(chunks)})

    @tool
    def verify_evidence_quality(chunks_json: str, min_score: float = 0.0) -> str:
        """Compute a lightweight quality score for current retrieved chunks.

        Args:
            chunks_json: JSON array of evidence chunks.
            min_score: Optional minimum score override; uses configured threshold when <= 0.
        """

        chunks = _parse_chunks(chunks_json)
        threshold = min_score if min_score > 0 else settings.agent_evidence_quality_threshold
        top_score = max((chunk.score for chunk in chunks), default=0.0)
        average_score = sum(chunk.score for chunk in chunks) / len(chunks) if chunks else 0.0
        quality_score = (top_score * 0.7) + (average_score * 0.3)
        is_sufficient = bool(chunks) and quality_score >= threshold
        return json.dumps(
            {
                "quality_score": round(quality_score, 4),
                "threshold": threshold,
                "is_sufficient": is_sufficient,
                "chunk_count": len(chunks),
            }
        )

    @tool
    def synthesize_answer(query: str, chunks_json: str) -> str:
        """Generate answer and citations from evidence chunks.

        Args:
            query: User query to answer.
            chunks_json: JSON array of evidence chunks.
        """

        chunks = _parse_chunks(chunks_json)
        answer, citation_chunk_ids, source, prompt_version = reasoner.synthesize_answer(
            query=query,
            chunks=chunks,
        )
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        citations = [
            f"{chunk_by_id[chunk_id].source}#{chunk_id}"
            for chunk_id in citation_chunk_ids
            if chunk_id in chunk_by_id
        ]
        return json.dumps(
            {
                "answer": answer,
                "citations": citations,
                "source": source,
                "prompt_version": prompt_version,
            }
        )

    return [
        search_documents,
        fetch_chunks_by_ids,
        compress_evidence,
        verify_evidence_quality,
        synthesize_answer,
    ]
