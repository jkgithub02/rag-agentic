from __future__ import annotations

from src.core.models import EvidenceChunk
from src.db.vector_db import VectorDbManager


class AgentTools:
    """Explicit tools for retrieval and evidence fetch."""

    def __init__(self, vector_db: VectorDbManager) -> None:
        self._vector_db = vector_db

    def search_chunks(self, query: str, top_k: int) -> list[EvidenceChunk]:
        return self._vector_db.search(query, top_k)

    def fetch_chunks_by_ids(self, chunk_ids: list[str]) -> list[EvidenceChunk]:
        return self._vector_db.fetch_by_ids(chunk_ids)
