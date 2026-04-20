from __future__ import annotations

import hashlib
import logging

from langchain_core.tools import BaseTool

from src.agent.langchain_tools import create_langchain_tools
from src.core.config import Settings
from src.core.models import EvidenceChunk
from src.db.vector_db import VectorDbManager
from src.services.reasoner import QueryReasoner

logger = logging.getLogger(__name__)


class AgentTools:
    """Explicit tools for retrieval and evidence fetch."""

    def __init__(self, vector_db: VectorDbManager) -> None:
        self._vector_db = vector_db

    def search_chunks(self, query: str, top_k: int) -> list[EvidenceChunk]:
        return self._vector_db.search(query, top_k)

    def fetch_chunks_by_ids(self, chunk_ids: list[str]) -> list[EvidenceChunk]:
        return self._vector_db.fetch_by_ids(chunk_ids)

    def web_search(self, query: str, *, settings: Settings) -> list[EvidenceChunk]:
        """Search the web via configured provider. Returns chunks with provenance='web'."""
        if not settings.web_search_enabled:
            return []

        provider = settings.web_search_provider.strip().lower()
        if provider == "tavily":
            return self._tavily_search(query, settings=settings)

        logger.warning("Unsupported web search provider: %s", provider)
        return []

    @staticmethod
    def _tavily_search(query: str, *, settings: Settings) -> list[EvidenceChunk]:
        try:
            from tavily import TavilyClient  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("tavily-python not installed; skipping web search.")
            return []

        api_key = settings.web_search_api_key
        if not api_key:
            logger.warning("AGENTIC_RAG_WEB_SEARCH_API_KEY not set; skipping web search.")
            return []

        try:
            client = TavilyClient(api_key=api_key)
            response = client.search(query=query, max_results=settings.web_search_top_k)
        except Exception as exc:
            logger.warning("Tavily web search failed: %s", exc)
            return []

        results = response.get("results", [])
        chunks: list[EvidenceChunk] = []
        for i, item in enumerate(results):
            url = item.get("url", "web")
            content = item.get("content", "")
            if not content:
                continue
            digest = hashlib.sha1(f"{url}:{i}".encode()).hexdigest()[:8]
            chunk_id = f"web-{digest}"
            chunks.append(
                EvidenceChunk(
                    chunk_id=chunk_id,
                    source=url,
                    text=content[:1200],
                    score=item.get("score", 0.5),
                    provenance="web",
                )
            )
        return chunks

    def get_langchain_tools(
        self,
        *,
        settings: Settings,
        reasoner: QueryReasoner,
    ) -> list[BaseTool]:
        """Expose LangChain tools for agentic orchestration."""

        return create_langchain_tools(
            settings=settings,
            vector_db=self._vector_db,
            reasoner=reasoner,
        )
