from __future__ import annotations

from functools import lru_cache

from agentic_rag.config import get_settings
from agentic_rag.ingestion import ensure_index
from agentic_rag.pipeline import AgenticPipeline
from agentic_rag.retrieval import KeywordRetriever
from agentic_rag.trace_store import TraceStore


@lru_cache(maxsize=1)
def get_trace_store() -> TraceStore:
    return TraceStore()


@lru_cache(maxsize=1)
def get_pipeline() -> AgenticPipeline:
    settings = get_settings()
    chunks = ensure_index(settings)
    retriever = KeywordRetriever(chunks)
    return AgenticPipeline(settings=settings, retriever=retriever, trace_store=get_trace_store())
