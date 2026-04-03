from __future__ import annotations

from functools import lru_cache

from src.agent.tools import AgentTools
from src.config import get_settings
from src.db.vector_db import VectorDbManager
from src.orchestration.pipeline import AgenticPipeline
from src.trace_store import TraceStore


@lru_cache(maxsize=1)
def get_trace_store() -> TraceStore:
    return TraceStore()


@lru_cache(maxsize=1)
def get_pipeline() -> AgenticPipeline:
    settings = get_settings()
    vector_db = VectorDbManager(settings)
    vector_db.build_index()
    tools = AgentTools(vector_db)
    return AgenticPipeline(settings=settings, tools=tools, trace_store=get_trace_store())
