from __future__ import annotations

from functools import lru_cache

from src.agent.tools import AgentTools
from src.config import Settings, get_settings
from src.db.vector_db import VectorDbManager
from src.orchestration.pipeline import AgenticPipeline
from src.trace_store import TraceStore
from src.upload_service import UploadService


@lru_cache(maxsize=1)
def get_trace_store() -> TraceStore:
    return TraceStore()


@lru_cache(maxsize=1)
def get_vector_db() -> VectorDbManager:
    settings = get_settings()
    vector_db = VectorDbManager(settings)
    vector_db.build_index()
    return vector_db


@lru_cache(maxsize=1)
def get_upload_service() -> UploadService:
    settings = get_settings()
    return UploadService(settings=settings, vector_db=get_vector_db())


@lru_cache(maxsize=1)
def get_pipeline() -> AgenticPipeline:
    settings: Settings = get_settings()
    tools = AgentTools(get_vector_db())
    return AgenticPipeline(settings=settings, tools=tools, trace_store=get_trace_store())
