from __future__ import annotations

from functools import lru_cache

from src.agent.tools import AgentTools
from src.core.config import Settings, get_settings
from src.db.vector_db import VectorDbManager
from src.orchestration.pipeline import AgenticPipeline
from src.services.llm_client import BedrockChatClient
from src.services.reasoner import QueryReasoner
from src.services.trace_store import TraceStore
from src.services.upload_service import UploadService


@lru_cache(maxsize=1)
def get_trace_store() -> TraceStore:
    return TraceStore()


@lru_cache(maxsize=1)
def get_vector_db() -> VectorDbManager:
    settings = get_settings()
    try:
        vector_db = VectorDbManager(settings)
        vector_db.build_index()
        return vector_db
    except Exception as exc:
        if _is_vector_db_lock_error(exc):
            raise RuntimeError(
                "Vector database path is locked by another running process. "
                f"Path: {settings.vector_db_path}. "
                "On Windows, local Qdrant path mode is single-process. "
                "Stop other backend/Streamlit Python processes and retry."
            ) from exc
        raise


def _is_vector_db_lock_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "lock" in message or "already used by another process" in message


@lru_cache(maxsize=1)
def get_upload_service() -> UploadService:
    settings = get_settings()
    return UploadService(settings=settings, vector_db=get_vector_db())


@lru_cache(maxsize=1)
def get_bedrock_chat_client() -> BedrockChatClient:
    settings = get_settings()
    return BedrockChatClient(settings)


@lru_cache(maxsize=1)
def get_query_reasoner() -> QueryReasoner:
    settings = get_settings()
    return QueryReasoner(settings=settings, llm_client=get_bedrock_chat_client())


@lru_cache(maxsize=1)
def get_pipeline() -> AgenticPipeline:
    settings: Settings = get_settings()
    tools = AgentTools(get_vector_db())
    return AgenticPipeline(
        settings=settings,
        tools=tools,
        trace_store=get_trace_store(),
        reasoner=get_query_reasoner(),
    )
