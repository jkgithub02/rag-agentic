from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for Agentic RAG."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AGENTIC_RAG_",
        extra="ignore",
    )

    aws_region: str = "ap-southeast-5"
    bedrock_chat_model_id: str = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
    reasoning_enabled: bool = True
    reasoning_temperature: float = 0.0
    reasoning_max_tokens: int = 300
    reasoning_retry_attempts: int = 3
    reasoning_retry_backoff_seconds: float = 0.35
    stream_token_delay_seconds: float = 0.015
    embedding_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "mxbai-embed-large"

    _backend_root: Path = Path(__file__).resolve().parents[2]

    documents_dir: Path = _backend_root/ "docs"
    vector_db_path: Path = _backend_root / ".data" / "qdrant"
    vector_collection_name: str = "document_chunks"
    allowed_upload_extensions: tuple[str, ...] = (".pdf", ".txt", ".md")
    upload_max_file_size_mb: int = 25
    cors_allowed_origins: tuple[str, ...] = (
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    )

    chunk_size: int = 1200
    chunk_overlap: int = 150
    retrieval_top_k: int = 4
    retrieval_neighbor_span: int = 1
    retrieval_mode: str = "hybrid"
    retrieval_dense_weight: float = 0.65
    retrieval_sparse_weight: float = 0.35
    context_compression_base_threshold: int = 2000
    context_compression_growth_factor: float = 0.9
    agent_max_iterations: int = 10
    agent_max_tool_calls: int = 8

    min_relevance_score: float = 0.07
    ambiguity_margin: float = 0.005

    safe_fail_message: str = (
        "I do not have sufficient evidence in the indexed documents to answer this confidently."
    )
    clarification_message: str = (
        "Please ask a more specific question about the uploaded or indexed documents."
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
