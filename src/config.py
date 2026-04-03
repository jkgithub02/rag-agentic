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

    aws_region: str = "us-east-1"
    bedrock_chat_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"
    bedrock_embedding_model_id: str = "amazon.titan-embed-text-v2:0"

    documents_dir: Path = Path("documents")
    vector_db_path: Path = Path(".data/qdrant")
    vector_collection_name: str = "paper_chunks"
    vector_manifest_file: Path = Path(".data/vector_manifest.json")

    chunk_size: int = 1200
    chunk_overlap: int = 150
    retrieval_top_k: int = 4

    min_relevance_score: float = 0.12
    ambiguity_margin: float = 0.04
    max_retry_count: int = 1

    safe_fail_message: str = (
        "I do not have sufficient evidence in the indexed documents to answer this confidently."
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
