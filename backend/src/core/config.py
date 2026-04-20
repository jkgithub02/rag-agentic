import logging
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Runtime settings for Agentic RAG."""

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[2] / ".env"),
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
    retrieval_top_k: int = 7
    retrieval_neighbor_span: int = 1
    retrieval_mode: str = "hybrid"
    retrieval_dense_weight: float = 0.65
    retrieval_sparse_weight: float = 0.35
    context_compression_base_threshold: int = 2000
    context_compression_growth_factor: float = 0.9
    enable_agent_mode: bool = True
    agent_max_iterations: int = 10
    agent_evidence_quality_threshold: float = 0.65
    agent_tool_timeout_seconds: float = 2.0
    enable_query_decomposition: bool = True
    max_decomposition_depth: int = 3
    web_search_enabled: bool = True
    web_search_provider: str = "tavily"
    web_search_api_key: str = ""
    web_search_top_k: int = 3
    agent_max_tool_calls: int = 8

    min_relevance_score: float = 0.05
    ambiguity_margin: float = 0.005

    safe_fail_message: str = (
        "I do not have sufficient evidence in the indexed documents to answer this confidently."
    )
    clarification_message: str = (
        "Please ask a more specific question about the uploaded or indexed documents."
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache settings from .env or defaults, with logging."""
    # Ensure logging is configured to show INFO level
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s'
    )
    
    settings = Settings()
    
    # Log configuration on startup
    env_file_path = Path(__file__).resolve().parents[2] / ".env"
    
    print("=" * 70)
    print("⚙️  CONFIGURATION LOADED")
    print("=" * 70)
    print(f"Environment file: {env_file_path}")
    print(f"Environment file exists: {env_file_path.exists()}")
    print("\n📋 Core Settings:")
    print(f"  • AWS Region: {settings.aws_region}")
    print(f"  • Bedrock Model: {settings.bedrock_chat_model_id}")
    print(f"  • Reasoning Enabled: {settings.reasoning_enabled}")
    print(f"  • Reasoning Temperature: {settings.reasoning_temperature}")
    print(f"  • Reasoning Max Tokens: {settings.reasoning_max_tokens}")
    print("\n🔍 Retrieval Settings:")
    print(f"  • Embedding Provider: {settings.embedding_provider}")
    print(f"  • Ollama Base URL: {settings.ollama_base_url}")
    print(f"  • Ollama Embedding Model: {settings.ollama_embedding_model}")
    print(f"  • Retrieval Mode: {settings.retrieval_mode}")
    print(f"  • Retrieval Top K: {settings.retrieval_top_k}")
    print(f"  • Dense Weight: {settings.retrieval_dense_weight}")
    print(f"  • Sparse Weight: {settings.retrieval_sparse_weight}")
    print(f"  • Agent Mode Enabled: {settings.enable_agent_mode}")
    print(f"  • Agent Max Iterations: {settings.agent_max_iterations}")
    print(f"  • Agent Quality Threshold: {settings.agent_evidence_quality_threshold}")
    print(f"  • Web Search Enabled: {settings.web_search_enabled}")
    print("\n📦 Document Settings:")
    print(f"  • Chunk Size: {settings.chunk_size}")
    print(f"  • Chunk Overlap: {settings.chunk_overlap}")
    print(f"  • Documents Dir: {settings.documents_dir}")
    print(f"  • Vector DB Path: {settings.vector_db_path}")
    print(f"  • Max Upload Size: {settings.upload_max_file_size_mb} MB")
    print("\n🌐 API Settings:")
    print(f"  • CORS Origins: {', '.join(settings.cors_allowed_origins)}")
    print(f"  • Stream Token Delay: {settings.stream_token_delay_seconds}s")
    print("=" * 70)
    
    # Also log through logger
    logger.info("✓ Settings initialized successfully")
    
    return settings
