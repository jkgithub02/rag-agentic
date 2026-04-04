from __future__ import annotations

import hashlib
import re
from pathlib import Path
from threading import Lock

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.core.config import Settings
from src.core.models import EvidenceChunk

_SOURCE_TOKEN_REGEX = re.compile(r"[^a-z0-9]+")


class VectorDbManager:
    """Persistent local Qdrant manager for document chunk retrieval."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = QdrantClient(path=str(settings.vector_db_path))
        self._embeddings = self._build_embeddings(settings)
        self._chunk_lookup: dict[str, EvidenceChunk] = {}
        self._write_lock = Lock()
        self._vector_size: int | None = None

    @staticmethod
    def _build_embeddings(settings: Settings) -> Embeddings:
        provider = settings.embedding_provider.strip().lower()
        if provider == "ollama":
            return OllamaEmbeddings(
                model=settings.ollama_embedding_model,
                base_url=settings.ollama_base_url,
            )

        raise ValueError("Unsupported embedding provider. Use 'ollama'.")

    def build_index(self) -> int:
        with self._write_lock:
            chunks = self._load_and_chunk_documents(self._settings.documents_dir)
            vector_size = self._embedding_dimension()
            self._client.recreate_collection(
                collection_name=self._settings.vector_collection_name,
                vectors_config=qmodels.VectorParams(
                    size=vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            self._chunk_lookup.clear()
            if chunks:
                self._upsert_chunks_locked(chunks)
            return len(chunks)

    def index_file(self, file_path: Path, source_name: str, *, replace_existing: bool) -> int:
        with self._write_lock:
            vector_size = self._embedding_dimension()
            self._ensure_collection(vector_size)
            if replace_existing:
                self._delete_source_locked(source_name)

            chunks = self._chunks_for_file(file_path=file_path, source_name=source_name)
            if not chunks:
                return 0
            self._upsert_chunks_locked(chunks)
            return len(chunks)

    def search(self, query: str, top_k: int) -> list[EvidenceChunk]:
        vector = self._embeddings.embed_query(query)
        response = self._client.query_points(
            collection_name=self._settings.vector_collection_name,
            query=vector,
            limit=top_k,
            with_payload=True,
        )

        hits: list[EvidenceChunk] = []
        for point in response.points:
            payload = point.payload or {}
            chunk_id = payload.get("chunk_id")
            source = payload.get("source")
            text = payload.get("text")
            if (
                not isinstance(chunk_id, str)
                or not isinstance(source, str)
                or not isinstance(text, str)
            ):
                continue
            hits.append(
                EvidenceChunk(
                    chunk_id=chunk_id,
                    source=source,
                    text=text,
                    score=float(point.score or 0.0),
                )
            )
        return hits

    def fetch_by_ids(self, chunk_ids: list[str]) -> list[EvidenceChunk]:
        return [self._chunk_lookup[cid] for cid in chunk_ids if cid in self._chunk_lookup]

    def _load_and_chunk_documents(self, docs_dir: Path) -> list[EvidenceChunk]:
        if not docs_dir.exists():
            return []

        supported = {ext.lower() for ext in self._settings.allowed_upload_extensions}
        docs = [
            path
            for path in docs_dir.iterdir()
            if path.is_file() and path.suffix.lower() in supported
        ]
        docs.sort(key=lambda path: path.name.lower())

        chunks: list[EvidenceChunk] = []
        for path in docs:
            chunks.extend(self._chunks_for_file(file_path=path, source_name=path.name))
        return chunks

    def _chunks_for_file(self, file_path: Path, source_name: str) -> list[EvidenceChunk]:
        text = self._extract_text(file_path)
        split_texts = self._chunk_text(
            text=text,
            chunk_size=self._settings.chunk_size,
            overlap=self._settings.chunk_overlap,
        )
        chunks: list[EvidenceChunk] = []
        for index, chunk_text in enumerate(split_texts):
            chunk_id = self._make_chunk_id(source_name, index)
            chunks.append(
                EvidenceChunk(
                    chunk_id=chunk_id,
                    source=source_name,
                    text=chunk_text,
                )
            )
        return chunks

    def _upsert_chunks_locked(self, chunks: list[EvidenceChunk]) -> None:
        vectors = self._embeddings.embed_documents([chunk.text for chunk in chunks])
        points: list[qmodels.PointStruct] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            points.append(
                qmodels.PointStruct(
                    id=self._point_id(chunk.chunk_id),
                    vector=vector,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "source": chunk.source,
                        "text": chunk.text,
                    },
                )
            )
            self._chunk_lookup[chunk.chunk_id] = chunk

        self._client.upsert(
            collection_name=self._settings.vector_collection_name,
            points=points,
            wait=True,
        )

    def _delete_source_locked(self, source_name: str) -> None:
        if not self._client.collection_exists(self._settings.vector_collection_name):
            return

        selector = qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="source",
                        match=qmodels.MatchValue(value=source_name),
                    )
                ]
            )
        )
        self._client.delete(
            collection_name=self._settings.vector_collection_name,
            points_selector=selector,
            wait=True,
        )

        stale_ids = [
            chunk_id
            for chunk_id, chunk in self._chunk_lookup.items()
            if chunk.source == source_name
        ]
        for chunk_id in stale_ids:
            del self._chunk_lookup[chunk_id]

    def _embedding_dimension(self) -> int:
        if self._vector_size is None:
            self._vector_size = len(self._embeddings.embed_query("dimension probe"))
        return self._vector_size

    def _ensure_collection(self, vector_size: int) -> None:
        if self._client.collection_exists(self._settings.vector_collection_name):
            return
        self._client.create_collection(
            collection_name=self._settings.vector_collection_name,
            vectors_config=qmodels.VectorParams(
                size=vector_size,
                distance=qmodels.Distance.COSINE,
            ),
        )

    @staticmethod
    def _make_chunk_id(source_name: str, index: int) -> str:
        token = _SOURCE_TOKEN_REGEX.sub("-", source_name.lower()).strip("-") or "document"
        return f"{token}-{index:04d}"

    @staticmethod
    def _extract_text(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            reader = PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)

        if suffix in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="ignore")

        raise ValueError(f"Unsupported document type: {path.suffix}")

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
        normalized = " ".join(text.split())
        if not normalized:
            return []

        result: list[str] = []
        start = 0
        while start < len(normalized):
            end = min(start + chunk_size, len(normalized))
            chunk = normalized[start:end].strip()
            if chunk:
                result.append(chunk)
            if end == len(normalized):
                break
            start = max(end - overlap, start + 1)
        return result

    @staticmethod
    def _point_id(chunk_id: str) -> int:
        digest = hashlib.sha1(chunk_id.encode("utf-8")).digest()[:8]
        return int.from_bytes(digest, byteorder="big", signed=False)
