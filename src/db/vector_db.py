from __future__ import annotations

import hashlib
import re
from pathlib import Path

from langchain_aws import BedrockEmbeddings
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.config import Settings
from src.models import EvidenceChunk

_WORD_REGEX = re.compile(r"[a-z0-9]+")


class VectorDbManager:
    """Persistent local Qdrant manager for document chunk retrieval."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = QdrantClient(path=str(settings.vector_db_path))
        self._embeddings = BedrockEmbeddings(
            model_id=settings.bedrock_embedding_model_id,
            region_name=settings.aws_region,
        )
        self._chunk_lookup: dict[str, EvidenceChunk] = {}

    def build_index(self) -> int:
        chunks = self._load_and_chunk_pdfs(self._settings.documents_dir)
        if not chunks:
            raise ValueError("No PDF content found to index.")

        vector_size = len(self._embeddings.embed_query("dimension probe"))
        self._client.recreate_collection(
            collection_name=self._settings.vector_collection_name,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
        )

        points: list[qmodels.PointStruct] = []
        vectors = self._embeddings.embed_documents([chunk.text for chunk in chunks])
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

    def _load_and_chunk_pdfs(self, docs_dir: Path) -> list[EvidenceChunk]:
        pdfs = sorted(docs_dir.glob("*.pdf"))
        chunks: list[EvidenceChunk] = []

        for pdf_path in pdfs:
            text = self._extract_text(pdf_path)
            split_texts = self._chunk_text(
                text=text,
                chunk_size=self._settings.chunk_size,
                overlap=self._settings.chunk_overlap,
            )
            for index, chunk_text in enumerate(split_texts):
                chunk_id = f"{pdf_path.stem}-{index:04d}"
                chunks.append(
                    EvidenceChunk(
                        chunk_id=chunk_id,
                        source=pdf_path.name,
                        text=chunk_text,
                    )
                )
        return chunks

    @staticmethod
    def _extract_text(pdf_path: Path) -> str:
        reader = PdfReader(str(pdf_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

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
