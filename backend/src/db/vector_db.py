from __future__ import annotations

from collections import Counter
import hashlib
import math
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
_SPARSE_TOKEN_REGEX = re.compile(r"[a-z0-9]+")


class VectorDbManager:
    """Persistent local Qdrant manager for document chunk retrieval."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = QdrantClient(
            path=str(settings.vector_db_path),
            force_disable_check_same_thread=True
        )
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
        fetch_limit = max(top_k * 20, 80)
        retrieval_mode = self._settings.retrieval_mode.strip().lower()
        if retrieval_mode not in {"dense", "sparse", "hybrid"}:
            retrieval_mode = "hybrid"

        dense_candidates: list[EvidenceChunk] = []
        sparse_candidates: list[EvidenceChunk] = []

        if retrieval_mode in {"dense", "hybrid"}:
            dense_candidates = self._search_dense(query=query, fetch_limit=fetch_limit)

        if retrieval_mode in {"sparse", "hybrid"}:
            sparse_candidates = self._search_sparse(query=query, fetch_limit=fetch_limit)

        candidate_hits = self._fuse_candidates(
            dense_candidates=dense_candidates,
            sparse_candidates=sparse_candidates,
            mode=retrieval_mode,
        )

        stale_sources: set[str] = set()
        filtered_hits: list[EvidenceChunk] = []
        for hit in candidate_hits:
            if not self._source_exists(hit.source):
                stale_sources.add(hit.source)
                continue
            filtered_hits.append(hit)

        if stale_sources:
            self.prune_stale_sources(stale_sources)

        if not filtered_hits:
            return []

        # Keep retrieval robust for comparison questions by preferring source diversity
        # in the primary top_k set when multiple sources are available.
        primary_hits: list[EvidenceChunk] = []
        selected_ids: set[str] = set()
        seen_sources: set[str] = set()

        for hit in filtered_hits:
            if hit.source in seen_sources:
                continue
            primary_hits.append(hit)
            selected_ids.add(hit.chunk_id)
            seen_sources.add(hit.source)
            if len(primary_hits) >= top_k:
                break

        if len(primary_hits) < top_k:
            for hit in filtered_hits:
                if hit.chunk_id in selected_ids:
                    continue
                primary_hits.append(hit)
                selected_ids.add(hit.chunk_id)
                if len(primary_hits) >= top_k:
                    break

        neighbor_span = max(0, self._settings.retrieval_neighbor_span)
        if neighbor_span == 0 or not primary_hits:
            return primary_hits

        ordered_hits: list[EvidenceChunk] = list(primary_hits)
        seen_ids = {chunk.chunk_id for chunk in ordered_hits}
        score_by_id = {chunk.chunk_id: chunk.score for chunk in primary_hits}

        for chunk in primary_hits:
            for neighbor_id in self._neighbor_chunk_ids(chunk.chunk_id, neighbor_span):
                if neighbor_id in seen_ids:
                    continue
                neighbor_chunk = self._chunk_lookup.get(neighbor_id)
                if neighbor_chunk is None or neighbor_chunk.source != chunk.source:
                    continue
                seen_ids.add(neighbor_id)
                ordered_hits.append(
                    neighbor_chunk.model_copy(
                        update={
                            "score": score_by_id.get(chunk.chunk_id, chunk.score) * 0.98,
                        }
                    )
                )

        max_hits = top_k + (top_k * neighbor_span * 2)
        return ordered_hits[:max_hits]

    def _search_dense(self, *, query: str, fetch_limit: int) -> list[EvidenceChunk]:
        vector = self._embeddings.embed_query(query)
        response = self._client.query_points(
            collection_name=self._settings.vector_collection_name,
            query=vector,
            limit=fetch_limit,
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

    def _search_sparse(self, *, query: str, fetch_limit: int) -> list[EvidenceChunk]:
        query_terms = self._tokenize_for_sparse(query)
        if not query_terms:
            return []

        documents: list[tuple[EvidenceChunk, Counter[str], int]] = []
        document_frequency: Counter[str] = Counter()
        total_length = 0

        for chunk in self._chunk_lookup.values():
            if not self._source_exists(chunk.source):
                continue
            tokens = self._tokenize_for_sparse(chunk.text)
            if not tokens:
                continue
            tf = Counter(tokens)
            length = len(tokens)
            documents.append((chunk, tf, length))
            total_length += length
            for term in tf.keys():
                document_frequency[term] += 1

        if not documents:
            return []

        num_docs = len(documents)
        avg_doc_len = max(1e-6, total_length / num_docs)
        k1 = 1.2
        b = 0.75
        scored: list[EvidenceChunk] = []

        for chunk, tf, doc_len in documents:
            score = 0.0
            for term in query_terms:
                term_freq = tf.get(term, 0)
                if term_freq == 0:
                    continue
                doc_freq = document_frequency.get(term, 0)
                idf = math.log(1.0 + (num_docs - doc_freq + 0.5) / (doc_freq + 0.5))
                denom = term_freq + k1 * (1.0 - b + b * (doc_len / avg_doc_len))
                score += idf * ((term_freq * (k1 + 1.0)) / max(1e-6, denom))

            if score > 0.0:
                scored.append(chunk.model_copy(update={"score": score}))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:fetch_limit]

    def _fuse_candidates(
        self,
        *,
        dense_candidates: list[EvidenceChunk],
        sparse_candidates: list[EvidenceChunk],
        mode: str,
    ) -> list[EvidenceChunk]:
        if mode == "dense":
            return dense_candidates
        if mode == "sparse":
            return sparse_candidates

        dense_by_id = {item.chunk_id: item for item in dense_candidates}
        sparse_by_id = {item.chunk_id: item for item in sparse_candidates}

        dense_norm = self._normalize_scores(dense_candidates)
        sparse_norm = self._normalize_scores(sparse_candidates)

        dense_weight = max(0.0, self._settings.retrieval_dense_weight)
        sparse_weight = max(0.0, self._settings.retrieval_sparse_weight)
        if dense_weight == 0.0 and sparse_weight == 0.0:
            dense_weight = 1.0

        fused: list[EvidenceChunk] = []
        all_ids = set(dense_by_id.keys()) | set(sparse_by_id.keys())
        for chunk_id in all_ids:
            dense_chunk = dense_by_id.get(chunk_id)
            sparse_chunk = sparse_by_id.get(chunk_id)
            base = dense_chunk or sparse_chunk
            if base is None:
                continue
            score = dense_weight * dense_norm.get(chunk_id, 0.0) + sparse_weight * sparse_norm.get(chunk_id, 0.0)
            fused.append(base.model_copy(update={"score": score}))

        fused.sort(key=lambda item: item.score, reverse=True)
        return fused

    @staticmethod
    def _normalize_scores(chunks: list[EvidenceChunk]) -> dict[str, float]:
        if not chunks:
            return {}
        values = [item.score for item in chunks]
        minimum = min(values)
        maximum = max(values)
        if maximum <= minimum:
            return {item.chunk_id: 1.0 for item in chunks}
        scale = maximum - minimum
        return {item.chunk_id: (item.score - minimum) / scale for item in chunks}

    @staticmethod
    def _tokenize_for_sparse(text: str) -> list[str]:
        return _SPARSE_TOKEN_REGEX.findall(text.lower())

    @staticmethod
    def _neighbor_chunk_ids(chunk_id: str, span: int) -> list[str]:
        prefix, sep, suffix = chunk_id.rpartition("-")
        if not sep or not suffix.isdigit():
            return []

        index = int(suffix)
        width = len(suffix)
        neighbors: list[str] = []
        for offset in range(1, span + 1):
            left = index - offset
            right = index + offset
            if left >= 0:
                neighbors.append(f"{prefix}-{left:0{width}d}")
            neighbors.append(f"{prefix}-{right:0{width}d}")
        return neighbors

    def fetch_by_ids(self, chunk_ids: list[str]) -> list[EvidenceChunk]:
        return [self._chunk_lookup[cid] for cid in chunk_ids if cid in self._chunk_lookup]

    def count_chunks_for_source(self, source_name: str) -> int:
        return sum(1 for chunk in self._chunk_lookup.values() if chunk.source == source_name)

    def delete_source(self, source_name: str) -> None:
        with self._write_lock:
            self._delete_source_locked(source_name)

    def prune_stale_sources(self, candidates: set[str] | None = None) -> int:
        with self._write_lock:
            if candidates is None:
                candidates = {chunk.source for chunk in self._chunk_lookup.values()}

            stale = [source for source in candidates if source and not self._source_exists(source)]
            for source in stale:
                self._delete_source_locked(source)
            return len(stale)

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

    def _source_exists(self, source_name: str) -> bool:
        return (self._settings.documents_dir / source_name).is_file()
