from __future__ import annotations

import json
from pathlib import Path

from pypdf import PdfReader

from agentic_rag.config import Settings, get_settings
from agentic_rag.models import EvidenceChunk


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")

    chunks: list[str] = []
    start = 0
    cleaned_text = " ".join(text.split())
    while start < len(cleaned_text):
        end = min(start + chunk_size, len(cleaned_text))
        chunk = cleaned_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(cleaned_text):
            break
        start = end - overlap
    return chunks


def _extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def build_chunks(settings: Settings) -> list[EvidenceChunk]:
    pdf_files = sorted(settings.documents_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {settings.documents_dir}")

    all_chunks: list[EvidenceChunk] = []
    for pdf_file in pdf_files:
        text = _extract_pdf_text(pdf_file)
        split_chunks = _chunk_text(
            text=text,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        for index, chunk in enumerate(split_chunks):
            all_chunks.append(
                EvidenceChunk(
                    chunk_id=f"{pdf_file.stem}-{index:04d}",
                    source=pdf_file.name,
                    text=chunk,
                )
            )
    return all_chunks


def save_chunks(chunks: list[EvidenceChunk], index_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_data = [chunk.model_dump() for chunk in chunks]
    index_path.write_text(json.dumps(index_data, ensure_ascii=True, indent=2), encoding="utf-8")


def load_chunks(index_path: Path) -> list[EvidenceChunk]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return [EvidenceChunk.model_validate(item) for item in payload]


def ensure_index(settings: Settings) -> list[EvidenceChunk]:
    if settings.index_file.exists():
        return load_chunks(settings.index_file)

    chunks = build_chunks(settings)
    save_chunks(chunks, settings.index_file)
    return chunks


def main() -> None:
    settings = get_settings()
    chunks = ensure_index(settings)
    print(f"Indexed {len(chunks)} chunks into {settings.index_file}")


if __name__ == "__main__":
    main()
