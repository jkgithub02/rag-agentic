from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import Settings
from src.core.models import ConflictPolicy, UploadStatus
from src.services.upload_service import UploadService, UploadValidationError


class FakeVectorDb:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def index_file(self, file_path: Path, source_name: str, *, replace_existing: bool) -> int:
        self.calls.append(
            {
                "file_path": file_path,
                "source_name": source_name,
                "replace_existing": replace_existing,
            }
        )
        return 2

    def count_chunks_for_source(self, source_name: str) -> int:
        return 2 if source_name == "paper.md" else 0

    def delete_source(self, source_name: str) -> None:
        self.calls.append({"deleted_source": source_name})

    def build_index(self) -> int:
        self.calls.append({"rebuilt": True})
        return 0


def test_upload_rejects_unsupported_extension(tmp_path: Path) -> None:
    service = UploadService(
        settings=Settings(documents_dir=tmp_path, upload_max_file_size_mb=25),
        vector_db=FakeVectorDb(),
    )

    with pytest.raises(UploadValidationError):
        service.upload_bytes(
            filename="notes.csv",
            content=b"hello",
            conflict_policy=ConflictPolicy.ASK,
        )


def test_upload_conflict_returns_ask_response(tmp_path: Path) -> None:
    vector_db = FakeVectorDb()
    service = UploadService(
        settings=Settings(documents_dir=tmp_path, upload_max_file_size_mb=25),
        vector_db=vector_db,
    )
    existing = tmp_path / "paper.txt"
    existing.write_text("already here", encoding="utf-8")

    response = service.upload_bytes(
        filename="paper.txt",
        content=b"new payload",
        conflict_policy=ConflictPolicy.ASK,
    )

    assert response.status == UploadStatus.CONFLICT
    assert response.existing_filename == "paper.txt"
    assert response.suggested_filename == "paper (1).txt"
    assert len(vector_db.calls) == 0


def test_upload_keep_both_uses_windows_suffix(tmp_path: Path) -> None:
    vector_db = FakeVectorDb()
    service = UploadService(
        settings=Settings(documents_dir=tmp_path, upload_max_file_size_mb=25),
        vector_db=vector_db,
    )

    (tmp_path / "paper.txt").write_text("v1", encoding="utf-8")
    (tmp_path / "paper (1).txt").write_text("v2", encoding="utf-8")

    response = service.upload_bytes(
        filename="paper.txt",
        content=b"v3",
        conflict_policy=ConflictPolicy.KEEP_BOTH,
    )

    assert response.status == UploadStatus.SUCCESS
    assert response.stored_filename == "paper (2).txt"
    assert (tmp_path / "paper (2).txt").exists()
    assert vector_db.calls[0]["replace_existing"] is False


def test_upload_replace_overwrites_existing_file(tmp_path: Path) -> None:
    vector_db = FakeVectorDb()
    service = UploadService(
        settings=Settings(documents_dir=tmp_path, upload_max_file_size_mb=25),
        vector_db=vector_db,
    )

    target = tmp_path / "paper.md"
    target.write_text("old", encoding="utf-8")

    response = service.upload_bytes(
        filename="paper.md",
        content=b"new",
        conflict_policy=ConflictPolicy.REPLACE,
    )

    assert response.status == UploadStatus.SUCCESS
    assert response.stored_filename == "paper.md"
    assert target.read_bytes() == b"new"
    assert vector_db.calls[0]["replace_existing"] is True


def test_list_documents_returns_supported_files_with_chunk_counts(tmp_path: Path) -> None:
    vector_db = FakeVectorDb()
    service = UploadService(
        settings=Settings(documents_dir=tmp_path, upload_max_file_size_mb=25),
        vector_db=vector_db,
    )

    (tmp_path / "paper.md").write_text("content", encoding="utf-8")
    (tmp_path / "ignore.csv").write_text("content", encoding="utf-8")

    docs = service.list_documents()

    assert len(docs) == 1
    assert docs[0]["filename"] == "paper.md"
    assert docs[0]["chunks_indexed"] == 2


def test_delete_document_removes_file_and_index_source(tmp_path: Path) -> None:
    vector_db = FakeVectorDb()
    service = UploadService(
        settings=Settings(documents_dir=tmp_path, upload_max_file_size_mb=25),
        vector_db=vector_db,
    )

    target = tmp_path / "paper.md"
    target.write_text("content", encoding="utf-8")

    deleted = service.delete_document("paper.md")

    assert deleted is True
    assert not target.exists()
    assert any(call.get("deleted_source") == "paper.md" for call in vector_db.calls)


def test_delete_all_documents_removes_supported_files_and_rebuilds_index(tmp_path: Path) -> None:
    vector_db = FakeVectorDb()
    service = UploadService(
        settings=Settings(documents_dir=tmp_path, upload_max_file_size_mb=25),
        vector_db=vector_db,
    )

    (tmp_path / "paper.md").write_text("content", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("content", encoding="utf-8")
    (tmp_path / "keep.json").write_text("content", encoding="utf-8")

    deleted_count = service.delete_all_documents()

    assert deleted_count == 2
    assert not (tmp_path / "paper.md").exists()
    assert not (tmp_path / "notes.txt").exists()
    assert (tmp_path / "keep.json").exists()
    assert any(call.get("rebuilt") is True for call in vector_db.calls)


def test_upload_rejects_files_exceeding_max_size(tmp_path: Path) -> None:
    """Test: Files larger than max_file_size_mb are rejected with validation error.
    
    Ensures upload service enforces size limits before processing to prevent
    memory exhaustion or processing of unexpectedly large files.
    """
    service = UploadService(
        settings=Settings(documents_dir=tmp_path, upload_max_file_size_mb=1),  # 1 MB limit
        vector_db=FakeVectorDb(),
    )

    # Create content larger than 1 MB
    oversized_content = b"x" * (2 * 1024 * 1024)  # 2 MB

    with pytest.raises(UploadValidationError):
        service.upload_bytes(
            filename="oversized.md",
            content=oversized_content,
            conflict_policy=ConflictPolicy.ASK,
        )
