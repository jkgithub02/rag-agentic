from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Settings
from src.models import ConflictPolicy, UploadStatus
from src.upload_service import UploadService, UploadValidationError


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
