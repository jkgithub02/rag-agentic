from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app
from src.bootstrap import get_upload_service
from src.core.models import ConflictPolicy, UploadResponse, UploadStatus


class FakeUploadService:
    def __init__(self) -> None:
        self.last_filename: str | None = None
        self.last_policy: ConflictPolicy | None = None
        self.documents: list[dict[str, object]] = []

    def upload_bytes(
        self,
        *,
        filename: str,
        content: bytes,
        conflict_policy: ConflictPolicy,
    ) -> UploadResponse:
        self.last_filename = filename
        self.last_policy = conflict_policy
        del content
        return UploadResponse(
            status=UploadStatus.SUCCESS,
            message="ok",
            original_filename=filename,
            stored_filename=filename,
            chunks_added=1,
        )

    def list_documents(self) -> list[dict[str, object]]:
        return self.documents

    def delete_document(self, source_name: str) -> bool:
        for index, doc in enumerate(self.documents):
            if doc.get("filename") == source_name:
                del self.documents[index]
                return True
        return False

    def delete_all_documents(self) -> int:
        deleted = len(self.documents)
        self.documents = []
        return deleted


def test_upload_endpoint_defaults_to_ask_policy() -> None:
    fake = FakeUploadService()
    app.dependency_overrides[get_upload_service] = lambda: fake

    try:
        client = TestClient(app)
        response = client.post(
            "/upload",
            files={"file": ("paper.txt", b"content", "text/plain")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert fake.last_filename == "paper.txt"
    assert fake.last_policy == ConflictPolicy.ASK


def test_upload_endpoint_respects_replace_policy() -> None:
    fake = FakeUploadService()
    app.dependency_overrides[get_upload_service] = lambda: fake

    try:
        client = TestClient(app)
        response = client.post(
            "/upload",
            files={"file": ("paper.md", b"content", "text/markdown")},
            data={"conflict_policy": "replace"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert fake.last_policy == ConflictPolicy.REPLACE


def test_documents_list_endpoint_returns_documents() -> None:
    fake = FakeUploadService()
    fake.documents = [
        {"filename": "paper.md", "size_bytes": 123, "chunks_indexed": 2},
        {"filename": "notes.txt", "size_bytes": 88, "chunks_indexed": 1},
    ]
    app.dependency_overrides[get_upload_service] = lambda: fake

    try:
        client = TestClient(app)
        response = client.get("/documents")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["documents"]) == 2
    assert payload["documents"][0]["filename"] == "paper.md"


def test_documents_delete_endpoint_returns_404_when_missing() -> None:
    fake = FakeUploadService()
    app.dependency_overrides[get_upload_service] = lambda: fake

    try:
        client = TestClient(app)
        response = client.delete("/documents/missing.txt")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404


def test_documents_delete_endpoint_deletes_existing_document() -> None:
    fake = FakeUploadService()
    fake.documents = [{"filename": "paper.md", "size_bytes": 123, "chunks_indexed": 2}]
    app.dependency_overrides[get_upload_service] = lambda: fake

    try:
        client = TestClient(app)
        response = client.delete("/documents/paper.md")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["deleted"] is True


def test_documents_delete_all_endpoint_returns_deleted_count() -> None:
    fake = FakeUploadService()
    fake.documents = [
        {"filename": "paper.md", "size_bytes": 123, "chunks_indexed": 2},
        {"filename": "notes.txt", "size_bytes": 55, "chunks_indexed": 1},
    ]
    app.dependency_overrides[get_upload_service] = lambda: fake

    try:
        client = TestClient(app)
        response = client.delete("/documents")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["deleted"] is True
    assert payload["deleted_count"] == 2
