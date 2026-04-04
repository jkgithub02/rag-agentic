from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app
from src.bootstrap import get_upload_service
from src.core.models import ConflictPolicy, UploadResponse, UploadStatus


class FakeUploadService:
    def __init__(self) -> None:
        self.last_filename: str | None = None
        self.last_policy: ConflictPolicy | None = None

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
