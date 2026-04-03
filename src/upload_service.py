from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

from src.config import Settings
from src.db.vector_db import VectorDbManager
from src.models import ConflictPolicy, UploadResponse, UploadStatus

_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class UploadValidationError(ValueError):
    """Raised when an uploaded file violates validation constraints."""


class UploadService:
    """Handles upload validation, conflict resolution, and indexing."""

    def __init__(self, settings: Settings, vector_db: VectorDbManager) -> None:
        self._settings = settings
        self._vector_db = vector_db

    def upload_bytes(
        self,
        *,
        filename: str,
        content: bytes,
        conflict_policy: ConflictPolicy,
    ) -> UploadResponse:
        clean_name = self._sanitize_filename(filename)
        self._validate_upload(clean_name, content)

        docs_dir = self._settings.documents_dir
        docs_dir.mkdir(parents=True, exist_ok=True)

        requested_target = docs_dir / clean_name
        if requested_target.exists() and conflict_policy == ConflictPolicy.ASK:
            suggestion = self._next_available_name(requested_target)
            return UploadResponse(
                status=UploadStatus.CONFLICT,
                message="File already exists. Choose replace or keep_both.",
                original_filename=clean_name,
                existing_filename=clean_name,
                suggested_filename=suggestion.name,
                conflict_options=[ConflictPolicy.REPLACE, ConflictPolicy.KEEP_BOTH],
            )

        if requested_target.exists() and conflict_policy == ConflictPolicy.KEEP_BOTH:
            final_target = self._next_available_name(requested_target)
            replace_existing = False
        else:
            final_target = requested_target
            replace_existing = (
                requested_target.exists() and conflict_policy == ConflictPolicy.REPLACE
            )

        self._atomic_write(final_target, content)
        chunks_added = self._vector_db.index_file(
            file_path=final_target,
            source_name=final_target.name,
            replace_existing=replace_existing,
        )

        action = "replaced" if replace_existing else "stored"
        return UploadResponse(
            status=UploadStatus.SUCCESS,
            message=f"File {action} and indexed successfully.",
            original_filename=clean_name,
            stored_filename=final_target.name,
            chunks_added=chunks_added,
        )

    def _validate_upload(self, filename: str, content: bytes) -> None:
        suffix = Path(filename).suffix.lower()
        supported = {ext.lower() for ext in self._settings.allowed_upload_extensions}
        if suffix not in supported:
            allowed = ", ".join(sorted(ext.lstrip(".") for ext in supported))
            raise UploadValidationError(f"Unsupported file type '{suffix}'. Allowed: {allowed}.")

        if not content:
            raise UploadValidationError("Uploaded file is empty.")

        max_bytes = self._settings.upload_max_file_size_mb * 1024 * 1024
        if len(content) > max_bytes:
            raise UploadValidationError(
                f"File exceeds {self._settings.upload_max_file_size_mb} MB upload limit."
            )

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        candidate = Path(filename).name.strip()
        if not candidate:
            raise UploadValidationError("Filename is required.")

        candidate = _INVALID_FILENAME_CHARS.sub("_", candidate)
        if candidate in {".", ".."}:
            raise UploadValidationError("Filename is invalid.")
        return candidate

    @staticmethod
    def _next_available_name(target: Path) -> Path:
        stem = target.stem
        suffix = target.suffix
        counter = 1
        while True:
            candidate = target.with_name(f"{stem} ({counter}){suffix}")
            if not candidate.exists():
                return candidate
            counter += 1

    @staticmethod
    def _atomic_write(target: Path, content: bytes) -> None:
        temp_target = target.with_name(f".{target.name}.{uuid4().hex}.uploadtmp")
        temp_target.write_bytes(content)
        temp_target.replace(target)
