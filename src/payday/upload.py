from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from payday.models import UploadedAsset

SUPPORTED_UPLOAD_EXTENSIONS: tuple[str, ...] = (
    "mp3",
    "mp4",
    "mpeg",
    "mpga",
    "m4a",
    "wav",
    "webm",
    "ogg",
    "aac",
)

_EXTENSION_TO_CONTENT_TYPE = {
    "aac": "audio/aac",
    "m4a": "audio/mp4",
    "mp3": "audio/mpeg",
    "mp4": "video/mp4",
    "mpeg": "audio/mpeg",
    "mpga": "audio/mpeg",
    "ogg": "audio/ogg",
    "wav": "audio/wav",
    "webm": "audio/webm",
}

_GENERIC_CONTENT_TYPES = {
    "",
    "application/octet-stream",
    "audio/*",
    "video/*",
}


class UploadValidationError(ValueError):
    """Raised when an uploaded file uses an unsupported extension."""


class UploadService:
    """Normalizes uploaded files for downstream processing."""

    def create_asset(self, filename: str, content_type: str, data: bytes, file_id: str | None = None) -> UploadedAsset:
        extension = self._extract_extension(filename)
        if extension not in SUPPORTED_UPLOAD_EXTENSIONS:
            raise UploadValidationError(
                f"Unsupported upload format '{extension or 'unknown'}'. Supported formats: {', '.join(SUPPORTED_UPLOAD_EXTENSIONS)}."
            )

        normalized_content_type = self._normalize_content_type(filename=filename, content_type=content_type)
        return UploadedAsset(
            filename=filename,
            content_type=normalized_content_type,
            size_bytes=len(data),
            raw_bytes=data,
            file_id=file_id or uuid4().hex,
        )

    def _normalize_content_type(self, *, filename: str, content_type: str) -> str:
        extension = self._extract_extension(filename)
        normalized = content_type.strip().lower()
        if normalized in _GENERIC_CONTENT_TYPES:
            return _EXTENSION_TO_CONTENT_TYPE[extension]
        return normalized

    def _extract_extension(self, filename: str) -> str:
        return Path(filename).suffix.lower().lstrip('.')
