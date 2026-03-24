from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

from payday.models import UploadedAsset

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UPLOAD_DIR = REPO_ROOT / "data" / "uploads"
DEFAULT_MAX_FILE_SIZE_BYTES = 25_000_000

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

_EXTENSION_TO_ALLOWED_CONTENT_TYPES = {
    "aac": {"audio/aac", "audio/x-aac"},
    "m4a": {"audio/mp4", "audio/m4a", "audio/x-m4a"},
    "mp3": {"audio/mpeg", "audio/mp3"},
    "mp4": {"video/mp4", "audio/mp4"},
    "mpeg": {"audio/mpeg", "video/mpeg"},
    "mpga": {"audio/mpeg", "audio/mpga"},
    "ogg": {"audio/ogg"},
    "wav": {"audio/wav", "audio/x-wav", "audio/wave"},
    "webm": {"audio/webm", "video/webm"},
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

    def __init__(
        self,
        *,
        upload_dir: Path | None = None,
        max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    ) -> None:
        self.upload_dir = (upload_dir or DEFAULT_UPLOAD_DIR).resolve()
        self.max_file_size_bytes = max_file_size_bytes

    def create_asset(self, filename: str, content_type: str, data: bytes, file_id: str | None = None) -> UploadedAsset:
        extension = self._extract_extension(filename)
        if extension not in SUPPORTED_UPLOAD_EXTENSIONS:
            raise UploadValidationError(
                f"Unsupported upload format '{extension or 'unknown'}'. Supported formats: {', '.join(SUPPORTED_UPLOAD_EXTENSIONS)}."
            )

        self._validate_file_size(filename=filename, data=data)
        safe_filename = self._sanitize_filename(filename)
        normalized_content_type = self._normalize_content_type(
            filename=safe_filename,
            content_type=content_type,
        )
        self._validate_content_type(filename=safe_filename, content_type=normalized_content_type)
        normalized_file_id = file_id or uuid4().hex
        file_path = self._persist_upload(file_id=normalized_file_id, safe_filename=safe_filename, data=data)
        return UploadedAsset(
            filename=safe_filename,
            content_type=normalized_content_type,
            size_bytes=len(data),
            file_path=str(file_path),
            raw_bytes=data,
            file_id=normalized_file_id,
        )

    def _normalize_content_type(self, *, filename: str, content_type: str) -> str:
        extension = self._extract_extension(filename)
        normalized = content_type.strip().lower()
        if normalized in _GENERIC_CONTENT_TYPES:
            return _EXTENSION_TO_CONTENT_TYPE[extension]
        return normalized

    def _validate_content_type(self, *, filename: str, content_type: str) -> None:
        extension = self._extract_extension(filename)
        allowed_types = _EXTENSION_TO_ALLOWED_CONTENT_TYPES[extension]
        if content_type in allowed_types:
            return
        raise UploadValidationError(
            f"File '{filename}' has extension '.{extension}' but content type '{content_type}'. "
            f"Use one of: {', '.join(sorted(allowed_types))}."
        )

    def _validate_file_size(self, *, filename: str, data: bytes) -> None:
        size_bytes = len(data)
        if size_bytes == 0:
            raise UploadValidationError(
                f"File '{filename}' is empty. Upload a non-empty recording and try again."
            )
        if size_bytes <= self.max_file_size_bytes:
            return
        max_size_mb = self.max_file_size_bytes / 1_000_000
        actual_size_mb = size_bytes / 1_000_000
        raise UploadValidationError(
            f"File '{filename}' is {actual_size_mb:.1f} MB, which exceeds the {max_size_mb:.0f} MB upload limit. "
            "Compress or split the recording and try again."
        )

    def _persist_upload(self, *, file_id: str, safe_filename: str, data: bytes) -> Path:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        persisted_path = self.upload_dir / f"{file_id}_{safe_filename}"
        persisted_path.write_bytes(data)
        return persisted_path

    def _sanitize_filename(self, filename: str) -> str:
        basename = Path(filename).name.strip()
        if not basename:
            raise UploadValidationError("Filename is empty. Provide a valid audio filename and retry.")
        sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", basename)
        sanitized = re.sub(r"_+", "_", sanitized).strip("._")
        if not sanitized:
            raise UploadValidationError(
                f"Filename '{filename}' could not be sanitized safely. Rename the file and try again."
            )
        return sanitized

    def _extract_extension(self, filename: str) -> str:
        return Path(filename).suffix.lower().lstrip('.')
