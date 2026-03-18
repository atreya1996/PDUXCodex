from __future__ import annotations

from uuid import uuid4

from payday.models import UploadedAsset


class UploadService:
    """Normalizes uploaded files for downstream processing."""

    def create_asset(self, filename: str, content_type: str, data: bytes, file_id: str | None = None) -> UploadedAsset:
        return UploadedAsset(
            filename=filename,
            content_type=content_type,
            size_bytes=len(data),
            raw_bytes=data,
            file_id=file_id or uuid4().hex,
        )
