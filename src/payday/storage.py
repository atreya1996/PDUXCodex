from __future__ import annotations

from pathlib import Path

from payday.config import SupabaseSettings
from payday.models import UploadedAsset


class StorageService:
    """Storage abstraction for Supabase buckets or local mock storage."""

    def __init__(self, settings: SupabaseSettings, storage_client: object | None = None) -> None:
        self.settings = settings
        self.storage_client = storage_client

    def build_audio_path(self, interview_id: str, filename: str) -> str:
        normalized_filename = Path(filename).name.replace(" ", "_")
        return f"audio/{interview_id}/{normalized_filename}"

    def upload_audio(self, asset: UploadedAsset, interview_id: str, sample_mode: bool = False) -> str:
        object_path = self.build_audio_path(interview_id=interview_id, filename=asset.filename)
        if sample_mode:
            return object_path

        if self.storage_client is None:
            raise RuntimeError(
                "Live storage upload requires a configured storage client or explicit upload implementation."
            )

        bucket_name = self.settings.storage_bucket
        if hasattr(self.storage_client, "from_"):
            bucket = self.storage_client.from_(bucket_name)
            bucket.upload(
                object_path,
                asset.raw_bytes,
                file_options={"content-type": asset.content_type},
            )
            return object_path

        upload = getattr(self.storage_client, "upload", None)
        if callable(upload):
            upload(bucket_name, object_path, asset.raw_bytes, content_type=asset.content_type)
            return object_path

        raise TypeError("Unsupported storage client: expected Supabase-compatible upload interface.")

    def store_asset(
        self,
        asset: UploadedAsset,
        sample_mode: bool = False,
        interview_id: str | None = None,
    ) -> bool:
        if sample_mode:
            return True
        if interview_id is None:
            raise ValueError("interview_id is required to store assets when sample mode is disabled.")
        return bool(
            self.upload_audio(
                asset=asset,
                interview_id=interview_id,
                sample_mode=sample_mode,
            )
        )
