from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

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

    def delete_asset(self, audio_url: str, *, sample_mode: bool = False) -> bool:
        object_path = self._object_path_from_audio_url(audio_url)
        if not object_path:
            return False
        if sample_mode:
            return True
        if self.storage_client is None:
            raise RuntimeError(
                "Live storage deletion requires a configured storage client or explicit delete implementation."
            )

        bucket_name = self.settings.storage_bucket
        if hasattr(self.storage_client, "from_"):
            bucket = self.storage_client.from_(bucket_name)
            remove = getattr(bucket, "remove", None)
            if callable(remove):
                remove([object_path])
                return True

        remove = getattr(self.storage_client, "remove", None)
        if callable(remove):
            remove(bucket_name, object_path)
            return True

        delete = getattr(self.storage_client, "delete", None)
        if callable(delete):
            delete(bucket_name, object_path)
            return True

        raise TypeError("Unsupported storage client: expected Supabase-compatible delete interface.")

    def _object_path_from_audio_url(self, audio_url: str) -> str:
        parsed = urlparse(audio_url)
        raw_path = (parsed.path or audio_url).lstrip("/")
        if not raw_path:
            return ""

        bucket_name = self.settings.storage_bucket.strip("/")
        bucket_markers = (
            f"object/public/{bucket_name}/",
            f"object/sign/{bucket_name}/",
            f"public/{bucket_name}/",
            f"sign/{bucket_name}/",
            f"{bucket_name}/",
        )
        for marker in bucket_markers:
            if marker in raw_path:
                return raw_path.split(marker, maxsplit=1)[1]
        return raw_path
