from __future__ import annotations

from pathlib import Path
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from payday.models import UploadedAsset


class StorageService:
    """Local filesystem storage for uploaded interview audio."""

    def __init__(self, root_directory: str | Path = "./data/uploads") -> None:
        self.root_directory = Path(root_directory).expanduser()

    def build_file_path(self, interview_id: str, filename: str) -> str:
        normalized_filename = Path(filename).name.replace(" ", "_")
        return str(self.root_directory / interview_id / normalized_filename)

    # Backward-compatible alias while call sites migrate.
    def build_audio_path(self, interview_id: str, filename: str) -> str:
        return self.build_file_path(interview_id=interview_id, filename=filename)

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

        file_path = Path(self.build_file_path(interview_id=interview_id, filename=asset.filename))
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(asset.raw_bytes)
        return True

    def delete_asset(self, file_path: str, *, sample_mode: bool = False) -> bool:
        if sample_mode:
            return True
        path = Path(file_path)
        if not path.exists():
            return True
        if path.is_dir():
            return False
        path.unlink(missing_ok=True)
        return True


class SupabaseStorageService(StorageService):
    """Supabase object storage for interview audio assets."""

    def __init__(
        self,
        *,
        supabase_url: str,
        service_role_key: str,
        storage_bucket: str,
        fallback_root_directory: str | Path = "./data/uploads",
        client: object | None = None,
    ) -> None:
        super().__init__(root_directory=fallback_root_directory)
        self.supabase_url = supabase_url.rstrip("/")
        self.service_role_key = service_role_key
        self.storage_bucket = storage_bucket
        self.client = client

    def build_file_path(self, interview_id: str, filename: str) -> str:
        normalized_filename = Path(filename).name.replace(" ", "_")
        return f"audio/{interview_id}/{normalized_filename}"

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

        object_key = self.build_file_path(interview_id=interview_id, filename=asset.filename)
        self._upload_bytes(object_key=object_key, payload=asset.raw_bytes, content_type=asset.content_type)
        return True

    def delete_asset(self, file_path: str, *, sample_mode: bool = False) -> bool:
        if sample_mode:
            return True
        object_key = self._coerce_object_key(file_path)
        if not object_key:
            return True
        self._delete_object(object_key=object_key)
        return True

    def _upload_bytes(self, *, object_key: str, payload: bytes, content_type: str) -> None:
        if self.client is not None:
            bucket = getattr(self.client, "from_")(self.storage_bucket)
            bucket.upload(
                object_key,
                payload,
                file_options={"content-type": content_type, "upsert": "true"},
            )
            return
        self._require_credentials()
        request = urllib_request.Request(
            url=f"{self.supabase_url}/storage/v1/object/{self.storage_bucket}/{object_key}",
            data=payload,
            method="POST",
            headers={
                "apikey": self.service_role_key,
                "Authorization": f"Bearer {self.service_role_key}",
                "x-upsert": "true",
                "content-type": content_type,
            },
        )
        with urllib_request.urlopen(request):
            return

    def _delete_object(self, *, object_key: str) -> None:
        if self.client is not None:
            bucket = getattr(self.client, "from_")(self.storage_bucket)
            bucket.remove([object_key])
            return
        self._require_credentials()
        request = urllib_request.Request(
            url=f"{self.supabase_url}/storage/v1/object/{self.storage_bucket}/{object_key}",
            method="DELETE",
            headers={
                "apikey": self.service_role_key,
                "Authorization": f"Bearer {self.service_role_key}",
            },
        )
        try:
            with urllib_request.urlopen(request):
                return
        except urllib_error.HTTPError as exc:
            if exc.code == 404:
                return
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Failed to delete Supabase object '{object_key}': {body}") from exc

    def _coerce_object_key(self, file_path: str) -> str:
        normalized = file_path.strip()
        if not normalized:
            return ""
        parsed = urllib_parse.urlparse(normalized)
        if parsed.scheme and parsed.netloc:
            path = parsed.path.lstrip("/")
        else:
            path = normalized.lstrip("/")
        prefix = f"{self.storage_bucket}/"
        if path.startswith(prefix):
            path = path[len(prefix) :]
        return path

    def _require_credentials(self) -> None:
        if not self.supabase_url:
            raise ValueError("SUPABASE_URL is required for SupabaseStorageService.")
        if not self.service_role_key:
            raise ValueError("SUPABASE_SERVICE_ROLE_KEY is required for SupabaseStorageService.")
