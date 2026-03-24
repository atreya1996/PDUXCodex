from __future__ import annotations

from pathlib import Path

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
