from __future__ import annotations

from payday.config import SupabaseSettings
from payday.models import UploadedAsset


class StorageService:
    """Storage abstraction for Supabase buckets or local mock storage."""

    def __init__(self, settings: SupabaseSettings) -> None:
        self.settings = settings

    def store_asset(self, asset: UploadedAsset, sample_mode: bool = False) -> bool:
        if sample_mode:
            return True
        return bool(self.settings.url and (self.settings.anon_key or self.settings.service_role_key) and asset.raw_bytes)
