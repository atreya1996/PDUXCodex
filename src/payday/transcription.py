from __future__ import annotations

from payday.config import TranscriptionSettings
from payday.models import Transcript, UploadedAsset


class TranscriptionService:
    """Provider facade for audio/video transcription."""

    def __init__(self, settings: TranscriptionSettings) -> None:
        self.settings = settings

    def transcribe(self, asset: UploadedAsset, sample_mode: bool = False) -> Transcript:
        if sample_mode:
            decoded_text = asset.raw_bytes.decode("utf-8", errors="ignore").strip()
            text = decoded_text or (
                "Sample transcript for MVP mode. Replace this stub with a real provider call "
                "when credentials are configured."
            )
        else:
            text = (
                f"Placeholder transcript for {asset.filename}. Integrate the "
                f"{self.settings.provider} provider in this service."
            )
        return Transcript(
            text=text,
            provider=self.settings.provider,
            model=self.settings.model,
            metadata={"sample_mode": sample_mode, "source": asset.filename},
        )
