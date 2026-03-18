from __future__ import annotations

from typing import Any

from payday.config import TranscriptionSettings
from payday.models import Transcript, UploadedAsset


class TranscriptionProviderError(RuntimeError):
    """Raised when a configured transcription provider is unavailable or unsupported."""


class TranscriptionService:
    """Provider facade for audio/video transcription."""

    def __init__(self, settings: TranscriptionSettings, client: object | None = None) -> None:
        self.settings = settings
        self._client = client

    def transcribe(self, asset: UploadedAsset, sample_mode: bool = False) -> Transcript:
        if sample_mode:
            return self._sample_transcript(asset)
        return self._provider_transcript(asset)

    def _sample_transcript(self, asset: UploadedAsset) -> Transcript:
        decoded_text = asset.raw_bytes.decode("utf-8", errors="ignore").strip()
        text = decoded_text or (
            "Sample transcript for MVP mode. Replace this stub with a real provider call "
            "when credentials are configured."
        )
        return self._build_transcript(asset=asset, sample_mode=True, text=text)

    def _provider_transcript(self, asset: UploadedAsset) -> Transcript:
        provider = self.settings.provider.strip().lower()
        if provider == "openai":
            return self._transcribe_with_openai(asset)
        raise ValueError(f"Unsupported transcription provider: {self.settings.provider}")

    def _transcribe_with_openai(self, asset: UploadedAsset) -> Transcript:
        from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

        if not self.settings.api_key:
            raise ValueError("TRANSCRIPTION_API_KEY must be configured when sample mode is disabled.")

        client = self._client
        if client is None:
            client = OpenAI(
                api_key=self.settings.api_key,
                max_retries=0,
                timeout=self.settings.timeout_seconds,
            )
            return Transcript(
                text=text,
                provider=self.settings.provider,
                model=self.settings.model,
                metadata={"sample_mode": sample_mode, "source": asset.filename},
            )

        raise TranscriptionProviderError(
            f"{self.settings.provider} transcription is not implemented yet. Keep "
            "PAYDAY_USE_SAMPLE_MODE=true to use sample transcripts, or add the live provider "
            "implementation for this branch."
        )


class OpenAITranscriptionService(TranscriptionService):
    """Reserved provider branch so OpenAI transcription can be implemented without changing callers."""



def build_transcription_service(
    settings: TranscriptionSettings,
    *,
    sample_mode: bool,
) -> TranscriptionService:
    """Select the transcription service from environment-controlled provider settings."""

    if sample_mode:
        return TranscriptionService(settings)
    if settings.provider == "openai":
        return OpenAITranscriptionService(settings)
    raise TranscriptionProviderError(
        f"Unsupported transcription provider '{settings.provider}'. Use TRANSCRIPTION_PROVIDER=openai."
    )
