from __future__ import annotations

from typing import Any

from payday.config import TranscriptionSettings
from payday.models import Transcript, UploadedAsset


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
            self._client = client

        try:
            response = client.audio.transcriptions.create(
                file=(asset.filename, asset.raw_bytes, asset.content_type),
                model=self.settings.model,
            )
        except APITimeoutError as exc:
            raise TimeoutError(
                f"OpenAI transcription timed out for {asset.filename} after {self.settings.timeout_seconds} seconds."
            ) from exc
        except APIConnectionError as exc:
            raise ConnectionError(f"OpenAI transcription connection failed for {asset.filename}.") from exc
        except APIStatusError as exc:
            request_id = getattr(exc, "request_id", None) or getattr(exc, "_request_id", None)
            status_code = getattr(exc, "status_code", "unknown")
            request_details = f" request_id={request_id}" if request_id else ""
            raise RuntimeError(
                f"OpenAI transcription failed for {asset.filename} with status {status_code}.{request_details}"
            ) from exc

        text = getattr(response, "text", "")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(f"OpenAI transcription returned empty text for {asset.filename}.")

        return self._build_transcript(
            asset=asset,
            sample_mode=False,
            text=text.strip(),
            extra_metadata={
                "response_format": getattr(response, "response_format", None),
                "language": getattr(response, "language", None),
                "duration_seconds": getattr(response, "duration", None),
            },
        )

    def _build_transcript(
        self,
        *,
        asset: UploadedAsset,
        sample_mode: bool,
        text: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Transcript:
        metadata: dict[str, Any] = {
            "sample_mode": sample_mode,
            "filename": asset.filename,
            "file_id": asset.file_id,
            "content_type": asset.content_type,
            "size_bytes": asset.size_bytes,
            "source": asset.filename,
        }
        if extra_metadata:
            metadata.update({key: value for key, value in extra_metadata.items() if value is not None})
        return Transcript(
            text=text,
            provider=self.settings.provider,
            model=self.settings.model,
            metadata=metadata,
        )
