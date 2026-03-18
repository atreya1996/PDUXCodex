from __future__ import annotations

import json
import uuid
from typing import Any, Protocol
from urllib import error, request

from payday.config import TranscriptionSettings
from payday.models import Transcript, UploadedAsset


class ExternalTranscriptionProviderError(RuntimeError):
    """Raised when an external transcription provider cannot transcribe audio."""


class TranscriptionProviderAdapter(Protocol):
    provider_name: str
    model_name: str

    def transcribe(self, asset: UploadedAsset) -> Transcript:
        """Return a transcript produced by an external provider."""


class OpenAITranscriptionAdapter:
    """Concrete adapter for OpenAI-compatible transcription calls."""

    def __init__(
        self,
        settings: TranscriptionSettings,
        *,
        transport: Any = None,
    ) -> None:
        self.settings = settings
        self.provider_name = settings.provider
        self.model_name = settings.model
        self._transport = transport or self._default_transport

    def transcribe(self, asset: UploadedAsset) -> Transcript:
        api_key = self.settings.api_key.strip()
        if not api_key:
            raise ExternalTranscriptionProviderError(
                f"TRANSCRIPTION_API_KEY is required when sample mode is disabled for provider '{self.settings.provider}'."
            )

        response_payload = self._transport(asset=asset, model=self.settings.model, api_key=api_key)
        transcript_text = self._extract_text(response_payload)
        return Transcript(
            text=transcript_text,
            provider=self.provider_name,
            model=self.model_name,
            metadata={"sample_mode": False, "source": asset.filename},
        )

    def _default_transport(self, *, asset: UploadedAsset, model: str, api_key: str) -> dict[str, Any]:
        boundary = f"----PayDayBoundary{uuid.uuid4().hex}"
        body = self._build_multipart_body(asset=asset, model=model, boundary=boundary)
        http_request = request.Request(
            "https://api.openai.com/v1/audio/transcriptions",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        try:
            with request.urlopen(http_request) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:  # pragma: no cover - depends on live provider
            body = exc.read().decode("utf-8", errors="ignore")
            raise ExternalTranscriptionProviderError(
                f"OpenAI transcription request failed with HTTP {exc.code}: {body}"
            ) from exc
        except error.URLError as exc:  # pragma: no cover - depends on live provider
            raise ExternalTranscriptionProviderError(f"OpenAI transcription request failed: {exc.reason}") from exc

    def _build_multipart_body(self, *, asset: UploadedAsset, model: str, boundary: str) -> bytes:
        boundary_bytes = boundary.encode("utf-8")
        lines = [
            b"--" + boundary_bytes,
            b'Content-Disposition: form-data; name="model"',
            b"",
            model.encode("utf-8"),
            b"--" + boundary_bytes,
            (
                f'Content-Disposition: form-data; name="file"; filename="{asset.filename}"'.encode("utf-8")
            ),
            f"Content-Type: {asset.content_type}".encode("utf-8"),
            b"",
            asset.raw_bytes,
            b"--" + boundary_bytes + b"--",
            b"",
        ]
        return b"\r\n".join(lines)

    def _extract_text(self, response_payload: Any) -> str:
        if isinstance(response_payload, str):
            normalized = response_payload.strip()
            if normalized:
                return normalized
            raise ExternalTranscriptionProviderError("OpenAI transcription provider returned an empty response.")
        if not isinstance(response_payload, dict):
            raise ExternalTranscriptionProviderError(
                "OpenAI transcription provider returned an unsupported response shape."
            )

        provider_error = response_payload.get("error")
        if isinstance(provider_error, dict):
            message = provider_error.get("message") or provider_error.get("type") or "unknown provider error"
            raise ExternalTranscriptionProviderError(f"OpenAI transcription provider error: {message}")

        text = response_payload.get("text")
        if isinstance(text, str) and text.strip():
            return text
        raise ExternalTranscriptionProviderError("OpenAI transcription provider response did not include text.")


class TranscriptionService:
    """Provider facade for audio/video transcription."""

    def __init__(
        self,
        settings: TranscriptionSettings,
        *,
        adapter: TranscriptionProviderAdapter | None = None,
    ) -> None:
        self.settings = settings
        self.adapter = adapter

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

        adapter = self.adapter or self._build_default_adapter()
        return adapter.transcribe(asset)

    def _build_default_adapter(self) -> TranscriptionProviderAdapter:
        if self.settings.provider == "openai":
            return OpenAITranscriptionAdapter(self.settings)
        raise ExternalTranscriptionProviderError(
            f"Unsupported transcription provider '{self.settings.provider}' in non-sample mode."
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
