from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, Protocol
from urllib import error, request

from payday.config import TranscriptionSettings
from payday.models import Transcript, UploadedAsset

OPENAI_TRANSCRIPTION_DOCUMENTED_LIMIT_BYTES = 25_000_000
OPENAI_TRANSCRIPTION_SAFE_FILE_LIMIT_BYTES = 24_000_000
_NON_TEXT_TRANSCRIPT_RATIO_THRESHOLD = 0.3
_BINARY_SIGNATURE_PATTERNS = (
    "ftyp",
    "id3",
    "lame",
    "moov",
    "mdat",
    "riff",
)
_ESCAPED_BINARY_HEX_PATTERN = re.compile(r"(?:\\x[0-9a-fA-F]{2}){4,}")


class ExternalTranscriptionProviderError(RuntimeError):
    """Raised when an external transcription provider cannot transcribe audio."""


class TranscriptionProviderError(ValueError):
    """Raised when an unsupported transcription provider is configured."""


class SampleModeDisabledError(RuntimeError):
    """Raised when sample transcription is requested while sample mode is disabled."""


class TranscriptionProviderAdapter(Protocol):
    provider_name: str
    model_name: str

    def transcribe(self, asset: UploadedAsset) -> Transcript:
        """Return a transcript produced by an external provider."""


def detect_malformed_transcript_reason(transcript_text: str) -> str | None:
    normalized = transcript_text.lower()
    if any(signature in normalized for signature in _BINARY_SIGNATURE_PATTERNS):
        return "binary signature detected in transcript output"
    if "\x00" * 8 in transcript_text or "\\x00" * 8 in transcript_text:
        return "null-byte binary sequence detected in transcript output"
    if _ESCAPED_BINARY_HEX_PATTERN.search(transcript_text):
        return "escaped binary hex sequence detected in transcript output"

    if not transcript_text:
        return "empty transcript output"
    suspicious_char_count = sum(
        1
        for char in transcript_text
        if (not char.isprintable() and not char.isspace()) or char == "\ufffd"
    )
    non_text_ratio = suspicious_char_count / len(transcript_text)
    if non_text_ratio > _NON_TEXT_TRANSCRIPT_RATIO_THRESHOLD:
        return (
            "excessive non-text character ratio "
            f"({non_text_ratio:.0%}, threshold {_NON_TEXT_TRANSCRIPT_RATIO_THRESHOLD:.0%})"
        )
    return None


def get_transcription_file_size_limit_bytes(settings: TranscriptionSettings) -> int | None:
    provider = settings.provider.strip().lower()
    if provider == "openai":
        return OPENAI_TRANSCRIPTION_SAFE_FILE_LIMIT_BYTES
    return None


def describe_transcription_file_size_limit(settings: TranscriptionSettings) -> str | None:
    limit_bytes = get_transcription_file_size_limit_bytes(settings)
    if limit_bytes is None:
        return None
    safe_limit_mb = limit_bytes / 1_000_000
    documented_limit_mb = OPENAI_TRANSCRIPTION_DOCUMENTED_LIMIT_BYTES / 1_000_000
    return (
        f"OpenAI transcriptions cap requests at {documented_limit_mb:.0f} MB, so PayDay preflights "
        f"uploads at {safe_limit_mb:.0f} MB to leave room for multipart overhead. Compress or split "
        "larger recordings before processing."
    )


def validate_transcription_asset_size(settings: TranscriptionSettings, asset: UploadedAsset) -> str | None:
    limit_bytes = get_transcription_file_size_limit_bytes(settings)
    if limit_bytes is None or asset.size_bytes <= limit_bytes:
        return None

    safe_limit_mb = limit_bytes / 1_000_000
    actual_size_mb = asset.size_bytes / 1_000_000
    documented_limit_mb = OPENAI_TRANSCRIPTION_DOCUMENTED_LIMIT_BYTES / 1_000_000
    return (
        f"{asset.filename} is {actual_size_mb:.1f} MB, which exceeds PayDay's {safe_limit_mb:.0f} MB "
        f"OpenAI transcription safety limit. OpenAI caps transcription requests at {documented_limit_mb:.0f} MB "
        "including multipart upload overhead, so please compress or split this recording and upload it again."
    )


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
            metadata={
                "sample_mode": False,
                "filename": asset.filename,
                "file_id": asset.file_id,
                "content_type": asset.content_type,
                "size_bytes": asset.size_bytes,
                "source": asset.filename,
            },
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
            f'Content-Disposition: form-data; name="file"; filename="{asset.filename}"'.encode("utf-8"),
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
    """Provider facade for audio/video transcription.

    Speaker-role labeling intentionally happens in structured analysis rather than
    in the raw transcription payload so the pipeline can keep provider output
    evidence-first, preserve plain transcript text, and apply weak filename or
    interview metadata hints without fabricating unsupported tags.
    """

    def __init__(
        self,
        settings: TranscriptionSettings,
        *,
        adapter: TranscriptionProviderAdapter | None = None,
        client: Any | None = None,
    ) -> None:
        self.settings = settings
        self.adapter = adapter
        self._client = client

    def transcribe(self, asset: UploadedAsset, sample_mode: bool = False) -> Transcript:
        if sample_mode:
            if not _is_sample_mode_enabled():
                raise SampleModeDisabledError(
                    "Sample-mode transcription is disabled. Set PAYDAY_USE_SAMPLE_MODE=true to enable sample transcript decoding."
                )
            return self._sample_transcript(asset)
        return self._provider_transcript(asset)

    def validate_asset(self, asset: UploadedAsset) -> str | None:
        return validate_transcription_asset_size(self.settings, asset)

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
        raise TranscriptionProviderError(f"Unsupported transcription provider: {self.settings.provider}")

    def _transcribe_with_openai(self, asset: UploadedAsset) -> Transcript:
        if not self.settings.api_key.strip():
            raise ValueError(
                f"TRANSCRIPTION_API_KEY is required when sample mode is disabled for provider '{self.settings.provider}'."
            )

        if self.adapter is not None:
            return self.adapter.transcribe(asset)

        if self._client is not None:
            client = self._client
        else:  # pragma: no cover - exercised in live/manual validation
            from openai import OpenAI

            client = OpenAI(
                api_key=self.settings.api_key,
                max_retries=0,
                timeout=self.settings.timeout_seconds,
            )

        try:
            response = client.audio.transcriptions.create(
                model=self.settings.model,
                file=(asset.filename, asset.raw_bytes, asset.content_type),
            )
        except Exception as exc:  # pragma: no cover - exact types depend on provider/runtime
            message = str(exc) or exc.__class__.__name__
            if "timed out" in message.lower() or "timeout" in exc.__class__.__name__.lower():
                raise TimeoutError(message) from exc
            raise ExternalTranscriptionProviderError(message) from exc

        text = getattr(response, "text", "")
        if not isinstance(text, str) or not text.strip():
            raise ExternalTranscriptionProviderError("OpenAI transcription provider returned an empty transcript.")

        metadata = {
            "sample_mode": False,
            "filename": asset.filename,
            "file_id": asset.file_id,
            "content_type": asset.content_type,
            "size_bytes": asset.size_bytes,
            "source": asset.filename,
        }
        language = getattr(response, "language", None)
        if language:
            metadata["language"] = language
        duration = getattr(response, "duration", None)
        if duration is not None:
            metadata["duration_seconds"] = duration

        return Transcript(
            text=text,
            provider=self.settings.provider,
            model=self.settings.model,
            metadata=metadata,
        )

    def _build_transcript(self, *, asset: UploadedAsset, sample_mode: bool, text: str) -> Transcript:
        return Transcript(
            text=text,
            provider=self.settings.provider,
            model=self.settings.model,
            metadata={
                "sample_mode": sample_mode,
                "filename": asset.filename,
                "file_id": asset.file_id,
                "content_type": asset.content_type,
                "size_bytes": asset.size_bytes,
                "source": asset.filename,
            },
        )

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

    if sample_mode and _is_sample_mode_enabled():
        return TranscriptionService(settings)
    if settings.provider == "openai":
        return OpenAITranscriptionService(settings)
    raise TranscriptionProviderError(
        f"Unsupported transcription provider '{settings.provider}'. Use TRANSCRIPTION_PROVIDER=openai."
    )


def _is_sample_mode_enabled() -> bool:
    return os.getenv("PAYDAY_USE_SAMPLE_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
