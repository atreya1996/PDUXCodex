from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class OpenAIClientService:
    """Thin OpenAI audio transcription client with structured request logs."""

    provider_name = "openai"

    def __init__(self, client: Any) -> None:
        self._client = client

    def transcribe_file(self, file_path: str, model: str) -> Any:
        path = Path(file_path)
        timestamp = datetime.now(timezone.utc).isoformat()
        file_id = path.name.split("-", maxsplit=1)[0]
        log_payload = {
            "file_id": file_id,
            "file_path": str(path),
            "timestamp": timestamp,
            "provider": self.provider_name,
            "model": model,
        }
        logger.info("openai transcription request started: %s", log_payload)
        try:
            with path.open("rb") as audio_file:
                response = self._client.audio.transcriptions.create(model=model, file=audio_file)
        except Exception as exc:  # noqa: BLE001
            failure_payload = {
                **log_payload,
                "exception_text": str(exc) or exc.__class__.__name__,
            }
            logger.exception("openai transcription request failed: %s", failure_payload)
            raise RuntimeError(failure_payload["exception_text"]) from exc

        logger.info("openai transcription request succeeded: %s", log_payload)
        return response
