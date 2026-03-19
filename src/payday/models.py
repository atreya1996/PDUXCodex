from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineStage(str, Enum):
    UPLOAD = "upload"
    TRANSCRIPTION = "transcription"
    ANALYSIS = "analysis"
    PERSONA = "persona"
    STORAGE = "storage"


@dataclass(slots=True)
class BatchUploadItem:
    filename: str
    content_type: str
    data: bytes
    file_id: str = field(default_factory=lambda: uuid4().hex)


@dataclass(slots=True)
class UploadedAsset:
    filename: str
    content_type: str
    size_bytes: int
    raw_bytes: bytes = b""
    file_id: str = field(default_factory=lambda: uuid4().hex)


@dataclass(slots=True)
class Transcript:
    text: str
    provider: str
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AnalysisResult:
    summary: str
    metrics: dict[str, Any] = field(default_factory=dict)
    structured_output: dict[str, Any] = field(default_factory=dict)
    evidence_quotes: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PersonaClassification:
    persona_id: str
    persona_name: str
    rationale: str
    evidence_quotes: tuple[str, ...] = ()
    explanation_payload: dict[str, Any] = field(default_factory=dict)
    is_non_target: bool = False


@dataclass(slots=True)
class PipelineResult:
    file_id: str
    filename: str
    status: ProcessingStatus = ProcessingStatus.PENDING
    current_stage: PipelineStage = PipelineStage.UPLOAD
    last_error: str | None = None
    asset: UploadedAsset | None = None
    transcript: Transcript | None = None
    analysis: AnalysisResult | None = None
    persona: PersonaClassification | None = None
    persisted: bool = False
    attempts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BatchPipelineResult:
    batch_id: str
    results: list[PipelineResult]

    @property
    def completed_count(self) -> int:
        return sum(1 for item in self.results if item.status is ProcessingStatus.COMPLETED)

    @property
    def failed_count(self) -> int:
        return sum(1 for item in self.results if item.status is ProcessingStatus.FAILED)
