from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class UploadedAsset:
    filename: str
    content_type: str
    size_bytes: int
    raw_bytes: bytes = b""


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
    persona_matches: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PipelineResult:
    asset: UploadedAsset
    transcript: Transcript
    analysis: AnalysisResult
    persisted: bool = False
