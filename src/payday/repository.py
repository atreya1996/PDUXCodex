from __future__ import annotations

from payday.models import PipelineResult


class PaydayRepository:
    """Persistence facade for analysis artifacts and dashboard reads."""

    def __init__(self) -> None:
        self._items: list[PipelineResult] = []

    def save_result(self, result: PipelineResult) -> PipelineResult:
        self._items.append(result)
        return result

    def list_results(self) -> list[PipelineResult]:
        return list(self._items)
