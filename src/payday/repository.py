from __future__ import annotations

from payday.models import PipelineResult


class PaydayRepository:
    """Persistence facade for analysis artifacts and dashboard reads."""

    def __init__(self) -> None:
        self._items: dict[str, PipelineResult] = {}

    def save_result(self, result: PipelineResult) -> PipelineResult:
        self._items[result.file_id] = result
        return result

    def get_result(self, file_id: str) -> PipelineResult | None:
        return self._items.get(file_id)

    def list_results(self) -> list[PipelineResult]:
        return list(self._items.values())
