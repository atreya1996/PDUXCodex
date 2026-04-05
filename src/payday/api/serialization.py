from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any


def to_json_compatible(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_json_compatible(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_json_compatible(item) for item in value]
    if isinstance(value, tuple):
        return [to_json_compatible(item) for item in value]
    return value
