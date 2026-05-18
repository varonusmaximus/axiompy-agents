"""Reasoning client settings."""

from dataclasses import dataclass
from typing import Optional

from axiompy.validators import ensure_positive


@dataclass
class ReasoningSettings:
    """Configuration for ReasoningFactory.create."""

    model: Optional[str] = None
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    cache_size: int = 128

    def __post_init__(self) -> None:
        ensure_positive(self.cache_size, "cache_size must be positive")
