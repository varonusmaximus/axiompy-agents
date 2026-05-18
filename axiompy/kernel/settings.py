"""Kernel configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from axiompy.kernel.adapters.checkpoints.in_memory import InMemoryCheckpointStore
from axiompy.kernel.adapters.events.in_memory import InMemoryEventPublisher
from axiompy.kernel.adapters.memory.in_memory import InMemoryMemoryStore
from axiompy.kernel.adapters.tools.simple_registry import SimpleToolRegistry
from axiompy.kernel.domain.ports import (
    CheckpointStore,
    EventPublisher,
    LLMPort,
    MemoryStore,
    ToolRegistry,
)
from axiompy.kernel.domain.models import ToolSpec
from axiompy.validators import ensure_in_range, ensure_positive


@dataclass
class KernelSettings:
    """Explicit dependencies for kernel creation."""

    llm: LLMPort
    tools: Optional[ToolRegistry] = None
    tool_specs: List[ToolSpec] = field(default_factory=list)
    memory: Optional[MemoryStore] = None
    events: Optional[EventPublisher] = None
    checkpoints: Optional[CheckpointStore] = None
    max_steps: int = 10

    def __post_init__(self) -> None:
        ensure_positive(self.max_steps, "max_steps must be positive")
        ensure_in_range(self.max_steps, 1, 100, "max_steps must be 1-100")
        if self.tools is None:
            self.tools = SimpleToolRegistry()
        if self.memory is None:
            self.memory = InMemoryMemoryStore()
        if self.events is None:
            self.events = InMemoryEventPublisher()
        if self.checkpoints is None:
            self.checkpoints = InMemoryCheckpointStore()
