"""In-memory event publisher."""

from __future__ import annotations

from typing import List

from axiompy.kernel.domain.models import AgentEvent
from axiompy.kernel.domain.ports import EventPublisher


class InMemoryEventPublisher(EventPublisher):
    """Collects events for tests and debugging."""

    def __init__(self) -> None:
        self.events: List[AgentEvent] = []

    def publish(self, event: AgentEvent) -> None:
        self.events.append(event)

    def clear(self) -> None:
        self.events.clear()
