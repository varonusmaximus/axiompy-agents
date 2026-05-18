"""Kernel ports (hexagonal interfaces)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from axiompy.kernel.domain.models import (
    AgentEvent,
    AgentResult,
    AgentRunConfig,
    CheckpointState,
    LLMCompletion,
    Message,
    ToolCall,
    ToolSpec,
)


@runtime_checkable
class LLMPort(Protocol):
    """Language model completion port."""

    def complete(
        self,
        messages: List[Message],
        tools: List[ToolSpec],
    ) -> LLMCompletion: ...

    def stream(
        self,
        messages: List[Message],
        tools: List[ToolSpec],
    ):
        """Yield text deltas; optional for adapters that support streaming."""
        ...


@runtime_checkable
class ToolHandler(Protocol):
    """Callable tool implementation."""

    def __call__(self, **kwargs: Any) -> Any: ...


@runtime_checkable
class ToolRegistry(Protocol):
    """Register and invoke tools."""

    def list_tools(self) -> List[ToolSpec]: ...

    def invoke(self, tool_call: ToolCall) -> str: ...


@runtime_checkable
class MemoryStore(Protocol):
    """Session and long-term memory."""

    def get(self, key: str, session_id: Optional[str] = None) -> Optional[str]: ...

    def put(self, key: str, value: str, session_id: Optional[str] = None) -> None: ...

    def append_message(self, session_id: str, message: Message) -> None: ...

    def get_messages(self, session_id: str) -> List[Message]: ...


@runtime_checkable
class CheckpointStore(Protocol):
    """Persist run state."""

    def save(self, state: CheckpointState) -> None: ...

    def load(self, run_id: str) -> Optional[CheckpointState]: ...


@runtime_checkable
class EventPublisher(Protocol):
    """Publish streaming agent events."""

    def publish(self, event: AgentEvent) -> None: ...


@runtime_checkable
class AgentCoordinator(Protocol):
    """Coordinate multi-agent handoffs."""

    def delegate(
        self,
        agent_id: str,
        goal: str,
        context: Dict[str, Any],
    ) -> AgentResult: ...


@runtime_checkable
class AgentRuntime(Protocol):
    """Execute an agent loop."""

    def run(
        self,
        goal: str,
        config: AgentRunConfig,
    ) -> AgentResult: ...
