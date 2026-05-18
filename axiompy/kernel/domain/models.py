"""Kernel domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolSpec:
    """Tool definition exposed to the LLM."""

    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    """A tool invocation requested by the LLM."""

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """Conversation message."""

    role: str
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None


@dataclass
class LLMCompletion:
    """Result of an LLM completion step."""

    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"


@dataclass
class AgentRunConfig:
    """Per-run configuration."""

    max_steps: int = 10
    session_id: Optional[str] = None
    system_prompt: Optional[str] = None


@dataclass
class AgentResult:
    """Final result of an agent run."""

    output: str
    messages: List[Message] = field(default_factory=list)
    steps: int = 0


@dataclass
class AgentEvent:
    """Streaming event published during a run."""

    event_type: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointState:
    """Serializable run state for resume."""

    run_id: str
    messages: List[Message]
    step: int
