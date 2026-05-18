"""Native plan-act-observe agent runtime."""

from __future__ import annotations

import uuid
from typing import Optional

from axiompy.kernel.domain.errors import KernelRuntimeError
from axiompy.kernel.domain.models import (
    AgentEvent,
    AgentResult,
    AgentRunConfig,
    CheckpointState,
    Message,
)
from axiompy.kernel.domain.ports import (
    AgentRuntime,
    CheckpointStore,
    EventPublisher,
    LLMPort,
    MemoryStore,
    ToolRegistry,
)
from axiompy.loggers import LoggerFactory

logger = LoggerFactory.create_logger(__name__)


class NativeAgentRuntime(AgentRuntime):
    """Default agent loop: LLM plan → tool act → observe until done."""

    def __init__(
        self,
        llm: LLMPort,
        tools: ToolRegistry,
        memory: Optional[MemoryStore] = None,
        events: Optional[EventPublisher] = None,
        checkpoints: Optional[CheckpointStore] = None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._memory = memory
        self._events = events
        self._checkpoints = checkpoints

    def run(self, goal: str, config: AgentRunConfig) -> AgentResult:
        run_id = config.session_id or str(uuid.uuid4())
        messages: list[Message] = []

        if config.system_prompt:
            messages.append(Message(role="system", content=config.system_prompt))

        if self._memory and config.session_id:
            messages.extend(self._memory.get_messages(config.session_id))

        messages.append(Message(role="user", content=goal))
        tool_specs = self._tools.list_tools()
        steps = 0

        self._emit("run_started", {"run_id": run_id, "goal": goal})

        for step in range(config.max_steps):
            steps = step + 1
            self._emit("step_started", {"run_id": run_id, "step": steps})

            completion = self._llm.complete(messages, tool_specs)
            assistant = Message(
                role="assistant",
                content=completion.content,
                tool_calls=list(completion.tool_calls),
            )
            messages.append(assistant)

            if self._memory and config.session_id:
                self._memory.append_message(config.session_id, assistant)

            if not completion.tool_calls:
                self._emit("run_completed", {"run_id": run_id, "steps": steps})
                self._maybe_checkpoint(run_id, messages, steps)
                return AgentResult(output=completion.content, messages=messages, steps=steps)

            for tool_call in completion.tool_calls:
                self._emit(
                    "tool_call",
                    {"run_id": run_id, "tool": tool_call.name, "args": tool_call.arguments},
                )
                observation = self._tools.invoke(tool_call)
                tool_message = Message(
                    role="tool",
                    content=observation,
                    tool_call_id=tool_call.id,
                )
                messages.append(tool_message)
                if self._memory and config.session_id:
                    self._memory.append_message(config.session_id, tool_message)

            self._maybe_checkpoint(run_id, messages, steps)

        raise KernelRuntimeError(f"Max steps ({config.max_steps}) exceeded")

    def _emit(self, event_type: str, payload: dict) -> None:
        if self._events:
            self._events.publish(AgentEvent(event_type=event_type, payload=payload))

    def _maybe_checkpoint(
        self,
        run_id: str,
        messages: list[Message],
        step: int,
    ) -> None:
        if self._checkpoints:
            self._checkpoints.save(
                CheckpointState(run_id=run_id, messages=list(messages), step=step)
            )
