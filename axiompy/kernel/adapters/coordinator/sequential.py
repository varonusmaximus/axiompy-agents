"""Sequential multi-agent coordinator."""

from __future__ import annotations

from typing import Any, Dict

from axiompy.kernel.domain.models import AgentResult, AgentRunConfig
from axiompy.kernel.domain.ports import AgentCoordinator, AgentRuntime


class SequentialCoordinator(AgentCoordinator):
    """Delegates to registered agent runtimes by id."""

    def __init__(self) -> None:
        self._agents: Dict[str, AgentRuntime] = {}

    def register(self, agent_id: str, runtime: AgentRuntime) -> "SequentialCoordinator":
        self._agents[agent_id] = runtime
        return self

    def delegate(
        self,
        agent_id: str,
        goal: str,
        context: Dict[str, Any],
    ) -> AgentResult:
        if agent_id not in self._agents:
            raise ValueError(f"Unknown agent: {agent_id}")
        config = AgentRunConfig(
            max_steps=context.get("max_steps", 10),
            session_id=context.get("session_id"),
            system_prompt=context.get("system_prompt"),
        )
        return self._agents[agent_id].run(goal, config)
