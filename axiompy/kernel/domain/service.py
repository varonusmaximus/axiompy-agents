"""Kernel service — application entry for agent runs."""

from __future__ import annotations

from axiompy.kernel.domain.models import AgentResult, AgentRunConfig
from axiompy.kernel.domain.ports import AgentRuntime


class KernelService:
    """Orchestrates agent execution via an injected runtime."""

    def __init__(self, runtime: AgentRuntime) -> None:
        self._runtime = runtime

    def run(
        self,
        goal: str,
        config: AgentRunConfig | None = None,
    ) -> AgentResult:
        return self._runtime.run(goal, config or AgentRunConfig())
