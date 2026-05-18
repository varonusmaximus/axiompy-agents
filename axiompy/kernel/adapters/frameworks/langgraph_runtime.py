"""LangGraph runtime adapter (optional extra)."""

from __future__ import annotations

from axiompy.kernel.domain.models import AgentResult, AgentRunConfig
from axiompy.kernel.domain.ports import AgentRuntime
from axiompy.kernel.settings import KernelSettings


class LangGraphRuntime(AgentRuntime):
    """Delegates to LangGraph when installed."""

    def __init__(self, settings: KernelSettings) -> None:
        self._settings = settings
        try:
            import langgraph  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "langgraph is required. Install with: pip install axiompy-agents[kernel-langgraph]"
            ) from exc

    def run(self, goal: str, config: AgentRunConfig) -> AgentResult:
        from axiompy.kernel.runtime.native import NativeAgentRuntime

        # MVP: fall back to native until full graph wiring lands
        native = NativeAgentRuntime(
            llm=self._settings.llm,
            tools=self._settings.tools,  # type: ignore[arg-type]
            memory=self._settings.memory,
            events=self._settings.events,
            checkpoints=self._settings.checkpoints,
        )
        return native.run(goal, config)
