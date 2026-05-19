"""LangChain runtime adapter (optional extra)."""

from __future__ import annotations

from axiompy.kernel.domain.models import AgentResult, AgentRunConfig
from axiompy.kernel.domain.ports import AgentRuntime
from axiompy.kernel.settings import KernelSettings
from axiompy.loggers import LoggerFactory

logger = LoggerFactory.create_logger(__name__)

_FALLBACK_WARNED = False


class LangChainRuntime(AgentRuntime):
    """Delegates to LangChain when installed."""

    def __init__(self, settings: KernelSettings) -> None:
        self._settings = settings
        try:
            import langchain_core  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "langchain-core is required. "
                "Install with: pip install axiompy-agents[kernel-langchain]"
            ) from exc

    def run(self, goal: str, config: AgentRunConfig) -> AgentResult:
        global _FALLBACK_WARNED
        if not _FALLBACK_WARNED:
            logger.warning(
                "LangChainRuntime is not fully wired; using NativeAgentRuntime (MVP fallback)"
            )
            _FALLBACK_WARNED = True
        from axiompy.kernel.runtime.native import NativeAgentRuntime

        native = NativeAgentRuntime(
            llm=self._settings.llm,
            tools=self._settings.tools,  # type: ignore[arg-type]
            memory=self._settings.memory,
            events=self._settings.events,
            checkpoints=self._settings.checkpoints,
        )
        return native.run(goal, config)
