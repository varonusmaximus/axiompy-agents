"""Kernel factory."""

from __future__ import annotations

from axiompy.kernel.domain.service import KernelService
from axiompy.kernel.runtime.native import NativeAgentRuntime
from axiompy.kernel.settings import KernelSettings
from axiompy.kernel.types import RuntimeType


class KernelFactory:
    """Create KernelService instances with enum-based runtime selection."""

    @staticmethod
    def create(
        runtime_type: RuntimeType,
        settings: KernelSettings,
    ) -> KernelService:
        match runtime_type:
            case RuntimeType.NATIVE:
                runtime = NativeAgentRuntime(
                    llm=settings.llm,
                    tools=settings.tools,  # type: ignore[arg-type]
                    memory=settings.memory,
                    events=settings.events,
                    checkpoints=settings.checkpoints,
                )
                return KernelService(runtime)
            case RuntimeType.LANGGRAPH:
                from axiompy.kernel.adapters.frameworks.langgraph_runtime import (
                    LangGraphRuntime,
                )

                return KernelService(LangGraphRuntime(settings))
            case RuntimeType.LANGCHAIN:
                from axiompy.kernel.adapters.frameworks.langchain_runtime import (
                    LangChainRuntime,
                )

                return KernelService(LangChainRuntime(settings))
            case _:
                raise ValueError(f"Unknown runtime type: {runtime_type}")

    @staticmethod
    def create_mock(
        responses: list[str] | None = None,
    ) -> KernelService:
        from axiompy.kernel.adapters.llm.mock import MockLLMPort

        return KernelFactory.create(
            RuntimeType.NATIVE,
            KernelSettings(llm=MockLLMPort(responses or ["Done."])),
        )
