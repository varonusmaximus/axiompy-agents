"""Tests for axiompy.kernel."""

import sys
import types

import pytest

from axiompy.kernel import KernelFactory, KernelSettings, RuntimeType
from axiompy.kernel.adapters.llm.mock import MockLLMPort
from axiompy.kernel.adapters.tools.simple_registry import SimpleToolRegistry
from axiompy.kernel.domain.models import AgentRunConfig, ToolCall, ToolSpec
from axiompy.kernel.adapters.events.in_memory import InMemoryEventPublisher
from axiompy.kernel.adapters.memory.in_memory import InMemoryMemoryStore
from axiompy.kernel.adapters.checkpoints.in_memory import InMemoryCheckpointStore


class TestKernelFactory:
    def test_create_mock(self) -> None:
        kernel = KernelFactory.create_mock(["Hello from agent."])
        result = kernel.run("Say hi")
        assert "Hello" in result.output
        assert result.steps >= 1

    def test_tool_loop(self) -> None:
        registry = SimpleToolRegistry()
        registry.register(
            ToolSpec(name="add", description="Add numbers", parameters={}),
            lambda a, b: a + b,
        )
        llm = MockLLMPort(
            responses=["", "The sum is 3."],
            tool_calls=[
                [ToolCall(id="1", name="add", arguments={"a": 1, "b": 2})],
            ],
        )
        kernel = KernelFactory.create(
            RuntimeType.NATIVE,
            KernelSettings(llm=llm, tools=registry),
        )
        result = kernel.run("What is 1+2?")
        assert result.steps == 2
        assert "3" in result.output

    def test_memory_and_events(self) -> None:
        events = InMemoryEventPublisher()
        memory = InMemoryMemoryStore()
        llm = MockLLMPort(["Done."])
        kernel = KernelFactory.create(
            RuntimeType.NATIVE,
            KernelSettings(
                llm=llm,
                memory=memory,
                events=events,
            ),
        )
        config = AgentRunConfig(session_id="sess-1", max_steps=5)
        kernel.run("test", config)
        assert any(e.event_type == "run_started" for e in events.events)
        assert len(memory.get_messages("sess-1")) >= 1

    def test_checkpoint_store(self) -> None:
        checkpoints = InMemoryCheckpointStore()
        llm = MockLLMPort(["ok"])
        kernel = KernelFactory.create(
            RuntimeType.NATIVE,
            KernelSettings(llm=llm, checkpoints=checkpoints),
        )
        config = AgentRunConfig(session_id="run-abc", max_steps=3)
        kernel.run("go", config)
        state = checkpoints.load("run-abc")
        assert state is not None
        assert state.step >= 1

    def test_langgraph_import_error(self) -> None:
        with pytest.raises(ImportError, match="langgraph"):
            KernelFactory.create(
                RuntimeType.LANGGRAPH,
                KernelSettings(llm=MockLLMPort()),
            )

    def test_framework_fallback_warn_flag_is_per_instance(self, monkeypatch) -> None:
        """Each framework runtime tracks its own fallback warning state."""
        fake_langgraph = types.ModuleType("langgraph")
        fake_langchain = types.ModuleType("langchain_core")
        monkeypatch.setitem(sys.modules, "langgraph", fake_langgraph)
        monkeypatch.setitem(sys.modules, "langchain_core", fake_langchain)

        from axiompy.kernel.adapters.frameworks.langchain_runtime import LangChainRuntime
        from axiompy.kernel.adapters.frameworks.langgraph_runtime import LangGraphRuntime

        settings = KernelSettings(llm=MockLLMPort())
        graph_a = LangGraphRuntime(settings)
        graph_b = LangGraphRuntime(settings)
        chain_a = LangChainRuntime(settings)

        assert graph_a._fallback_warned is False
        assert graph_b._fallback_warned is False
        assert chain_a._fallback_warned is False

        config = AgentRunConfig(max_steps=1)
        graph_a.run("goal", config)
        assert graph_a._fallback_warned is True
        assert graph_b._fallback_warned is False
        graph_b.run("goal", config)
        assert graph_b._fallback_warned is True

    def test_sequential_coordinator(self) -> None:
        from axiompy.kernel.adapters.coordinator.sequential import SequentialCoordinator
        from axiompy.kernel.runtime.native import NativeAgentRuntime

        runtime = NativeAgentRuntime(
            llm=MockLLMPort(["delegated result"]),
            tools=SimpleToolRegistry(),
        )
        coord = SequentialCoordinator().register("worker", runtime)
        result = coord.delegate("worker", "task", {})
        assert "delegated" in result.output
