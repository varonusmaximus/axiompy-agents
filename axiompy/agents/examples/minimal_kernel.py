"""Minimal kernel example (requires Ollama for live run)."""

from axiompy.kernel import KernelFactory, KernelSettings, RuntimeType
from axiompy.reasoning import ReasoningFactory, ReasoningProvider
from axiompy.reasoning.adapter import ReasoningLLMAdapter


def main() -> None:
    llm = ReasoningFactory.create(ReasoningProvider.OLLAMA)
    kernel = KernelFactory.create(
        RuntimeType.NATIVE,
        KernelSettings(llm=ReasoningLLMAdapter(llm)),
    )
    result = kernel.run("Say hello in one sentence.")
    print(result.output)


if __name__ == "__main__":
    main()
