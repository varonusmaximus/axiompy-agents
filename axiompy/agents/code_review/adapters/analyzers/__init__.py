"""AI Analyzer - Wraps axiompy.reasoning for code review.

Uses ReasoningFactory from axiompy.reasoning with added streaming support
for Ollama to handle large prompts without timeout.

Example:
    from axiompy.agents.code_review.adapters.analyzers import (
        AnalyzerFactory,
        AnalyzerType,
    )

    # Ollama (local, no API key)
    analyzer = AnalyzerFactory.create(AnalyzerType.OLLAMA)
    response = analyzer.analyze(prompt)

    # OpenAI
    analyzer = AnalyzerFactory.create(AnalyzerType.OPENAI, api_key="sk-...")

    # Mock for testing
    analyzer = AnalyzerFactory.create_mock()
"""

from .analyzer import (
    Analyzer,
    AnalyzerFactory,
    AnalyzerSettings,
    AnalyzerType,
    MockAnalyzer,
)

__all__ = [
    "AnalyzerFactory",
    "AnalyzerType",
    "AnalyzerSettings",
    "Analyzer",
    "MockAnalyzer",
]
