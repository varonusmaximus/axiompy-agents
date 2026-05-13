"""Adapters Layer - External adapter implementations.

This layer contains implementations of the ports defined in the domain layer:

Sources (CodeSource):
- FileSystemSource: Read from local filesystem
- GitHubSource: Fetch from GitHub API
- GitSource: Read from local git repository

Rules (RulesSource):
- FileRulesSource: Read AGENTS.md from filesystem
- GitHubRulesSource: Fetch AGENTS.md from GitHub

Analyzers (AIAnalyzer):
- Uses axiompy.reasoning under the hood
- AnalyzerFactory: Create analyzers for Ollama, OpenAI, Anthropic
- OllamaStreamingAnalyzer: Streaming for large prompts

Publishers (ReviewPublisher):
- ConsolePublisher: Print to terminal
- GitHubPublisher: Post PR review comments
- JSONPublisher: Output JSON for CI
"""

# Sources
# Analyzers
from .analyzers import AnalyzerFactory, AnalyzerSettings, AnalyzerType, MockAnalyzer

# Publishers
from .publishers import ConsolePublisher, GitHubPublisher, JSONPublisher

# Rules
from .rules import FileRulesSource, GitHubRulesSource, MockRulesSource
from .sources import FileSystemSource, GitHubSource, GitSource, MockCodeSource

__all__ = [
    # Sources
    "FileSystemSource",
    "GitHubSource",
    "GitSource",
    "MockCodeSource",
    # Rules
    "FileRulesSource",
    "GitHubRulesSource",
    "MockRulesSource",
    # Analyzers
    "AnalyzerFactory",
    "AnalyzerType",
    "AnalyzerSettings",
    "MockAnalyzer",
    # Publishers
    "ConsolePublisher",
    "GitHubPublisher",
    "JSONPublisher",
]
