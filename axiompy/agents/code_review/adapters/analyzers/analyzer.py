"""Analyzer - Wraps axiompy.reasoning for code review.

Provides a simple analyze(prompt) interface on top of ReasoningFactory,
with added streaming support for Ollama to handle large prompts.
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol

from axiompy.agents.code_review.defaults import (
    DEFAULT_FIRST_TOKEN_TIMEOUT,
    DEFAULT_IDLE_TIMEOUT_SECS,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_STREAM_TIMEOUT_SECS,
    DEFAULT_TIMEOUT_SECS,
)
from axiompy.loggers import LoggerFactory
from axiompy.reasoning import ReasoningFactory, ReasoningProvider

logger = LoggerFactory.create_logger(__name__)


class AnalyzerType(Enum):
    """Supported analyzer types (maps to ReasoningProvider)."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MOCK = "mock"


class Analyzer(Protocol):
    """Protocol for code analyzers."""

    def analyze(self, prompt: str) -> str:
        """Analyze code with the given prompt."""
        ...


@dataclass
class AnalyzerSettings:
    """
    Settings for analyzer creation.

    Attributes:
        model: Model name (provider-specific defaults if not set)
        api_key: API key (required for cloud providers)
        host: Ollama host URL (only for OLLAMA type)
        timeout_secs: Request timeout
        stream: Use streaming for Ollama (avoids timeout on large prompts)
    """

    model: str = DEFAULT_MODEL
    api_key: Optional[str] = None
    host: str = DEFAULT_OLLAMA_HOST
    timeout_secs: int = DEFAULT_TIMEOUT_SECS
    stream: bool = True  # Streaming for Ollama by default
    show_progress: bool = False  # Show tqdm progress bar (for CLI)


class ReasoningAnalyzer:
    """
    Analyzer that wraps axiompy.reasoning.AIClient.

    Provides a simple analyze(prompt) interface for code review.
    """

    def __init__(self, client, provider_type: AnalyzerType):
        """
        Initialize with an AIClient from ReasoningFactory.

        Args:
            client: AIClient from ReasoningFactory
            provider_type: The provider type for logging
        """
        self._client = client
        self._provider_type = provider_type
        logger.info(f"ReasoningAnalyzer initialized: {provider_type.value}")

    def analyze(self, prompt: str) -> str:
        """
        Analyze code with the given prompt.

        Args:
            prompt: Full prompt including code and rules

        Returns:
            AI response as string
        """
        logger.debug(f"Analyzing prompt ({len(prompt)} chars)")

        response = self._client.generate_completion(
            prompt=prompt,
            temperature=0.1,  # Low temperature for consistent reviews
            max_tokens=4096,
            use_cache=False,  # Don't cache code review responses
        )

        logger.debug(f"Got response ({len(response)} chars)")
        return response


class OllamaStreamingAnalyzer:
    """
    Ollama analyzer with streaming support.

    Streaming prevents timeout on large prompts by receiving tokens
    as they're generated instead of waiting for the full response.
    """

    def __init__(
        self,
        host: str = DEFAULT_OLLAMA_HOST,
        model: str = DEFAULT_MODEL,
        timeout_secs: int = DEFAULT_TIMEOUT_SECS,
        show_progress: bool = False,
    ):
        """
        Initialize Ollama streaming analyzer.

        Args:
            host: Ollama server URL
            model: Model name
            timeout_secs: Connection timeout (streaming has separate read timeout)
        """
        self._host = host.rstrip("/")
        self._model = model
        self._timeout_secs = timeout_secs
        self._first_token_timeout = DEFAULT_FIRST_TOKEN_TIMEOUT
        self._stream_timeout_secs = DEFAULT_STREAM_TIMEOUT_SECS
        self._idle_timeout_secs = DEFAULT_IDLE_TIMEOUT_SECS
        self._show_progress = show_progress

        logger.info(f"OllamaStreamingAnalyzer initialized: {host}, model={model}")

        # Verify Ollama is responsive
        self._check_health()

    def _check_health(self) -> None:
        """Verify Ollama is responsive before attempting analysis."""
        import requests

        try:
            response = requests.get(
                f"{self._host}/api/tags",
                timeout=5,
            )
            response.raise_for_status()

            # Check if our model is available
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            if self._model not in models:
                available = ", ".join(models[:5])
                logger.warning(
                    f"Model '{self._model}' not found. Available: {available}. "
                    f"Run: ollama pull {self._model}"
                )
            else:
                logger.debug(f"Ollama healthy, model '{self._model}' available")

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self._host}. Is Ollama running? Try: ollama serve"
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Ollama at {self._host} is not responding. "
                "Try restarting: pkill ollama && ollama serve"
            )

    def analyze(self, prompt: str) -> str:
        """
        Analyze using streaming to avoid timeout.

        Args:
            prompt: Full prompt including code and rules

        Returns:
            AI response as string

        Raises:
            RuntimeError: If Ollama doesn't respond within timeout
        """
        import time

        import requests

        logger.debug(f"Analyzing prompt ({len(prompt)} chars, streaming)")

        url = f"{self._host}/api/generate"
        start_time = time.time()

        try:
            response = requests.post(
                url,
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": True,
                },
                headers={"Content-Type": "application/json"},
                stream=True,
                timeout=(10, self._first_token_timeout),  # Fail fast on first token
            )
            response.raise_for_status()
        except requests.exceptions.ReadTimeout:
            raise RuntimeError(
                f"Ollama didn't respond within {self._first_token_timeout}s. "
                f"Model '{self._model}' may be loading or stuck. "
                "Try: pkill ollama && ollama serve"
            )

        # Progress tracking
        last_progress_time = start_time

        # Collect streamed tokens with idle detection
        full_response = []
        first_token_received = False
        last_token_time = time.time()
        token_count = 0

        try:
            for line in response.iter_lines():
                now = time.time()
                elapsed = now - start_time
                idle_time = now - last_token_time

                # Check overall timeout
                if elapsed > self._stream_timeout_secs:
                    logger.warning(
                        f"Stream timeout after {elapsed:.0f}s, returning partial response"
                    )
                    break

                # Check idle timeout (no tokens received)
                if first_token_received and idle_time > self._idle_timeout_secs:
                    logger.warning(f"No tokens for {idle_time:.0f}s, Ollama may be stuck")
                    break

                if line:
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")

                        if token:
                            token_count += 1
                            last_token_time = now

                            if not first_token_received:
                                first_token_received = True
                                if self._show_progress:
                                    print(f"🤖 {self._model}: streaming...", end="", flush=True)
                                logger.debug(f"First token after {elapsed:.1f}s")

                            # Show progress every 5 seconds
                            if self._show_progress and (now - last_progress_time) >= 5:
                                tokens_per_sec = token_count / elapsed if elapsed > 0 else 0
                                msg = f"\r🤖 {self._model}: {token_count} tokens "
                                msg += f"[{elapsed:.0f}s, {tokens_per_sec:.1f} tok/s]"
                                print(msg, end="", flush=True)
                                last_progress_time = now
                            elif token_count % 100 == 0:
                                logger.debug(f"Streaming... {token_count} tokens, {elapsed:.0f}s")

                        full_response.append(token)

                        if chunk.get("done", False):
                            logger.debug(f"Complete: {token_count} tokens in {elapsed:.1f}s")
                            break
                    except json.JSONDecodeError:
                        continue
        finally:
            if self._show_progress and first_token_received:
                print()  # Newline after progress

        elapsed = time.time() - start_time
        response_text = "".join(full_response)
        logger.debug(f"Got response ({len(response_text)} chars) in {elapsed:.1f}s")

        if not response_text.strip():
            raise RuntimeError(
                f"Ollama returned empty response for model '{self._model}'. "
                "Model may be corrupted. Try: ollama rm {self._model} && ollama pull {self._model}"
            )

        return response_text


class MockAnalyzer:
    """Mock analyzer for testing."""

    DEFAULT_RESPONSE = """
## Summary
No significant issues found.

## Score
85

## Violations
No violations found.
"""

    def __init__(self, response: Optional[str] = None):
        self._response = response or self.DEFAULT_RESPONSE
        self.calls: list = []

    def set_response(self, response: str) -> "MockAnalyzer":
        """Set the response to return."""
        self._response = response
        return self

    def analyze(self, prompt: str) -> str:
        """Return mock response and record the call."""
        self.calls.append(("analyze", prompt))
        return self._response


class AnalyzerFactory:
    """
    Factory for creating analyzers.

    Uses axiompy.reasoning.ReasoningFactory under the hood,
    with added streaming support for Ollama.

    Example:
        # Ollama with streaming (default)
        analyzer = AnalyzerFactory.create(AnalyzerType.OLLAMA)

        # OpenAI
        analyzer = AnalyzerFactory.create(
            AnalyzerType.OPENAI,
            AnalyzerSettings(api_key="sk-..."),
        )

        # Mock for testing
        analyzer = AnalyzerFactory.create_mock()
    """

    @staticmethod
    def create(
        analyzer_type: AnalyzerType,
        settings: Optional[AnalyzerSettings] = None,
    ) -> Analyzer:
        """
        Create an analyzer instance.

        Args:
            analyzer_type: Type of analyzer to create
            settings: Configuration settings

        Returns:
            Configured analyzer instance
        """
        if settings is None:
            settings = AnalyzerSettings()

        match analyzer_type:
            case AnalyzerType.OLLAMA:
                # Use streaming analyzer for Ollama to avoid timeout
                if settings.stream:
                    return OllamaStreamingAnalyzer(
                        host=settings.host,
                        model=settings.model,
                        timeout_secs=settings.timeout_secs,
                        show_progress=settings.show_progress,
                    )
                else:
                    # Non-streaming uses ReasoningFactory
                    client = ReasoningFactory.create(
                        ReasoningProvider.OLLAMA,
                        model=settings.model,
                        endpoint=f"{settings.host}/api/generate",
                    )
                    return ReasoningAnalyzer(client, analyzer_type)

            case AnalyzerType.OPENAI:
                if not settings.api_key:
                    raise ValueError("api_key required for OpenAI")

                client = ReasoningFactory.create(
                    ReasoningProvider.OPENAI,
                    model=settings.model or "gpt-4o",
                    api_key=settings.api_key,
                )
                return ReasoningAnalyzer(client, analyzer_type)

            case AnalyzerType.ANTHROPIC:
                if not settings.api_key:
                    raise ValueError("api_key required for Anthropic")

                client = ReasoningFactory.create(
                    ReasoningProvider.ANTHROPIC,
                    model=settings.model or "claude-sonnet-4-20250514",
                    api_key=settings.api_key,
                )
                return ReasoningAnalyzer(client, analyzer_type)

            case AnalyzerType.MOCK:
                return MockAnalyzer()

            case _:
                raise ValueError(f"Unknown analyzer type: {analyzer_type}")

    @staticmethod
    def create_mock(response: Optional[str] = None) -> MockAnalyzer:
        """Create a mock analyzer for testing."""
        return MockAnalyzer(response=response)
