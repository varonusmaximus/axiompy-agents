"""Tests for AIClient and ReasoningFactory"""

from unittest.mock import MagicMock, patch

import pytest

from axiompy.reasoning.client import AIClient
from axiompy.reasoning.factory import ReasoningFactory
from axiompy.reasoning.types import ReasoningProvider
from axiompy.reasoning.metadata import (
    DatasetMetadata,
    ScopeMetadata,
    TableSchemaMetadata,
)
from axiompy.reasoning.providers.ollama import OllamaProviderConfig
from axiompy.reasoning.providers.openai import OpenAIProviderConfig


class TestAIClientInit:
    """Tests for AIClient initialization"""

    def test_init_with_provider_string(self):
        """Test initializing with provider name"""
        client = AIClient(provider="ollama", model="mistral", endpoint="http://localhost:11434")
        assert client.model == "mistral"
        assert client.endpoint == "http://localhost:11434"
        assert client.provider == OllamaProviderConfig

    def test_init_with_provider_class(self):
        """Test initializing with provider class"""
        client = AIClient(
            provider=OllamaProviderConfig, model="mistral", endpoint="http://localhost:11434"
        )
        assert client.provider == OllamaProviderConfig

    def test_init_with_invalid_provider_string(self):
        """Test initialization with invalid provider name"""
        with pytest.raises(ValueError, match="Unknown provider"):
            AIClient(provider="nonexistent", model="test", endpoint="http://test")

    def test_init_with_invalid_provider_type(self):
        """Test initialization with invalid provider type"""

        class NotAProvider:
            pass

        with pytest.raises(TypeError):
            AIClient(provider=NotAProvider, model="test", endpoint="http://test")

    def test_init_with_api_key(self):
        """Test initialization with API key"""
        client = AIClient(
            provider="openai", model="gpt-4", endpoint="https://api.openai.com", api_key="sk-test"
        )
        assert client.api_key == "sk-test"


class TestAIClientGenerateCompletion:
    """Tests for generate_completion method"""

    @patch("axiompy.reasoning.client.HTTPClientFactory.create")
    def test_generate_completion_ollama(self, mock_factory):
        """Test generating completion with Ollama"""
        # Mock the HTTP client
        mock_http_client = MagicMock()
        mock_factory.return_value = mock_http_client

        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Test response"}
        mock_http_client.post.return_value = mock_response

        client = AIClient(provider="ollama", model="mistral", endpoint="http://localhost:11434")

        result = client.generate_completion("Test prompt", use_cache=False)

        assert result == "Test response"
        mock_http_client.post.assert_called_once()

    @patch("axiompy.reasoning.client.HTTPClientFactory.create")
    def test_generate_completion_openai(self, mock_factory):
        """Test generating completion with OpenAI"""
        mock_http_client = MagicMock()
        mock_factory.return_value = mock_http_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Test response"}}]}
        mock_http_client.post.return_value = mock_response

        client = AIClient(
            provider="openai", model="gpt-4", endpoint="https://api.openai.com", api_key="sk-test"
        )

        result = client.generate_completion("Test prompt", use_cache=False)

        assert result == "Test response"

    @patch("axiompy.reasoning.client.HTTPClientFactory.create")
    def test_generate_completion_with_cache(self, mock_factory):
        """Test that caching works"""
        mock_http_client = MagicMock()
        mock_factory.return_value = mock_http_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Cached response"}
        mock_http_client.post.return_value = mock_response

        client = AIClient(provider="ollama", model="mistral", endpoint="http://localhost:11434")

        # First call should hit the provider
        result1 = client.generate_completion("Same prompt", use_cache=True)
        assert mock_http_client.post.call_count == 1

        # Second call with same parameters should use cache
        result2 = client.generate_completion("Same prompt", use_cache=True)
        assert mock_http_client.post.call_count == 1  # No additional call
        assert result1 == result2

    @patch("axiompy.reasoning.client.HTTPClientFactory.create")
    def test_generate_completion_without_cache(self, mock_factory):
        """Test that caching can be disabled"""
        mock_http_client = MagicMock()
        mock_factory.return_value = mock_http_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Response"}
        mock_http_client.post.return_value = mock_response

        client = AIClient(provider="ollama", model="mistral", endpoint="http://localhost:11434")

        # Calls with use_cache=False should not cache
        client.generate_completion("Prompt", use_cache=False)
        client.generate_completion("Prompt", use_cache=False)
        assert mock_http_client.post.call_count == 2

    @patch("axiompy.reasoning.client.HTTPClientFactory.create")
    def test_generate_completion_api_error(self, mock_factory):
        """Test error handling on API failure"""
        mock_http_client = MagicMock()
        mock_factory.return_value = mock_http_client

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_http_client.post.return_value = mock_response

        client = AIClient(provider="ollama", model="mistral", endpoint="http://localhost:11434")

        with pytest.raises(ConnectionError, match="API error"):
            client.generate_completion("Test", use_cache=False)


class TestAIClientGenerateSQLFromQuestion:
    """Tests for generate_sql_from_question method"""

    @patch("axiompy.reasoning.client.HTTPClientFactory.create")
    def test_generate_sql_from_question(self, mock_factory):
        """Test generating SQL from natural language"""
        mock_http_client = MagicMock()
        mock_factory.return_value = mock_http_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "SELECT * FROM users"}
        mock_http_client.post.return_value = mock_response

        metadata = DatasetMetadata(
            dataset="test",
            description="Test DB",
            scope=ScopeMetadata(geographic="Global"),
            schema={"users": TableSchemaMetadata(columns={"id": "INTEGER", "name": "TEXT"})},
        )

        client = AIClient(provider="ollama", model="mistral", endpoint="http://localhost:11434")

        sql = client.generate_sql_from_question("Select all users", metadata)
        assert sql == "SELECT * FROM users"


class TestAIClientCache:
    """Tests for cache management"""

    @patch("axiompy.reasoning.client.HTTPClientFactory.create")
    def test_clear_cache(self, mock_factory):
        """Test clearing the cache"""
        mock_http_client = MagicMock()
        mock_factory.return_value = mock_http_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Response"}
        mock_http_client.post.return_value = mock_response

        client = AIClient(provider="ollama", model="mistral", endpoint="http://localhost:11434")

        # Cache something
        client.generate_completion("Test", use_cache=True)
        assert mock_http_client.post.call_count == 1

        # Use cache
        client.generate_completion("Test", use_cache=True)
        assert mock_http_client.post.call_count == 1

        # Clear cache
        client.clear_cache()

        # Should hit provider again
        client.generate_completion("Test", use_cache=True)
        assert mock_http_client.post.call_count == 2

    @patch("axiompy.reasoning.client.HTTPClientFactory.create")
    def test_get_cache_info(self, mock_factory):
        """Test getting cache statistics"""
        mock_http_client = MagicMock()
        mock_factory.return_value = mock_http_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Response"}
        mock_http_client.post.return_value = mock_response

        client = AIClient(provider="ollama", model="mistral", endpoint="http://localhost:11434")

        # Make some calls
        client.generate_completion("Test 1", use_cache=True)
        client.generate_completion("Test 1", use_cache=True)  # Cache hit
        client.generate_completion("Test 2", use_cache=True)  # Cache miss

        info = client.get_cache_info()
        assert info["hits"] >= 1
        assert info["misses"] >= 1
        assert info["maxsize"] == 128


class TestReasoningFactory:
    """Tests for ReasoningFactory"""

    def test_create_ollama(self):
        """Test creating Ollama client"""
        client = ReasoningFactory.create(ReasoningProvider.OLLAMA)
        assert client.provider == OllamaProviderConfig
        assert client.model == "mistral"

    def test_create_ollama_custom_model(self):
        """Test creating Ollama with custom model"""
        from axiompy.reasoning.settings import ReasoningSettings

        client = ReasoningFactory.create(
            ReasoningProvider.OLLAMA,
            settings=ReasoningSettings(model="llama2"),
        )
        assert client.model == "llama2"

    def test_create_openai(self):
        """Test creating OpenAI client"""
        from axiompy.reasoning.settings import ReasoningSettings

        client = ReasoningFactory.create(
            ReasoningProvider.OPENAI,
            settings=ReasoningSettings(api_key="sk-test"),
        )
        assert client.provider == OpenAIProviderConfig
        assert client.api_key == "sk-test"

    def test_create_anthropic(self):
        """Test creating Anthropic client"""
        from axiompy.reasoning.providers.anthropic import AnthropicProviderConfig
        from axiompy.reasoning.settings import ReasoningSettings

        client = ReasoningFactory.create(
            ReasoningProvider.ANTHROPIC,
            settings=ReasoningSettings(api_key="sk-ant-test"),
        )
        assert client.provider == AnthropicProviderConfig

    def test_create_invalid_provider(self):
        """Test creating with invalid provider - enum prevents this at type level"""
        # With enum, invalid providers are caught at type-check time
        # This test now verifies the error message format
        with pytest.raises((ValueError, AttributeError)):
            # Try to create with invalid string (would fail type check)
            ReasoningFactory.create("nonexistent")  # type: ignore

    def test_create_ollama_helper(self):
        """Test create_ollama helper method"""
        client = ReasoningFactory.create_ollama(model="llama2")
        assert client.model == "llama2"

    def test_create_openai_helper(self):
        """Test create_openai helper method"""
        client = ReasoningFactory.create_openai(api_key="sk-test", model="gpt-4")
        assert client.model == "gpt-4"
        assert client.api_key == "sk-test"

    def test_create_anthropic_helper(self):
        """Test create_anthropic helper method"""
        client = ReasoningFactory.create_anthropic(api_key="sk-ant-test", model="claude-3-opus")
        assert client.model == "claude-3-opus"
        assert client.api_key == "sk-ant-test"
