"""Tests for reasoning provider configurations"""

import pytest

from axiompy.reasoning.providers import get_provider, list_providers
from axiompy.reasoning.providers.anthropic import AnthropicProviderConfig
from axiompy.reasoning.providers.ollama import OllamaProviderConfig
from axiompy.reasoning.providers.openai import OpenAIProviderConfig


class TestProviderFactory:
    """Tests for provider factory functions"""

    def test_get_provider_ollama(self):
        """Test getting Ollama provider"""
        provider = get_provider("ollama")
        assert provider == OllamaProviderConfig

    def test_get_provider_openai(self):
        """Test getting OpenAI provider"""
        provider = get_provider("openai")
        assert provider == OpenAIProviderConfig

    def test_get_provider_anthropic(self):
        """Test getting Anthropic provider"""
        provider = get_provider("anthropic")
        assert provider == AnthropicProviderConfig

    def test_get_provider_case_insensitive(self):
        """Test that provider lookup is case-insensitive"""
        provider1 = get_provider("OLLAMA")
        provider2 = get_provider("Ollama")
        assert provider1 == provider2 == OllamaProviderConfig

    def test_get_provider_invalid(self):
        """Test getting invalid provider raises error"""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("nonexistent")

    def test_list_providers(self):
        """Test listing all providers"""
        providers = list_providers()
        assert "ollama" in providers
        assert "openai" in providers
        assert "anthropic" in providers
        assert len(providers) >= 3


class TestOllamaProvider:
    """Tests for Ollama provider configuration"""

    def test_format_prompt_with_system_and_user(self):
        """Test formatting prompt with system and user messages"""
        prompt_dict = {"system": "You are a SQL expert", "user": "Write a query"}
        formatted = OllamaProviderConfig.format_prompt(prompt_dict)

        assert isinstance(formatted, str)
        assert "SQL expert" in formatted
        assert "Write a query" in formatted

    def test_format_prompt_user_only(self):
        """Test formatting prompt with only user message"""
        prompt_dict = {"user": "Write a query"}
        formatted = OllamaProviderConfig.format_prompt(prompt_dict)

        assert formatted == "Write a query"

    def test_build_payload(self):
        """Test building Ollama API payload"""
        payload = OllamaProviderConfig.build_payload(
            formatted_prompt="Test prompt",
            model="mistral",
            endpoint="http://localhost:11434",
            temperature=0.8,
            max_tokens=200,
        )

        assert payload["model"] == "mistral"
        assert payload["prompt"] == "Test prompt"
        assert payload["temperature"] == 0.8
        assert payload["num_predict"] == 200
        assert payload["stream"] is False

    def test_parse_response(self):
        """Test parsing Ollama response"""
        response = {"response": "SELECT * FROM users"}
        result = OllamaProviderConfig.parse_response(response)

        assert result == "SELECT * FROM users"

    def test_parse_response_missing_field(self):
        """Test parsing response with missing field raises error"""
        response = {"result": "missing"}

        with pytest.raises(KeyError):
            OllamaProviderConfig.parse_response(response)


class TestOpenAIProvider:
    """Tests for OpenAI provider configuration"""

    def test_format_prompt_with_system_and_user(self):
        """Test formatting prompt for OpenAI"""
        prompt_dict = {"system": "You are a SQL expert", "user": "Write a query"}
        formatted = OpenAIProviderConfig.format_prompt(prompt_dict)

        assert isinstance(formatted, list)
        assert len(formatted) == 2
        assert formatted[0]["role"] == "system"
        assert formatted[1]["role"] == "user"
        assert formatted[0]["content"] == "You are a SQL expert"
        assert formatted[1]["content"] == "Write a query"

    def test_format_prompt_user_only(self):
        """Test formatting with only user message"""
        prompt_dict = {"user": "Write a query"}
        formatted = OpenAIProviderConfig.format_prompt(prompt_dict)

        assert len(formatted) == 1
        assert formatted[0]["role"] == "user"

    def test_build_payload(self):
        """Test building OpenAI API payload"""
        messages = [{"role": "system", "content": "Expert"}, {"role": "user", "content": "Query"}]
        payload = OpenAIProviderConfig.build_payload(
            formatted_prompt=messages,
            model="gpt-4",
            endpoint="https://api.openai.com",
            temperature=0.7,
            max_tokens=500,
        )

        assert payload["model"] == "gpt-4"
        assert payload["messages"] == messages
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 500

    def test_parse_response(self):
        """Test parsing OpenAI response"""
        response = {"choices": [{"message": {"content": "SELECT * FROM users"}}]}
        result = OpenAIProviderConfig.parse_response(response)

        assert result == "SELECT * FROM users"

    def test_parse_response_missing_choices(self):
        """Test parsing response with missing choices"""
        response = {}

        with pytest.raises(KeyError):
            OpenAIProviderConfig.parse_response(response)

    def test_parse_response_empty_choices(self):
        """Test parsing response with empty choices"""
        response = {"choices": []}

        with pytest.raises(KeyError):
            OpenAIProviderConfig.parse_response(response)


class TestAnthropicProvider:
    """Tests for Anthropic provider configuration"""

    def test_format_prompt_with_system_and_user(self):
        """Test formatting prompt for Anthropic"""
        prompt_dict = {"system": "You are a SQL expert", "user": "Write a query"}
        formatted = AnthropicProviderConfig.format_prompt(prompt_dict)

        assert isinstance(formatted, str)
        assert "SQL expert" in formatted
        assert "Human:" in formatted
        assert "Assistant:" in formatted

    def test_format_prompt_user_only(self):
        """Test formatting with only user message"""
        prompt_dict = {"user": "Write a query"}
        formatted = AnthropicProviderConfig.format_prompt(prompt_dict)

        assert "Human:" in formatted
        assert "Assistant:" in formatted

    def test_build_payload(self):
        """Test building Anthropic API payload"""
        payload = AnthropicProviderConfig.build_payload(
            formatted_prompt="Formatted prompt",
            model="claude-2",
            endpoint="https://api.anthropic.com",
            temperature=0.7,
            max_tokens=500,
        )

        assert payload["model"] == "claude-2"
        assert payload["prompt"] == "Formatted prompt"
        assert payload["temperature"] == 0.7
        assert payload["max_tokens_to_sample"] == 500

    def test_parse_response(self):
        """Test parsing Anthropic response"""
        response = {"completion": "SELECT * FROM users"}
        result = AnthropicProviderConfig.parse_response(response)

        assert result == "SELECT * FROM users"

    def test_parse_response_missing_field(self):
        """Test parsing response with missing field"""
        response = {"result": "missing"}

        with pytest.raises(KeyError):
            AnthropicProviderConfig.parse_response(response)


class TestProviderConsistency:
    """Tests ensuring all providers have consistent interface"""

    def test_all_providers_have_format_prompt(self):
        """Test all providers implement format_prompt"""
        providers = [OllamaProviderConfig, OpenAIProviderConfig, AnthropicProviderConfig]

        for provider in providers:
            assert hasattr(provider, "format_prompt")
            assert callable(provider.format_prompt)

    def test_all_providers_have_build_payload(self):
        """Test all providers implement build_payload"""
        providers = [OllamaProviderConfig, OpenAIProviderConfig, AnthropicProviderConfig]

        for provider in providers:
            assert hasattr(provider, "build_payload")
            assert callable(provider.build_payload)

    def test_all_providers_have_parse_response(self):
        """Test all providers implement parse_response"""
        providers = [OllamaProviderConfig, OpenAIProviderConfig, AnthropicProviderConfig]

        for provider in providers:
            assert hasattr(provider, "parse_response")
            assert callable(provider.parse_response)
