"""AIClient - Provider-agnostic AI service for LLM operations

This module provides a unified interface for interacting with various AI providers
(Ollama, OpenAI, Anthropic, etc.) without vendor lock-in.
"""

from __future__ import annotations

import functools
from typing import Any, Optional, Type, Union

from axiompy.io.http import HTTPClient, HTTPClientFactory
from axiompy.reasoning.metadata import DatasetMetadata
from axiompy.reasoning.providers.base import ProviderConfig
from axiompy.reasoning.providers.factory import get_provider


class AIClient:
    """
    Provider-agnostic AI client for LLM operations.

    Enables switching between different AI providers (Ollama, OpenAI, Anthropic)
    without changing application code. Uses AxiomPy's HTTPClient for all requests.

    Features:
    - Provider abstraction - same code works with any provider
    - Built-in response caching with lru_cache for expensive calls
    - Type-safe with full type hints
    - Comprehensive error handling

    Example:
        # Create client for Ollama
        client = AIClient(
            provider="ollama",
            model="mistral",
            endpoint="http://localhost:11434/api/generate"
        )

        # Generate completion
        response = client.generate_completion(
            prompt="Explain SQL queries in simple terms",
            temperature=0.7,
            max_tokens=500
        )

        # Switch to OpenAI
        client = AIClient(
            provider="openai",
            model="gpt-4",
            endpoint="https://api.openai.com/v1/chat/completions",
            api_key="sk-..."  # Will be read from environment if not provided
        )

        # Same method calls work with different provider
        response = client.generate_completion(
            prompt="Explain SQL queries",
            temperature=0.7
        )
    """

    def __init__(
        self,
        provider: Union[str, Type[ProviderConfig]],
        model: str,
        endpoint: str,
        api_key: Optional[str] = None,
        http_client: Optional[HTTPClient] = None,
        cache_size: int = 128,
    ):
        """
        Initialize AIClient.

        Args:
            provider: Provider name ("ollama", "openai", "anthropic") or ProviderConfig class
            model: Model name/ID for the provider
            endpoint: API endpoint URL
            api_key: API key for authentication (optional, may come from environment)
            http_client: Custom HTTPClient (default: creates new one)
            cache_size: LRU cache size for expensive calls (default: 128)

        Raises:
            ValueError: If provider name is invalid
            TypeError: If provider is not ProviderConfig subclass
        """
        # Resolve provider
        if isinstance(provider, str):
            self.provider: Type[ProviderConfig] = get_provider(provider)
        elif isinstance(provider, type) and issubclass(provider, ProviderConfig):
            self.provider = provider
        else:
            raise TypeError(
                f"provider must be provider name (str) or ProviderConfig subclass, "
                f"got {type(provider)}"
            )

        self.model = model
        self.endpoint = endpoint
        self.api_key = api_key
        self.http_client = http_client or HTTPClientFactory.create(timeout_secs=30)
        self._cache_size = cache_size

        # Initialize cached methods
        self._cached_generate_completion = functools.lru_cache(maxsize=cache_size)(
            self._generate_completion_impl
        )

    def generate_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        top_p: float = 1.0,
        top_k: int = 0,
        use_cache: bool = True,
    ) -> str:
        """
        Generate completion for a prompt.

        Args:
            prompt: User prompt/question
            temperature: Sampling temperature (0.0-1.0, default: 0.7)
            max_tokens: Maximum tokens in response (default: 500)
            top_p: Nucleus sampling parameter (default: 1.0)
            top_k: Top-k sampling parameter (default: 0, disabled)
            use_cache: Whether to use caching (default: True)

        Returns:
            Generated completion text

        Raises:
            ConnectionError: If provider API request fails
            ValueError: If response format is invalid
        """
        if use_cache:
            return self._cached_generate_completion(prompt, temperature, max_tokens, top_p, top_k)
        else:
            return self._generate_completion_impl(prompt, temperature, max_tokens, top_p, top_k)

    def _generate_completion_impl(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        top_k: int,
    ) -> str:
        """
        Implementation of completion generation (cacheable).

        Args: Same as generate_completion()

        Returns:
            Generated text
        """
        # Format prompt for provider
        prompt_dict = {"user": prompt}
        formatted_prompt = self.provider.format_prompt(prompt_dict)

        # Build API payload
        payload = self.provider.build_payload(
            formatted_prompt=formatted_prompt,
            model=self.model,
            endpoint=self.endpoint,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
        )

        # Add authentication if needed
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Make HTTP request
        response = self.http_client.post(
            url=self.endpoint,
            json=payload,
            headers=headers if headers else None,
        )

        if response.status_code >= 400:
            raise ConnectionError(f"Provider API error ({response.status_code}): {response.text}")

        # Parse response
        response_json = response.json()
        return self.provider.parse_response(response_json)

    def generate_sql_from_question(
        self,
        question: str,
        metadata: DatasetMetadata,
        examples: Optional[list[tuple[str, str]]] = None,
    ) -> str:
        """
        Generate SQL query from natural language question.

        Args:
            question: Natural language question
            metadata: Dataset metadata for context
            examples: Optional list of (question, sql) tuples for few-shot learning

        Returns:
            Generated SQL query

        Example:
            question = "How many homicides in 2023?"
            metadata = crime_service.get_metadata()
            sql = client.generate_sql_from_question(question, metadata)
            # Returns: "SELECT COUNT(*) FROM incidents WHERE
            # crime_type='HOMICIDE' AND YEAR(date)=2023"
        """
        from axiompy.reasoning.metadata_helpers import format_schema_for_llm

        # Build system prompt with schema
        schema_text = format_schema_for_llm(metadata)
        system_prompt = (
            "You are an expert SQL query generator. "
            "Generate accurate SQL queries based on natural language questions. "
            "CRITICAL RULES:\n"
            "1. Return ONLY the SQL query, no explanations or markdown\n"
            "2. ALWAYS use table names or aliases for column references "
            "(e.g., products.category NOT category)\n"
            "3. When joining tables, qualify ALL selected columns\n"
            "4. Use aliases: o=orders, p=products, c=customers\n\n" + schema_text
        )

        # Add examples if provided
        if examples:
            system_prompt += "\n\nExamples:\n"
            for example_q, example_sql in examples:
                system_prompt += f"\nQ: {example_q}\nSQL: {example_sql}"

        # Generate SQL
        prompt_dict = {
            "system": system_prompt,
            "user": f"Generate SQL for: {question}",
        }
        formatted_prompt = self.provider.format_prompt(prompt_dict)

        payload = self.provider.build_payload(
            formatted_prompt=formatted_prompt,
            model=self.model,
            endpoint=self.endpoint,
            temperature=0.1,  # Low temperature for deterministic SQL
            max_tokens=500,
        )

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = self.http_client.post(
            url=self.endpoint,
            json=payload,
            headers=headers if headers else None,
        )

        if response.status_code >= 400:
            raise ConnectionError(
                f"SQL generation failed ({response.status_code}): "
                f"{response.text if hasattr(response, 'text') else str(response)}"
            )

        response_json = response.json()
        sql = self.provider.parse_response(response_json)

        # Log raw SQL for debugging
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(f"Raw SQL from AI: {sql[:200]}...")

        # Clean up SQL - extract actual SQL from potential explanation text
        sql = self._extract_sql_from_response(sql)

        logger.debug(f"Cleaned SQL: {sql[:200]}...")

        return sql.strip()

    def _extract_sql_from_response(self, text: str) -> str:
        """
        Extract SQL from AI response that may contain explanation or markdown.

        Handles cases like:
        - ```sql\nSELECT...```
        - Here is the query:\nSELECT...
        - SELECT... (with trailing explanation)
        - <s> SELECT... (special tokens from models)

        Args:
            text: Raw response from AI

        Returns:
            Cleaned SQL query
        """
        text = text.strip()

        # Remove special tokens that some models add
        # Common tokens: <s>, </s>, <|endoftext|>, etc.
        import re

        text = re.sub(r"</?s>|<\|endoftext\|>|<\|im_start\|>|<\|im_end\|>", "", text)
        text = text.strip()

        # Check if response is empty or whitespace only
        if not text or text.isspace():
            raise ValueError("AI returned empty response")

        # Remove markdown code blocks with language specifier
        if "```sql" in text:
            # Extract content between ```sql and ```
            parts = text.split("```sql")
            if len(parts) > 1:
                sql_part = parts[1].split("```")[0].strip()
                sql_keywords = ("SELECT", "INSERT", "UPDATE", "DELETE", "WITH")
                if sql_part and sql_part.upper().startswith(sql_keywords):
                    return sql_part.rstrip(";").strip()

        # Remove generic markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            sql_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```"):
                    in_block = not in_block
                elif in_block:
                    sql_lines.append(line)
            if sql_lines:
                sql = "\n".join(sql_lines).strip()
                if sql.upper().startswith(("SELECT", "INSERT", "UPDATE", "DELETE", "WITH")):
                    return sql.rstrip(";").strip()

        # Look for SQL by finding keywords
        lines = text.split("\n")
        sql_start_idx = -1
        _sql_end_idx = len(lines)  # reserved for multi-line SQL boundary detection

        # Find where SQL starts
        for i, line in enumerate(lines):
            upper_line = line.strip().upper()
            sql_kw = ["SELECT", "INSERT", "UPDATE", "DELETE", "WITH"]
            if any(upper_line.startswith(kw) for kw in sql_kw):
                sql_start_idx = i
                break

        # If we found SQL start, collect until we hit non-SQL lines
        if sql_start_idx >= 0:
            sql_lines = []
            for i in range(sql_start_idx, len(lines)):
                line = lines[i]
                upper_line = line.strip().upper()

                # Stop at explanation markers (but not SQL keywords)
                sql_keywords_list = [
                    "SELECT",
                    "FROM",
                    "WHERE",
                    "GROUP",
                    "ORDER",
                    "JOIN",
                    "LIMIT",
                    "UNION",
                    "LEFT",
                    "RIGHT",
                    "INNER",
                    "OUTER",
                    "ON",
                    "AND",
                    "OR",
                    "HAVING",
                    "CASE",
                    "WHEN",
                    "THEN",
                    "END",
                    "AS",
                    "DISTINCT",
                ]
                if upper_line and not any(kw in upper_line for kw in sql_keywords_list):
                    # Check if this looks like explanation
                    explanation_markers = [
                        "THIS",
                        "NOTE",
                        "EXPLANATION",
                        "HERE",
                        "THE",
                        "RESULT",
                        "QUERY",
                        "I",
                        "HOPE",
                    ]
                    if any(upper_line.startswith(p) for p in explanation_markers):
                        break

                sql_lines.append(line)

            if sql_lines:
                sql = "\n".join(sql_lines).strip()
                # Remove trailing explanation on last line
                sql_lines = sql.split("\n")
                last_line = sql_lines[-1].strip().upper()
                end_markers = ["THIS", "NOTE", "EXPLANATION", "THE", "HOPE", "RESULT"]
                if any(last_line.startswith(p) for p in end_markers):
                    sql_lines = sql_lines[:-1]

                sql = "\n".join(sql_lines).strip()
                return sql.rstrip(";").strip()

        # Fallback: return text as-is if we couldn't extract
        return text.rstrip(";").strip()

    def generate_insight(
        self,
        data: list[dict[str, Any]],
        prompt_dict: dict[str, str],
        domain: str = "general",
    ) -> str:
        """
        Generate AI insights from data.

        Args:
            data: List of dictionaries representing query results
            prompt_dict: Prompt template with "system" and "user" keys
                - "system": System prompt (role/behavior)
                - "user": User question template (will be formatted with data context)
            domain: Domain for context (default: "general")

        Returns:
            AI-generated insight text

        Example:
            data = [{"crime_type": "HOMICIDE", "count": 45}, ...]
            prompt = {
                "system": "You are a crime data analyst",
                "user": "Analyze this data and provide insights: {data}"
            }
            insight = client.generate_insight(data, prompt)
        """
        # Format data for context
        data_text = "\n".join([str(row) for row in data[:10]])  # Limit to first 10 rows
        if len(data) > 10:
            data_text += f"\n... and {len(data) - 10} more rows"

        # Build prompt
        prompt_dict_filled = {
            "system": prompt_dict.get("system", "You are a helpful analyst"),
            "user": prompt_dict.get("user", "").format(data=data_text),
        }

        # Generate insight
        formatted_prompt = self.provider.format_prompt(prompt_dict_filled)

        payload = self.provider.build_payload(
            formatted_prompt=formatted_prompt,
            model=self.model,
            endpoint=self.endpoint,
            temperature=0.7,  # More creative for insights
            max_tokens=500,
        )

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = self.http_client.post(
            url=self.endpoint,
            json=payload,
            headers=headers if headers else None,
        )

        if response.status_code >= 400:
            resp_text = response.text if hasattr(response, "text") else str(response)
            raise ConnectionError(
                f"Insight generation failed ({response.status_code}): {resp_text}"
            )

        response_json = response.json()
        return self.provider.parse_response(response_json)

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cached_generate_completion.cache_clear()

    def get_cache_info(self) -> dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache info: hits, misses, currsize, maxsize
        """
        cache_info = self._cached_generate_completion.cache_info()
        return {
            "hits": cache_info.hits,
            "misses": cache_info.misses,
            "currsize": cache_info.currsize,
            "maxsize": cache_info.maxsize,
        }
