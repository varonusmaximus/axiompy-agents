"""Additional coverage for reasoning agents and validators."""

from unittest.mock import MagicMock

from axiompy.reasoning.agents.sql_generator import SQLGenerator
from axiompy.reasoning.agents.validation_pipeline import QueryValidationPipeline
from axiompy.reasoning.factory import ReasoningFactory
from axiompy.reasoning.types import ReasoningProvider
from axiompy.reasoning.validators import SQLValidator


class TestSQLGenerator:
    """Tests for SQLGenerator."""

    def test_generate_calls_client(self):
        """Test SQL generation delegates to AI client when validation passes."""
        client = ReasoningFactory.create_mock()
        client.generate_sql_from_question = MagicMock(return_value="SELECT 1")  # type: ignore[method-assign]
        pipeline = MagicMock(spec=QueryValidationPipeline)
        pipeline.validate.return_value = MagicMock(valid=True, errors=[])
        gen = SQLGenerator(client, validation_pipeline=pipeline)
        metadata = MagicMock()
        sql = gen.generate("count rows", metadata=metadata, db_connection=None)
        assert sql == "SELECT 1"
        client.generate_sql_from_question.assert_called()


class TestSQLValidator:
    """Tests for SQLValidator."""

    def test_validate_syntax_rejects_empty(self):
        """Test empty SQL fails validation."""
        result = SQLValidator.validate_syntax("")
        assert not result.valid

    def test_validate_syntax_accepts_select(self):
        """Test simple SELECT passes syntax check."""
        result = SQLValidator.validate_syntax("SELECT id FROM users")
        assert result.valid


class TestReasoningFactoryMock:
    """Tests for ReasoningFactory.create_mock."""

    def test_create_mock_cycles_responses(self):
        """Test mock client cycles canned responses."""
        client = ReasoningFactory.create_mock(
            responses=["a", "b"],
            provider=ReasoningProvider.OLLAMA,
        )
        assert client.generate_completion(prompt="x", use_cache=False) == "a"
        assert client.generate_completion(prompt="x", use_cache=False) == "b"
