"""Tests for Phase 2 reasoning components: PromptBuilder, QueryAgent, SQLValidator"""

from unittest.mock import MagicMock, patch

import pytest

from axiompy.reasoning.agents.query import QueryAgent
from axiompy.reasoning.base import BaseDatasetService
from axiompy.reasoning.client import AIClient
from axiompy.reasoning.metadata import (
    DatasetMetadata,
    ScopeMetadata,
    TableSchemaMetadata,
)
from axiompy.reasoning.prompts import DynamicPromptBuilder
from axiompy.reasoning.validators import SQLValidator, ValidationResult


# ==================== DynamicPromptBuilder Tests ====================


class TestDynamicPromptBuilder:
    """Tests for DynamicPromptBuilder"""

    def test_build_sql_generation_prompt_basic(self):
        """Test building basic SQL generation prompt"""
        metadata = DatasetMetadata(
            dataset="test",
            description="Test database",
            scope=ScopeMetadata(geographic="Global"),
            schema={"users": TableSchemaMetadata(columns={"id": "INT", "name": "TEXT"})},
        )

        prompt = DynamicPromptBuilder.build_sql_generation_prompt(
            question="List all users", metadata=metadata
        )

        assert "system" in prompt
        assert "user" in prompt
        assert "SQL" in prompt["system"]
        assert "List all users" in prompt["user"]

    def test_build_sql_generation_prompt_with_instructions(self):
        """Test SQL prompt with additional instructions"""
        metadata = DatasetMetadata(
            dataset="test",
            description="Test",
            scope=ScopeMetadata(geographic="Global"),
            schema={"users": TableSchemaMetadata(columns={"id": "INT"})},
        )

        prompt = DynamicPromptBuilder.build_sql_generation_prompt(
            question="Find users",
            metadata=metadata,
            additional_instructions="Only return first 10 rows",
        )

        assert "Only return first 10 rows" in prompt["system"]

    def test_build_planning_prompt(self):
        """Test building planning prompt for multi-dataset routing"""
        metadata1 = DatasetMetadata(
            dataset="crime",
            description="Crime data",
            scope=ScopeMetadata(geographic="Chicago"),
            schema={"incidents": TableSchemaMetadata(columns={"id": "INT"})},
        )
        metadata2 = DatasetMetadata(
            dataset="housing",
            description="Housing data",
            scope=ScopeMetadata(geographic="Chicago"),
            schema={"properties": TableSchemaMetadata(columns={"id": "INT"})},
        )

        prompt = DynamicPromptBuilder.build_planning_prompt(
            question="Compare crime and housing trends",
            available_datasets={"crime": metadata1, "housing": metadata2},
        )

        assert "crime" in prompt["system"]
        assert "housing" in prompt["system"]
        assert "Compare crime and housing" in prompt["user"]

    def test_build_insight_generation_prompt(self):
        """Test building insight generation prompt"""
        metadata = DatasetMetadata(
            dataset="test",
            description="Test",
            scope=ScopeMetadata(geographic="Global"),
            schema={"data": TableSchemaMetadata(columns={"value": "INT"})},
        )
        results = [{"value": 100}, {"value": 200}]

        prompt = DynamicPromptBuilder.build_insight_generation_prompt(
            question="Analyze the data", results=results, metadata=metadata
        )

        assert "analyst" in prompt["system"].lower()
        assert "100" in prompt["user"]
        assert "200" in prompt["user"]

    def test_build_validation_prompt(self):
        """Test building SQL validation prompt"""
        metadata = DatasetMetadata(
            dataset="test",
            description="Test",
            scope=ScopeMetadata(geographic="Global"),
            schema={"users": TableSchemaMetadata(columns={"id": "INT"})},
        )

        prompt = DynamicPromptBuilder.build_validation_prompt(
            sql="SELECT * FROM users", metadata=metadata
        )

        assert "expert" in prompt["system"].lower()
        assert "SELECT * FROM users" in prompt["user"]


# ==================== QueryAgent Tests ====================


class MockDatasetService(BaseDatasetService):
    """Mock dataset service for testing"""

    dataset_name = "test_dataset"
    description = "Test dataset service"

    def __init__(self, metadata: DatasetMetadata = None):
        self.metadata = metadata or DatasetMetadata(
            dataset="test",
            description="Test",
            scope=ScopeMetadata(geographic="Global"),
            schema={"table": TableSchemaMetadata(columns={"id": "INT", "name": "TEXT"})},
        )

    def query(self, sql: str, limit: int = None):
        return [{"id": 1, "name": "Test"}]

    def get_capabilities(self):
        return ["search", "filter"]

    def get_metadata(self):
        return self.metadata


class TestQueryAgent:
    """Tests for QueryAgent"""

    def test_init_single_dataset(self):
        """Test initializing with single dataset"""
        service = MockDatasetService()
        mock_client = MagicMock(spec=AIClient)

        agent = QueryAgent(ai_client=mock_client, datasets={"test": service})

        assert agent.ai_client == mock_client
        assert "test" in agent.datasets

    def test_init_multiple_datasets(self):
        """Test initializing with multiple datasets"""
        service1 = MockDatasetService()
        service2 = MockDatasetService()
        mock_client = MagicMock(spec=AIClient)

        agent = QueryAgent(
            ai_client=mock_client, datasets={"dataset1": service1, "dataset2": service2}
        )

        assert len(agent.datasets) == 2

    def test_init_no_datasets_raises_error(self):
        """Test that initialization without datasets raises error"""
        mock_client = MagicMock(spec=AIClient)

        with pytest.raises(ValueError, match="At least one dataset"):
            QueryAgent(ai_client=mock_client, datasets={})

    @patch("axiompy.reasoning.agents.query.DynamicPromptBuilder.build_insight_generation_prompt")
    def test_execute_query_single_dataset(self, mock_prompt):
        """Test executing query with single dataset"""
        service = MockDatasetService()
        mock_client = MagicMock(spec=AIClient)
        mock_client.generate_sql_from_question.return_value = "SELECT * FROM table WHERE id = 1"
        mock_client.generate_completion.return_value = "Insight text"

        agent = QueryAgent(ai_client=mock_client, datasets={"test": service})

        result = agent.execute_query("Get test data")

        assert "results" in result
        assert "sql" in result
        assert "dataset" in result
        assert result["dataset"] == "test"
        assert len(result["results"]) > 0

    def test_execute_query_planning_disabled(self):
        """Test that planning is skipped when disabled"""
        service = MockDatasetService()
        mock_client = MagicMock(spec=AIClient)
        mock_client.generate_sql_from_question.return_value = "SELECT * FROM table"

        agent = QueryAgent(ai_client=mock_client, datasets={"test": service}, enable_planning=False)

        result = agent.execute_query("Query data")
        assert result["dataset"] == "test"

    def test_execute_query_insights_disabled(self):
        """Test that insights are skipped when disabled"""
        service = MockDatasetService()
        mock_client = MagicMock(spec=AIClient)
        mock_client.generate_sql_from_question.return_value = "SELECT * FROM table"

        agent = QueryAgent(ai_client=mock_client, datasets={"test": service}, enable_insights=False)

        result = agent.execute_query("Query data")
        assert result["insights"] is None

    def test_get_dataset_names(self):
        """Test getting dataset names"""
        service1 = MockDatasetService()
        service2 = MockDatasetService()
        mock_client = MagicMock(spec=AIClient)

        agent = QueryAgent(
            ai_client=mock_client, datasets={"dataset1": service1, "dataset2": service2}
        )

        names = agent.get_dataset_names()
        assert "dataset1" in names
        assert "dataset2" in names

    def test_get_dataset_capabilities(self):
        """Test getting capabilities for all datasets"""
        service = MockDatasetService()
        mock_client = MagicMock(spec=AIClient)

        agent = QueryAgent(ai_client=mock_client, datasets={"test": service})

        capabilities = agent.get_dataset_capabilities()
        assert "test" in capabilities
        assert "search" in capabilities["test"]


# ==================== SQLValidator Tests ====================


class TestSQLValidator:
    """Tests for SQLValidator"""

    def test_validate_columns_valid(self):
        """Test validating valid columns"""
        schema = {"id", "name", "email"}
        sql = "SELECT id, name FROM users WHERE email = 'test@test.com'"

        result = SQLValidator.validate_columns(sql, schema)

        assert result.valid
        assert len(result.errors) == 0

    def test_validate_columns_invalid(self):
        """Test validating invalid columns in strict mode"""
        schema = {"id", "name"}
        sql = "SELECT id, invalid_column FROM users"

        result = SQLValidator.validate_columns(sql, schema, strict=True)

        assert not result.valid
        assert len(result.errors) > 0
        assert "invalid_column" in result.missing_columns

    def test_validate_columns_strict_mode(self):
        """Test strict validation mode"""
        schema = {"id", "name"}
        sql = "SELECT id FROM users WHERE unknown_col > 5"

        result = SQLValidator.validate_columns(sql, schema, strict=True)

        assert not result.valid

    def test_extract_columns_select(self):
        """Test extracting columns from SELECT"""
        sql = "SELECT id, name, email FROM users"
        columns = SQLValidator.extract_columns(sql)

        assert "id" in columns
        assert "name" in columns
        assert "email" in columns

    def test_extract_columns_where(self):
        """Test extracting columns from WHERE"""
        sql = "SELECT * FROM users WHERE age > 18 AND status = 'active'"
        columns = SQLValidator.extract_columns(sql)

        assert "age" in columns
        assert "status" in columns

    def test_extract_columns_order_by(self):
        """Test extracting columns from ORDER BY"""
        sql = "SELECT * FROM users ORDER BY created_at DESC"
        columns = SQLValidator.extract_columns(sql)

        assert "created_at" in columns

    def test_extract_columns_group_by(self):
        """Test extracting columns from GROUP BY"""
        sql = "SELECT category, COUNT(*) FROM products GROUP BY category"
        columns = SQLValidator.extract_columns(sql)

        assert "category" in columns

    def test_extract_columns_functions(self):
        """Test extracting columns from functions"""
        sql = "SELECT COUNT(id), SUM(amount) FROM orders"
        columns = SQLValidator.extract_columns(sql)

        assert "id" in columns
        assert "amount" in columns

    def test_validate_tables_valid(self):
        """Test validating valid tables"""
        tables = {"users", "orders"}
        sql = "SELECT * FROM users u JOIN orders o ON u.id = o.user_id"

        result = SQLValidator.validate_tables(sql, tables)

        assert result.valid

    def test_validate_tables_invalid(self):
        """Test validating invalid tables"""
        tables = {"users"}
        sql = "SELECT * FROM users JOIN products ON users.id = products.user_id"

        result = SQLValidator.validate_tables(sql, tables)

        assert not result.valid
        assert len(result.errors) > 0

    def test_validate_syntax_basic(self):
        """Test basic SQL syntax validation"""
        sql = "SELECT * FROM users WHERE id = 1"

        result = SQLValidator.validate_syntax(sql)

        assert result.valid

    def test_validate_syntax_unmatched_parentheses(self):
        """Test validation catches unmatched parentheses"""
        sql = "SELECT * FROM users WHERE (id = 1"

        result = SQLValidator.validate_syntax(sql)

        assert not result.valid
        assert any("parenthes" in e.lower() for e in result.errors)

    def test_validate_syntax_unmatched_quotes(self):
        """Test validation catches unmatched quotes"""
        sql = "SELECT * FROM users WHERE name = 'test"

        result = SQLValidator.validate_syntax(sql)

        assert not result.valid

    def test_validate_syntax_missing_select(self):
        """Test validation warns about missing SELECT"""
        sql = "FROM users"

        result = SQLValidator.validate_syntax(sql)

        assert any("select" in w.lower() for w in result.warnings)

    def test_is_sql_keyword(self):
        """Test SQL keyword detection"""
        assert SQLValidator._is_sql_keyword("count")
        assert SQLValidator._is_sql_keyword("SELECT")
        assert SQLValidator._is_sql_keyword("where")
        assert not SQLValidator._is_sql_keyword("invalid_column")


class TestValidationResult:
    """Tests for ValidationResult dataclass"""

    def test_validation_result_creation(self):
        """Test creating validation result"""
        result = ValidationResult(valid=True, errors=[], warnings=["test warning"])

        assert result.valid
        assert len(result.errors) == 0
        assert len(result.warnings) == 1

    def test_validation_result_with_missing_columns(self):
        """Test validation result with missing columns"""
        result = ValidationResult(
            valid=False, errors=["Column not found"], warnings=[], missing_columns={"invalid_col"}
        )

        assert not result.valid
        assert "invalid_col" in result.missing_columns


# ==================== ErrorFeedbackGenerator Tests ====================


class TestErrorFeedbackGenerator:
    """Tests for ErrorFeedbackGenerator"""

    def test_generate_empty_sql_feedback(self):
        """Test feedback generation for empty SQL errors"""
        from axiompy.reasoning.agents.feedback import ErrorFeedbackGenerator
        from axiompy.validators import SQLErrorType

        feedback = ErrorFeedbackGenerator.generate(
            error_type=SQLErrorType.EMPTY_SQL,
            previous_sql="",
            errors=["SQL is empty"],
        )

        assert "PREVIOUS ATTEMPT FAILED" in feedback
        assert "(empty)" in feedback
        assert "empty_sql" in feedback
        assert "Must start with SELECT" in feedback

    def test_generate_syntax_error_feedback(self):
        """Test feedback generation for syntax errors"""
        from axiompy.reasoning.agents.feedback import ErrorFeedbackGenerator
        from axiompy.validators import SQLErrorType

        feedback = ErrorFeedbackGenerator.generate(
            error_type=SQLErrorType.SYNTAX_ERROR,
            previous_sql="SELECT * LIMIT 10",
            errors=["LIMIT requires FROM"],
        )

        assert "SELECT * LIMIT 10" in feedback
        assert "syntax_error" in feedback
        assert "LIMIT requires FROM clause" in feedback

    def test_generate_column_error_feedback(self):
        """Test feedback generation for column errors"""
        from axiompy.reasoning.agents.feedback import ErrorFeedbackGenerator
        from axiompy.validators import SQLErrorType

        feedback = ErrorFeedbackGenerator.generate(
            error_type=SQLErrorType.COLUMN_ERROR,
            previous_sql="SELECT invalid_col FROM users",
            errors=["Column not found"],
            context={"valid_columns": ["id", "name", "email"]},
        )

        assert "column_error" in feedback
        assert "email" in feedback
        assert "id" in feedback
        assert "name" in feedback

    def test_generate_column_error_without_context(self):
        """Test feedback generation for column errors without valid columns"""
        from axiompy.reasoning.agents.feedback import ErrorFeedbackGenerator
        from axiompy.validators import SQLErrorType

        feedback = ErrorFeedbackGenerator.generate(
            error_type=SQLErrorType.COLUMN_ERROR,
            previous_sql="SELECT bad_col FROM users",
            errors=["Column not found"],
        )

        assert "column_error" in feedback
        assert "Check spelling" in feedback

    def test_generate_database_error_feedback(self):
        """Test feedback generation for database errors"""
        from axiompy.reasoning.agents.feedback import ErrorFeedbackGenerator
        from axiompy.validators import SQLErrorType

        feedback = ErrorFeedbackGenerator.generate(
            error_type=SQLErrorType.DATABASE_ERROR,
            previous_sql="SELECT * FROM nonexistent",
            errors=["no such table"],
        )

        assert "database_error" in feedback
        assert "Table names are correct" in feedback
        assert "SQLite syntax" in feedback

    def test_generate_unknown_error_feedback(self):
        """Test feedback generation for unknown errors"""
        from axiompy.reasoning.agents.feedback import ErrorFeedbackGenerator
        from axiompy.validators import SQLErrorType

        feedback = ErrorFeedbackGenerator.generate(
            error_type=SQLErrorType.UNKNOWN,
            previous_sql="SELECT something",
            errors=["Unknown error"],
        )

        assert "PREVIOUS ATTEMPT FAILED" in feedback
        assert "Please generate a corrected SQL query" in feedback
