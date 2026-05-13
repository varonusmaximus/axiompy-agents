"""Tests for reasoning metadata module"""

import pytest

from axiompy.reasoning.metadata import (
    DatasetMetadata,
    ExampleMetadata,
    ScopeMetadata,
    TableSchemaMetadata,
)
from axiompy.reasoning.metadata_helpers import (
    extract_all_columns,
    extract_columns_from_sql,
    format_schema_for_llm,
    get_metadata_template,
    match_keywords,
    validate_metadata_completeness,
)


class TestScopeMetadata:
    """Tests for ScopeMetadata dataclass"""

    def test_basic_scope(self):
        """Test creating basic scope metadata"""
        scope = ScopeMetadata(geographic="Chicago, IL")
        assert scope.geographic == "Chicago, IL"
        assert scope.temporal is None
        assert scope.domain is None

    def test_full_scope(self):
        """Test scope with all fields"""
        scope = ScopeMetadata(
            geographic="Chicago, IL",
            temporal="2000-2024",
            domain="Public Safety",
            important="Data updated monthly",
        )
        assert scope.geographic == "Chicago, IL"
        assert scope.temporal == "2000-2024"
        assert scope.domain == "Public Safety"
        assert scope.important == "Data updated monthly"


class TestTableSchemaMetadata:
    """Tests for TableSchemaMetadata dataclass"""

    def test_basic_table_schema(self):
        """Test creating table schema"""
        schema = TableSchemaMetadata(columns={"id": "INTEGER", "name": "TEXT"})
        assert schema.columns == {"id": "INTEGER", "name": "TEXT"}
        assert schema.description is None
        assert schema.row_count is None
        assert schema.indexes is None

    def test_full_table_schema(self):
        """Test table schema with all fields"""
        schema = TableSchemaMetadata(
            columns={"id": "INTEGER", "name": "TEXT"},
            description="User table",
            row_count=1000,
            indexes=["id", "name"],
        )
        assert schema.row_count == 1000
        assert "id" in schema.indexes


class TestExampleMetadata:
    """Tests for ExampleMetadata dataclass"""

    def test_basic_example(self):
        """Test creating example metadata"""
        example = ExampleMetadata(question="How many users?", sql="SELECT COUNT(*) FROM users")
        assert example.question == "How many users?"
        assert example.sql == "SELECT COUNT(*) FROM users"

    def test_full_example(self):
        """Test example with all fields"""
        example = ExampleMetadata(
            question="How many users?",
            sql="SELECT COUNT(*) FROM users",
            expected_results="1000",
            keywords=["count", "users"],
        )
        assert example.expected_results == "1000"
        assert "count" in example.keywords


class TestDatasetMetadata:
    """Tests for DatasetMetadata dataclass"""

    def test_basic_metadata(self):
        """Test creating basic dataset metadata"""
        metadata = DatasetMetadata(
            dataset="users",
            description="User database",
            scope=ScopeMetadata(geographic="Global"),
            schema={"users": TableSchemaMetadata(columns={"id": "INTEGER", "name": "TEXT"})},
        )
        assert metadata.dataset == "users"
        assert metadata.description == "User database"
        assert "users" in metadata.schema

    def test_full_metadata(self):
        """Test metadata with all optional fields"""
        metadata = DatasetMetadata(
            dataset="crime",
            description="Crime data",
            scope=ScopeMetadata(geographic="Chicago", temporal="2000-2024", domain="Public Safety"),
            schema={
                "incidents": TableSchemaMetadata(columns={"id": "INTEGER", "crime_type": "TEXT"})
            },
            capabilities=["search", "filter"],
            keywords={"types": ["homicide", "theft"]},
            constraints=["Monthly updates"],
            common_mistakes={"wrong_table": "Use 'incidents' not 'crimes'"},
        )
        assert len(metadata.capabilities) == 2
        assert "homicide" in metadata.keywords["types"]


class TestExtractAllColumns:
    """Tests for extract_all_columns helper"""

    def test_single_table(self):
        """Test extracting columns from single table"""
        metadata = DatasetMetadata(
            dataset="test",
            description="Test",
            scope=ScopeMetadata(geographic="Test"),
            schema={
                "users": TableSchemaMetadata(columns={"id": "INT", "name": "TEXT", "email": "TEXT"})
            },
        )
        columns = extract_all_columns(metadata)
        assert columns == {"id", "name", "email"}

    def test_multiple_tables(self):
        """Test extracting columns from multiple tables"""
        metadata = DatasetMetadata(
            dataset="test",
            description="Test",
            scope=ScopeMetadata(geographic="Test"),
            schema={
                "users": TableSchemaMetadata(columns={"id": "INT", "name": "TEXT"}),
                "orders": TableSchemaMetadata(columns={"order_id": "INT", "user_id": "INT"}),
            },
        )
        columns = extract_all_columns(metadata)
        assert columns == {"id", "name", "order_id", "user_id"}


class TestExtractColumnsFromSQL:
    """Tests for extract_columns_from_sql helper"""

    def test_simple_select(self):
        """Test extracting columns from simple SELECT"""
        sql = "SELECT id, name FROM users"
        columns = extract_columns_from_sql(sql)
        assert "id" in columns
        assert "name" in columns

    def test_where_clause(self):
        """Test extracting columns from WHERE clause"""
        sql = "SELECT * FROM users WHERE age > 18 AND status = 'active'"
        columns = extract_columns_from_sql(sql)
        assert "age" in columns
        assert "status" in columns

    def test_order_by(self):
        """Test extracting columns from ORDER BY"""
        sql = "SELECT * FROM users ORDER BY created_at DESC"
        columns = extract_columns_from_sql(sql)
        assert "created_at" in columns

    def test_group_by(self):
        """Test extracting columns from GROUP BY"""
        sql = "SELECT category, COUNT(*) FROM products GROUP BY category"
        columns = extract_columns_from_sql(sql)
        assert "category" in columns

    def test_select_star(self):
        """Test that SELECT * doesn't add star"""
        sql = "SELECT * FROM users"
        columns = extract_columns_from_sql(sql)
        assert "*" not in columns

    def test_function_calls(self):
        """Test extracting columns from function calls"""
        sql = "SELECT COUNT(id), SUM(amount) FROM orders"
        columns = extract_columns_from_sql(sql)
        assert "id" in columns
        assert "amount" in columns

    def test_multiple_statements(self):
        """Test complex query with multiple clauses"""
        sql = """
            SELECT user_id, COUNT(*) as count
            FROM orders
            WHERE created_at > '2023-01-01' AND status = 'completed'
            GROUP BY user_id
            ORDER BY count DESC
        """
        columns = extract_columns_from_sql(sql)
        assert "user_id" in columns
        assert "created_at" in columns
        assert "status" in columns


class TestFormatSchemaForLLM:
    """Tests for format_schema_for_llm helper"""

    def test_simple_format(self):
        """Test formatting simple schema"""
        metadata = DatasetMetadata(
            dataset="test",
            description="Test database",
            scope=ScopeMetadata(geographic="Global"),
            schema={"users": TableSchemaMetadata(columns={"id": "INTEGER", "name": "TEXT"})},
        )
        formatted = format_schema_for_llm(metadata)

        assert "test" in formatted
        assert "users" in formatted
        assert "id" in formatted
        assert "name" in formatted

    def test_format_with_indexes(self):
        """Test formatting schema with indexes"""
        metadata = DatasetMetadata(
            dataset="crime",
            description="Crime data",
            scope=ScopeMetadata(geographic="Chicago", temporal="2000-2024"),
            schema={
                "incidents": TableSchemaMetadata(
                    columns={"id": "INTEGER", "date": "DATE"},
                    row_count=7000000,
                    indexes=["id", "date"],
                )
            },
            constraints=["Data updated monthly"],
        )
        formatted = format_schema_for_llm(metadata)

        assert "7,000,000" in formatted  # Formatted row count
        assert "indexes" in formatted.lower()
        assert "Constraints" in formatted


class TestMatchKeywords:
    """Tests for match_keywords helper"""

    def test_single_match(self):
        """Test keyword matching with single match"""
        metadata = DatasetMetadata(
            dataset="test",
            description="Test",
            scope=ScopeMetadata(geographic="Test"),
            schema={},
            keywords={"crime_types": ["homicide", "theft", "assault"]},
        )
        scores = match_keywords("How many homicides?", metadata)

        assert "crime_types" in scores
        assert scores["crime_types"] > 0

    def test_multiple_matches(self):
        """Test keyword matching with multiple keywords"""
        metadata = DatasetMetadata(
            dataset="test",
            description="Test",
            scope=ScopeMetadata(geographic="Test"),
            schema={},
            keywords={
                "crime_types": ["homicide", "theft"],
                "locations": ["downtown", "north side"],
            },
        )
        scores = match_keywords("Homicides in downtown", metadata)

        assert "crime_types" in scores
        assert "locations" in scores

    def test_no_matches(self):
        """Test when no keywords match"""
        metadata = DatasetMetadata(
            dataset="test",
            description="Test",
            scope=ScopeMetadata(geographic="Test"),
            schema={},
            keywords={"types": ["apple", "orange"]},
        )
        scores = match_keywords("bananas and grapes", metadata)

        assert len(scores) == 0

    def test_no_keywords_in_metadata(self):
        """Test when metadata has no keywords"""
        metadata = DatasetMetadata(
            dataset="test", description="Test", scope=ScopeMetadata(geographic="Test"), schema={}
        )
        scores = match_keywords("Any question", metadata)

        assert scores == {}


class TestValidateMetadataCompleteness:
    """Tests for validate_metadata_completeness helper"""

    def test_valid_complete_metadata(self):
        """Test validating complete metadata"""
        metadata = DatasetMetadata(
            dataset="users",
            description="User database",
            scope=ScopeMetadata(geographic="Global"),
            schema={"users": TableSchemaMetadata(columns={"id": "INTEGER"})},
            examples=[ExampleMetadata("q", "sql")],
            keywords={"types": ["a"]},
            constraints=["constraint"],
        )
        results = validate_metadata_completeness(metadata)

        assert results["required"]
        assert results["schema"]
        assert results["examples"]
        assert results["keywords"]
        assert results["constraints"]

    def test_minimal_metadata(self):
        """Test validating minimal but valid metadata"""
        metadata = DatasetMetadata(
            dataset="users",
            description="User database",
            scope=ScopeMetadata(geographic="Global"),
            schema={"users": TableSchemaMetadata(columns={"id": "INT"})},
        )
        results = validate_metadata_completeness(metadata)

        assert results["required"]
        assert results["schema"]
        assert not results["examples"]
        assert not results["keywords"]


class TestGetMetadataTemplate:
    """Tests for get_metadata_template helper"""

    def test_template_structure(self):
        """Test that template has all required fields"""
        template = get_metadata_template()

        assert template.dataset
        assert template.description
        assert template.scope
        assert template.schema
        assert template.capabilities
        assert template.keywords
        assert template.examples
