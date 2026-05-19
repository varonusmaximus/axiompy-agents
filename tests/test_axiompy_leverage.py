"""Tests for axiompy core leverage (sql_engine re-export, secrets, arrow adapter)."""

from unittest.mock import MagicMock

from axiompy.reasoning.metadata import DatasetMetadata, ScopeMetadata, TableSchemaMetadata
from axiompy.reasoning.metadata_helpers import extract_columns_from_sql
from axiompy.reasoning.validators import SQLValidator, ValidationResult
from axiompy.sql_engine import SQLValidator as CoreSQLValidator


class TestSQLValidatorReexport:
    """SQLValidator implementation lives in axiompy.sql_engine."""

    def test_reexport_is_core_class(self):
        assert SQLValidator is CoreSQLValidator

    def test_extract_columns_delegates(self):
        sql = "SELECT id, name FROM users WHERE age > 18"
        cols = extract_columns_from_sql(sql)
        assert cols >= {"id", "name", "age"}

    def test_validation_result_bool(self):
        assert bool(ValidationResult(valid=True, errors=[], warnings=[]))


class TestArrowDatasetService:
    """ArrowDatasetService converts QueryResult to dict rows."""

    def test_query_returns_dict_rows(self):
        from axiompy.reasoning.datasets.arrow_service import ArrowDatasetService

        mock_frame = MagicMock()
        mock_frame.to_dict.return_value = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
        mock_result = MagicMock()
        mock_result.to_pandas.return_value = mock_frame
        mock_client = MagicMock()
        mock_client.query.return_value = mock_result

        metadata = DatasetMetadata(
            dataset="test",
            description="Test",
            scope=ScopeMetadata(geographic="Test"),
            schema={"t": TableSchemaMetadata(columns={"id": "INT", "name": "TEXT"})},
        )
        service = ArrowDatasetService(mock_client, metadata, default_limit=10)
        rows = service.query("SELECT id, name FROM t")

        assert rows == [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
        mock_client.query.assert_called_once()
