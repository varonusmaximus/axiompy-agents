"""BaseDatasetService - Abstract interface for AI-powered dataset services

This module provides the abstract base class for dataset services that can be
queried by AI agents and other applications.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from axiompy.reasoning.metadata import DatasetMetadata


class BaseDatasetService(ABC):
    """
    Abstract interface for dataset services.

    Provides a standard interface for querying datasets, schema introspection,
    and capability discovery. This enables AI agents to intelligently route
    questions to the correct dataset and generate appropriate queries.

    Key Features:
    - Backend-agnostic: Works with databases, APIs, files, search engines
    - Self-describing: Services expose rich metadata for AI reasoning
    - Composable: Services compose repositories; no forced inheritance

    Example:
        class CrimeService(BaseDatasetService):
            dataset_name = "crime_incidents"
            description = "Chicago crime incidents database"

            def __init__(self, repository):
                self.repository = repository

            def query(self, sql: str, limit: int = 100):
                return self.repository.execute(sql, limit=limit)

            def get_capabilities(self):
                return ["search_by_date", "filter_by_type", "analyze_trends"]

            def get_metadata(self):
                return DatasetMetadata(...)
    """

    # Class attributes for service identification
    dataset_name: str = "unknown"
    description: str = "No description provided"

    @abstractmethod
    def query(self, sql: str, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Execute a query and return results.

        Args:
            sql: SQL query string (dialect depends on backend)
            limit: Maximum number of rows to return (optional)

        Returns:
            List of dictionaries representing query results

        Raises:
            ValueError: If query is invalid or limit is negative
            Exception: Backend-specific exceptions (SQLError, APIError, etc.)
        """
        ...

    @abstractmethod
    def get_capabilities(self) -> list[str]:
        """
        Get list of capabilities for AI agent routing.

        Returns:
            List of capability strings (e.g., ["search_by_date", "spatial_queries"])

        Notes:
            - Used by AI agents to determine if this service can handle a question
            - Should be descriptive and discoverable
            - Examples: "time_series_analysis", "geographic_filtering", "aggregations"
        """
        ...

    @abstractmethod
    def get_metadata(self) -> DatasetMetadata:
        """
        Get rich metadata for AI reasoning and query generation.

        Returns:
            DatasetMetadata containing:
            - Dataset name and description
            - Schema information (tables, columns, types)
            - Scope (geographic, temporal, domain)
            - Example queries and constraints
            - Keywords for natural language matching

        Notes:
            - Called by AI agents during query planning
            - Should be comprehensive but not excessive
            - Metadata is cached by agents to minimize calls

        Example:
            return DatasetMetadata(
                dataset="crime_incidents",
                description="Chicago crime data 2000-2024",
                scope=ScopeMetadata(
                    geographic="Chicago, IL",
                    temporal="2000-2024"
                ),
                schema={
                    "incidents": TableSchemaMetadata(
                        columns={
                            "id": "INTEGER PRIMARY KEY",
                            "date_reported": "DATE",
                            "crime_type": "TEXT",
                            "latitude": "FLOAT",
                            "longitude": "FLOAT"
                        }
                    )
                }
            )
        """
        ...
