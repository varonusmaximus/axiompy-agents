"""Dataset Metadata Schema - Type-safe definitions for self-describing tools

This module provides dataclasses for structured metadata that enables AI agents
to understand dataset capabilities, structure, and constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ScopeMetadata:
    """
    Defines the scope (geographic, temporal, domain) of a dataset.

    Attributes:
        geographic: Geographic scope (e.g., "Chicago, IL", "United States")
        temporal: Temporal range or granularity (e.g., "2000-2024", "Daily")
        domain: Business domain (e.g., "Public Safety", "Real Estate")
        important: Important scope limitations or constraints
    """

    geographic: str
    temporal: Optional[str] = None
    domain: Optional[str] = None
    important: Optional[str] = None


@dataclass
class TableSchemaMetadata:
    """
    Schema information for a single table or collection.

    Attributes:
        columns: Mapping of column name to type (e.g., {"id": "INTEGER PRIMARY KEY"})
        description: Human-readable table description
        row_count: Approximate number of rows (for query planning)
        indexes: List of indexed columns for performance hints
    """

    columns: dict[str, str]
    description: Optional[str] = None
    row_count: Optional[int] = None
    indexes: Optional[list[str]] = None


@dataclass
class ExampleMetadata:
    """
    Example queries and expected results for AI learning.

    Attributes:
        question: Natural language question
        sql: SQL query that answers the question
        expected_results: Sample results or description
        keywords: Keywords associated with this example
    """

    question: str
    sql: str
    expected_results: Optional[str] = None
    keywords: Optional[list[str]] = None


@dataclass
class DatasetMetadata:
    """
    Complete metadata for a self-describing dataset service.

    This metadata enables AI agents to:
    - Understand what data is available
    - Generate appropriate queries
    - Validate generated SQL
    - Provide intelligent suggestions
    - Understand scope limitations

    Attributes:
        dataset: Dataset identifier (e.g., "crime_incidents")
        description: Human-readable dataset description
        scope: Geographic, temporal, and domain scope
        schema: Mapping of table name to schema metadata
        capabilities: List of supported operations
        keywords: Domain keywords for natural language matching
        examples: Example queries for few-shot learning
        constraints: Important constraints and limitations
        common_mistakes: Common errors and how to avoid them

    Example:
        metadata = DatasetMetadata(
            dataset="crime_incidents",
            description="Chicago crime incident database",
            scope=ScopeMetadata(
                geographic="Chicago, IL",
                temporal="2000-2024",
                domain="Public Safety"
            ),
            schema={
                "incidents": TableSchemaMetadata(
                    columns={
                        "id": "INTEGER PRIMARY KEY",
                        "date_reported": "DATE",
                        "crime_type": "TEXT",
                        "latitude": "FLOAT",
                        "longitude": "FLOAT"
                    },
                    row_count=7000000,
                    indexes=["date_reported", "crime_type"]
                )
            },
            capabilities=[
                "search_by_date",
                "filter_by_crime_type",
                "geographic_queries",
                "temporal_analysis"
            ],
            keywords={
                "crime_types": ["homicide", "robbery", "theft", "assault"],
                "areas": ["downtown", "south side", "north side"]
            },
            examples=[
                ExampleMetadata(
                    question="How many homicides were there in 2023?",
                    sql="SELECT COUNT(*) FROM incidents WHERE "
                    "YEAR(date_reported)=2023 AND crime_type='HOMICIDE'",
                    keywords=["homicide", "count", "2023"]
                )
            ],
            constraints=[
                "Data is updated monthly",
                "Only includes reported incidents",
                "Geographic coordinates are approximate"
            ]
        )
    """

    dataset: str
    description: str
    scope: ScopeMetadata
    schema: dict[str, TableSchemaMetadata]
    capabilities: Optional[list[str]] = None
    keywords: Optional[dict[str, list[str]]] = None
    examples: Optional[list[ExampleMetadata]] = None
    constraints: Optional[list[str]] = None
    common_mistakes: Optional[dict[str, str]] = None
