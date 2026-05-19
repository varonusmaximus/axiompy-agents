"""Metadata Helper Functions - Utilities for working with dataset metadata

This module provides utility functions for extracting information from
DatasetMetadata and formatting it for use by AI agents and other components.
"""

from __future__ import annotations

from axiompy.reasoning.metadata import DatasetMetadata
from axiompy.reasoning.validators import SQLValidator


def extract_all_columns(metadata: DatasetMetadata) -> set[str]:
    """
    Extract all column names from dataset metadata.

    Iterates through all tables in the metadata schema and collects
    all column names into a single set.

    Args:
        metadata: DatasetMetadata containing schema information

    Returns:
        Set of all column names across all tables

    Example:
        >>> columns = extract_all_columns(crime_metadata)
        >>> assert "date_reported" in columns
        >>> assert "latitude" in columns
    """
    all_columns: set[str] = set()

    for table_metadata in metadata.schema.values():
        all_columns.update(table_metadata.columns.keys())

    return all_columns


def extract_columns_from_sql(sql: str) -> set[str]:
    """
    Extract column references from SQL (delegates to :class:`SQLValidator`).

    Args:
        sql: SQL query string

    Returns:
        Set of lowercase column names referenced in the query
    """
    return SQLValidator.extract_columns(sql)


def format_schema_for_llm(metadata: DatasetMetadata) -> str:
    """
    Format dataset schema as human-readable text for LLM consumption.

    Produces a formatted string describing tables, columns, and their types.
    Suitable for inclusion in LLM prompts for SQL generation.

    Args:
        metadata: DatasetMetadata containing schema information

    Returns:
        Formatted schema string

    Example output:
        Dataset: crime_incidents
        Description: Chicago crime data 2000-2024

        Tables:
        - incidents (7,000,000 rows)
          Columns:
          * id: INTEGER PRIMARY KEY
          * date_reported: DATE
          * crime_type: TEXT
          * latitude: FLOAT
          * longitude: FLOAT
    """
    lines = [
        f"Dataset: {metadata.dataset}",
        f"Description: {metadata.description}",
        "",
        "Schema:",
    ]

    for table_name, table_meta in metadata.schema.items():
        # Table header
        row_count = f" ({table_meta.row_count:,} rows)" if table_meta.row_count else ""
        lines.append("")
        lines.append(f"Table: {table_name}{row_count}")

        if table_meta.description:
            lines.append(f"  Description: {table_meta.description}")

        # Columns
        lines.append("  Columns:")
        for col_name, col_type in table_meta.columns.items():
            lines.append(f"    - {col_name}: {col_type}")

        # Indexes if present
        if table_meta.indexes:
            lines.append(f"  Indexes: {', '.join(table_meta.indexes)}")

    # Scope information
    lines.append("")
    lines.append("Scope:")
    lines.append(f"  Geographic: {metadata.scope.geographic}")
    if metadata.scope.temporal:
        lines.append(f"  Temporal: {metadata.scope.temporal}")
    if metadata.scope.domain:
        lines.append(f"  Domain: {metadata.scope.domain}")
    if metadata.scope.important:
        lines.append(f"  Important: {metadata.scope.important}")

    # Constraints
    if metadata.constraints:
        lines.append("")
        lines.append("Constraints:")
        for constraint in metadata.constraints:
            lines.append(f"  - {constraint}")

    return "\n".join(lines)


def match_keywords(question: str, metadata: DatasetMetadata) -> dict[str, float]:
    """
    Match question keywords to metadata keywords with scoring.

    Performs simple keyword matching to estimate relevance of a dataset
    to a natural language question.

    Args:
        question: Natural language question
        metadata: DatasetMetadata containing keywords

    Returns:
        Dictionary mapping keyword categories to match scores (0.0-1.0)
        Higher scores indicate more relevant keywords found

    Example:
        >>> question = "How many homicides in downtown?"
        >>> scores = match_keywords(question, crime_metadata)
        >>> assert scores["crime_types"] > 0  # Found "homicides"
        >>> assert scores["areas"] > 0        # Found "downtown"
    """
    if not metadata.keywords:
        return {}

    question_lower = question.lower()
    scores: dict[str, float] = {}

    for category, keywords in metadata.keywords.items():
        matches = 0
        for keyword in keywords:
            if keyword.lower() in question_lower:
                matches += 1

        # Score: fraction of keywords that matched
        score = matches / len(keywords) if keywords else 0.0
        if matches > 0:
            scores[category] = min(score, 1.0)

    return scores


def validate_metadata_completeness(metadata: DatasetMetadata) -> dict[str, bool]:
    """
    Validate metadata has required and recommended fields.

    Checks that metadata is sufficiently complete for AI agent use.

    Args:
        metadata: DatasetMetadata to validate

    Returns:
        Dictionary with validation results:
        - "required": True if all required fields present
        - "schema": True if schema is non-empty
        - "examples": True if examples are present
        - "keywords": True if keywords are present
        - "constraints": True if constraints documented

    Example:
        >>> results = validate_metadata_completeness(metadata)
        >>> assert results["required"]
        >>> if not results["examples"]:
        ...     logger.warning("No examples provided for AI learning")
    """
    results = {
        "required": bool(
            metadata.dataset
            and metadata.description
            and metadata.scope
            and metadata.scope.geographic
            and metadata.schema
        ),
        "schema": bool(metadata.schema),
        "examples": bool(metadata.examples),
        "keywords": bool(metadata.keywords),
        "constraints": bool(metadata.constraints),
    }

    return results


def get_metadata_template() -> DatasetMetadata:
    """
    Get a template DatasetMetadata for creating new services.

    Useful for developers building new dataset services.

    Returns:
        Template DatasetMetadata with placeholder values

    Example:
        >>> template = get_metadata_template()
        >>> template.dataset = "my_dataset"
        >>> template.description = "My custom dataset"
        >>> # ... fill in other fields ...
    """
    from axiompy.reasoning.metadata import (
        DatasetMetadata,
        ExampleMetadata,
        ScopeMetadata,
        TableSchemaMetadata,
    )

    return DatasetMetadata(
        dataset="<dataset_name>",
        description="<dataset_description>",
        scope=ScopeMetadata(
            geographic="<geographic_scope>",
            temporal="<temporal_scope>",
            domain="<domain>",
        ),
        schema={
            "<table_name>": TableSchemaMetadata(
                columns={
                    "<column_name>": "<column_type>",
                },
                description="<table_description>",
            )
        },
        capabilities=["<capability_1>", "<capability_2>"],
        keywords={
            "<category>": ["<keyword_1>", "<keyword_2>"],
        },
        examples=[
            ExampleMetadata(
                question="<example_question>",
                sql="<example_sql>",
            )
        ],
        constraints=["<constraint_1>", "<constraint_2>"],
    )
