"""Metadata Helper Functions - Utilities for working with dataset metadata

This module provides utility functions for extracting information from
DatasetMetadata and formatting it for use by AI agents and other components.
"""

from __future__ import annotations

import re

from axiompy.reasoning.metadata import DatasetMetadata


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
    Extract column references from SQL query using regex.

    This function performs basic regex-based column extraction.
    For complex SQL with subqueries, consider using sqlparse/sqlglot.

    Patterns matched:
    - SELECT column_name
    - WHERE column_name = value
    - ORDER BY column_name
    - GROUP BY column_name
    - JOIN ... ON table.column_name

    Args:
        sql: SQL query string

    Returns:
        Set of column names referenced in the query

    Notes:
        - Uses simple regex patterns; may not handle complex SQL perfectly
        - For production use with LLM-generated SQL, consider sqlparse
        - Returns lowercase column names for consistency

    Example:
        >>> sql = "SELECT name, age FROM users WHERE age > 18"
        >>> cols = extract_columns_from_sql(sql)
        >>> assert "name" in cols
        >>> assert "age" in cols
    """
    columns: set[str] = set()

    # Normalize SQL for easier parsing
    sql_normalized = sql.replace("\n", " ").replace("\t", " ")

    # Pattern 1: SELECT column_name or SELECT table.column_name
    select_pattern = r"SELECT\s+(?:DISTINCT\s+)?(.+?)(?:FROM|WHERE|GROUP|ORDER|LIMIT|;|$)"
    select_match = re.search(select_pattern, sql_normalized, re.IGNORECASE)
    if select_match:
        select_clause = select_match.group(1)
        # Split by comma and extract column names
        for col in select_clause.split(","):
            col = col.strip()
            # Remove aliases (as alias)
            col = re.sub(r"\s+AS\s+\w+", "", col, flags=re.IGNORECASE)
            # Handle table.column notation
            if "." in col:
                col = col.split(".")[-1]
            # Handle functions like COUNT(column)
            col = re.sub(r"\w+\s*\((.+?)\)", r"\1", col)
            # Clean up and add
            col = col.strip().replace("`", "").replace('"', "")
            if col and col != "*":
                columns.add(col.lower())

    # Pattern 2: WHERE, ON, HAVING clauses - extract all identifiers before operators
    # Use original SQL with strings for this, we'll be careful about quotes
    condition_pattern = r"(?:WHERE|ON|HAVING)\s+(.+?)(?:GROUP|ORDER|LIMIT|;|$)"
    for match in re.finditer(condition_pattern, sql_normalized, re.IGNORECASE):
        condition = match.group(1)
        # Extract column references: word before comparison operators
        # First remove quoted strings from condition for regex matching
        condition_no_quotes = re.sub(r"'[^']*'|\"[^\"]*\"", "", condition)
        col_matches = re.findall(
            r"\b([a-zA-Z_]\w*)\s*(?:=|<|>|!=|LIKE|IN|BETWEEN|NOT)",
            condition_no_quotes,
            re.IGNORECASE,
        )
        for col in col_matches:
            col_lower = col.lower()
            if col_lower not in ("and", "or", "not"):
                columns.add(col_lower)

    # Pattern 3: ORDER BY, GROUP BY
    order_group_pattern = r"(?:ORDER\s+BY|GROUP\s+BY)\s+(.+?)(?:ORDER|GROUP|LIMIT|;|$)"
    for match in re.finditer(order_group_pattern, sql_normalized, re.IGNORECASE):
        cols_str = match.group(1)
        # Split by comma for multiple columns
        for col_expr in cols_str.split(","):
            col_expr = col_expr.strip()
            # Remove ASC/DESC and other keywords
            col_expr = re.sub(r"\s+(?:ASC|DESC).*$", "", col_expr, flags=re.IGNORECASE)
            # Get the first identifier (column name)
            # Match identifiers that may be quoted
            id_match = re.match(r"(?:`|\")?([a-zA-Z_]\w*)(?:`|\")?", col_expr)
            if id_match:
                col = id_match.group(1)
                columns.add(col.lower())

    return columns


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
