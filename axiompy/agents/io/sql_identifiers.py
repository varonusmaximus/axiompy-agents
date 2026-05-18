"""Validate SQL table/column names before identifier interpolation."""

from __future__ import annotations

import re

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_sql_identifier(name: str, field: str = "identifier") -> str:
    """
    Ensure a value is safe to use as a SQL identifier (table/column name).

    Args:
        name: Proposed identifier
        field: Label for error messages

    Returns:
        The same string if valid

    Raises:
        ValueError: If the name is not a simple SQL identifier
    """
    if not name or not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid SQL {field}: {name!r}")
    return name


def join_sql_identifiers(names: list[str], field: str = "identifier") -> str:
    """Return comma-separated validated identifiers."""
    return ", ".join(validate_sql_identifier(n, field) for n in names)
