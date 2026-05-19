"""SQL validators — re-exported from axiompy core :mod:`axiompy.sql_engine`."""

from __future__ import annotations

from axiompy.sql_engine import SQLValidationResult, SQLValidator

# Backward-compatible alias used across reasoning and tests.
ValidationResult = SQLValidationResult

__all__ = ["SQLValidator", "ValidationResult"]
