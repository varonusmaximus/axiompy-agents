"""Error Feedback Generator - Creates helpful feedback for SQL generation retries

This module generates contextual error messages to help AI models correct
their SQL generation mistakes.
"""

from __future__ import annotations

from typing import Any

from axiompy.validators import SQLErrorType


class ErrorFeedbackGenerator:
    """
    Generates helpful feedback messages for SQL generation errors.

    This class creates detailed, actionable feedback that helps AI models
    understand what went wrong and how to fix it on retry attempts.

    Example:
        generator = ErrorFeedbackGenerator()

        feedback = generator.generate(
            error_type=SQLErrorType.SYNTAX_ERROR,
            previous_sql="SELECT * LIMIT 10",
            errors=["LIMIT requires FROM clause"]
        )

        # Returns formatted feedback message with hints
    """

    @staticmethod
    def generate(
        error_type: SQLErrorType,
        previous_sql: str,
        errors: list[str],
        context: dict[str, Any] = None,
    ) -> str:
        """
        Generate feedback message for SQL error.

        Args:
            error_type: Type of SQL error (from SQLErrorType enum)
            previous_sql: The SQL that failed validation
            errors: List of error messages
            context: Optional context (e.g., valid_columns for column errors)

        Returns:
            Formatted feedback message with hints
        """
        context = context or {}

        feedback = "\nPREVIOUS ATTEMPT FAILED:\n"
        feedback += f"Generated SQL: {previous_sql if previous_sql else '(empty)'}\n"
        feedback += f"Error Type: {error_type.value}\n"
        feedback += f"Errors: {', '.join(str(e) for e in errors)}\n"

        # Add type-specific hints
        if error_type == SQLErrorType.EMPTY_SQL:
            feedback += ErrorFeedbackGenerator._empty_sql_hints()
        elif error_type == SQLErrorType.SYNTAX_ERROR:
            feedback += ErrorFeedbackGenerator._syntax_error_hints()
        elif error_type == SQLErrorType.COLUMN_ERROR:
            feedback += ErrorFeedbackGenerator._column_error_hints(context)
        elif error_type == SQLErrorType.DATABASE_ERROR:
            feedback += ErrorFeedbackGenerator._database_error_hints()

        feedback += "\nPlease generate a corrected SQL query."

        return feedback

    @staticmethod
    def _empty_sql_hints() -> str:
        """Generate hints for empty SQL errors."""
        return (
            "\nYou must return a valid SQL query. Requirements:\n"
            "- Must start with SELECT, INSERT, UPDATE, DELETE, or WITH\n"
            "- Do NOT return explanations or empty responses\n"
            "- Return ONLY the SQL query itself\n"
            "- Remove any special tokens like <s> or </s>\n"
        )

    @staticmethod
    def _syntax_error_hints() -> str:
        """Generate hints for syntax errors."""
        return (
            "\nFix the SQL syntax. Common issues:\n"
            "- LIMIT requires FROM clause before it\n"
            "- Check for unmatched parentheses or quotes\n"
            "- Ensure proper keyword order: SELECT ... FROM ... WHERE ... "
            "GROUP BY ... ORDER BY ... LIMIT\n"
        )

    @staticmethod
    def _column_error_hints(context: dict[str, Any]) -> str:
        """Generate hints for column errors."""
        valid_columns = context.get("valid_columns", [])

        hints = "\nColumn validation failed. Check:\n"

        if valid_columns:
            hints += f"- Use only these valid columns: {', '.join(sorted(valid_columns))}\n"

        hints += (
            "- Check spelling and case sensitivity\n"
            "- Don't invent column names\n"
            "- Verify column names match the schema exactly\n"
        )

        return hints

    @staticmethod
    def _database_error_hints() -> str:
        """Generate hints for database validation errors."""
        return (
            "\nDatabase validation failed. Check:\n"
            "- Table names are correct\n"
            "- Column names match schema exactly\n"
            "- JOIN conditions are valid\n"
            "- All columns in SELECT with JOIN are qualified (e.g., table.column)\n"
            "- Use SQLite syntax (not MySQL/PostgreSQL)\n"
            "  * SQLite dates: date('now'), datetime('now', '-1 year')\n"
            "  * NO DATE_SUB, CURDATE, INTERVAL - use SQLite date functions\n"
            "  * NO AUTO_INCREMENT - use AUTOINCREMENT\n"
        )
