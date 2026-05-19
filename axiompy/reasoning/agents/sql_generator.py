"""SQL Generator - Generates and validates SQL with intelligent retry logic

This module handles SQL generation from natural language questions,
with automatic validation and retry on errors.
"""

from __future__ import annotations

from typing import Any, Optional

from axiompy.loggers import LoggerFactory
from axiompy.reasoning.agents.feedback import ErrorFeedbackGenerator
from axiompy.reasoning.agents.validation_pipeline import QueryValidationPipeline
from axiompy.reasoning.client import AIClient
from axiompy.reasoning.metadata import DatasetMetadata
from axiompy.validators import SQLErrorType

logger = LoggerFactory.create_logger(__name__)


class SQLGenerator:
    """
    Generates SQL from natural language with validation and retry.

    This class orchestrates the SQL generation process:
    1. Generate SQL from question using AI
    2. Validate SQL through validation pipeline
    3. If validation fails, retry with error feedback
    4. Return validated SQL or raise error after max retries

    Features:
    - Automatic retry on validation failures
    - Contextual error feedback for AI
    - Validates before execution
    - Supports custom validation rules

    Example:
        generator = SQLGenerator(
            ai_client=client,
            validation_pipeline=pipeline,
            max_retries=2
        )

        sql = generator.generate(
            question="Show me top 10 users",
            metadata=dataset_metadata,
            db_connection=connection
        )

        # Returns validated SQL ready for execution
    """

    def __init__(
        self,
        ai_client: AIClient,
        validation_pipeline: QueryValidationPipeline,
        feedback_generator: Optional[ErrorFeedbackGenerator] = None,
        max_retries: int = 2,
    ):
        """
        Initialize SQL generator.

        Args:
            ai_client: AIClient for SQL generation
            validation_pipeline: Pipeline for validating SQL
            feedback_generator: Generator for error feedback (default: ErrorFeedbackGenerator())
            max_retries: Maximum retry attempts (default: 2, total 3 attempts)
        """
        self.ai_client = ai_client
        self.validator = validation_pipeline
        self.feedback_gen = feedback_generator or ErrorFeedbackGenerator()
        self.max_retries = max_retries

    def generate(
        self, question: str, metadata: DatasetMetadata, db_connection: Optional[Any] = None
    ) -> str:
        """
        Generate validated SQL from natural language question.

        This method attempts to generate valid SQL, retrying with error
        feedback if validation fails.

        Args:
            question: Natural language question
            metadata: Dataset metadata for context
            db_connection: Optional database connection for validation

        Returns:
            Validated SQL query string

        Raises:
            ValueError: If unable to generate valid SQL after all retries
        """
        sql = None
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                # Generate SQL
                if attempt == 0:
                    sql = self._generate_initial(question, metadata)
                else:
                    sql = self._generate_with_feedback(question, metadata, last_error, attempt)

                logger.debug(f"Generated SQL (attempt {attempt + 1}): {sql[:100]}...")

                # Validate SQL
                validation_result = self.validator.validate(sql, metadata, db_connection)

                if validation_result.valid:
                    logger.info(f"✓ SQL validation passed on attempt {attempt + 1}")
                    return sql

                # Validation failed - prepare for retry
                logger.warning(
                    f"Validation failed (attempt {attempt + 1}): {validation_result.error_type}"
                )

                last_error = {
                    "type": validation_result.error_type,
                    "sql": sql,
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings,
                }

            except Exception as e:
                error_msg = f"SQL generation failed: {str(e)}"
                logger.error(f"Generation error (attempt {attempt + 1}): {error_msg}")

                last_error = {
                    "type": SQLErrorType.GENERATION_ERROR,
                    "sql": sql if sql else None,
                    "errors": [error_msg],
                    "warnings": [],
                }

                if attempt >= self.max_retries:
                    raise ValueError(error_msg) from e

        # All retries exhausted
        error_msg = self._build_final_error_message(last_error)
        raise ValueError(error_msg)

    def _generate_initial(self, question: str, metadata: DatasetMetadata) -> str:
        """Generate SQL for first attempt."""
        return self.ai_client.generate_sql_from_question(question=question, metadata=metadata)

    def _generate_with_feedback(
        self, question: str, metadata: DatasetMetadata, error_info: dict[str, Any], attempt: int
    ) -> str:
        """Generate SQL with error feedback from previous attempt."""
        logger.info(f"Retry attempt {attempt}/{self.max_retries} with error feedback")

        # Generate feedback message
        context = {}
        if error_info["type"] == SQLErrorType.COLUMN_ERROR:
            # Add valid columns to context
            schema_columns = set()
            for table_schema in metadata.schema.values():
                schema_columns.update(table_schema.columns.keys())
            context["valid_columns"] = sorted(schema_columns)

        feedback = self.feedback_gen.generate(
            error_type=error_info["type"],
            previous_sql=error_info["sql"],
            errors=error_info["errors"],
            context=context,
        )

        # Enhanced question with feedback
        enhanced_question = f"{question}\n{feedback}"

        # Generate new SQL with feedback
        try:
            sql = self.ai_client.generate_sql_from_question(
                question=enhanced_question, metadata=metadata
            )
            return sql
        except Exception as e:
            raise ValueError(
                f"Failed to generate SQL with feedback (attempt {attempt}): {str(e)}"
            ) from e

    def _build_final_error_message(self, last_error: dict[str, Any]) -> str:
        """Build final error message after all retries exhausted."""
        error_type = last_error.get("type", SQLErrorType.UNKNOWN)
        errors = last_error.get("errors", [])
        error_summary = "; ".join(str(e) for e in errors)

        message = (
            f"Failed to generate valid SQL after {self.max_retries + 1} attempts. "
            f"Last error ({error_type.value}): {error_summary}"
        )

        # Add helpful hints for common issues
        if error_type == SQLErrorType.EMPTY_SQL or "empty" in error_summary.lower():
            message += (
                "\n\nHINT: The AI model is returning empty responses. Try:\n"
                "  1. Use a different model (e.g., make mcp-example MODEL=qwen2.5-coder:1.5b)\n"
                "  2. Check if the model is properly loaded: ollama list\n"
                "  3. Try pulling the model again: ollama pull <model-name>\n"
                "  4. Use a more reliable model: mistral, codellama, or deepseek-coder"
            )

        return message
