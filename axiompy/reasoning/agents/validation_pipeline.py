"""Validation Pipeline - Composable SQL validation using filter chain pattern

This module provides a pipeline for validating SQL queries through a chain of validators.
Validators are composed from axiompy.validators, making them reusable across the codebase.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

from axiompy.validators import (
    EmptySQLValidator,
    SQLColumnValidator,
    SQLDatabaseValidator,
    SQLErrorType,
    SQLSyntaxValidator,
    ValidationContext,
    Validator,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Result of validation pipeline.

    This is the external-facing result that components like SQLGenerator expect.
    """

    valid: bool
    errors: list[str]
    warnings: list[str]
    error_type: Optional[SQLErrorType] = None

    def __bool__(self):
        """Allow using validation result as boolean."""
        return self.valid


class QueryValidationPipeline:
    """
    Composable validation pipeline using filter chain pattern.

    This pipeline chains validators together, passing context through each.
    Validators from axiompy.validators can be added, removed, or reordered
    without modifying the pipeline itself.

    The pipeline follows the chain-of-responsibility pattern where each
    validator processes the context and passes it to the next validator.

    Features:
    - Composable: Add/remove validators dynamically
    - Reusable: Validators from axiompy.validators work everywhere
    - Extensible: Easy to add custom validators
    - Testable: Test validators independently

    Example:
        # Default pipeline
        pipeline = QueryValidationPipeline.default()
        result = pipeline.validate(sql, metadata, db_connection)

        # Custom pipeline
        pipeline = QueryValidationPipeline([
            EmptySQLValidator(),
            SQLSyntaxValidator(),
            MyCustomValidator(),
        ])

        # Add validator at runtime
        pipeline.add_validator(SQLDatabaseValidator())

        # Remove validator
        pipeline.remove_validator(SQLDatabaseValidator)
    """

    def __init__(self, validators: List[Validator]):
        """
        Initialize pipeline with list of validators.

        Args:
            validators: Ordered list of validators to apply
        """
        self.validators = validators

    @classmethod
    def default(
        cls,
        enable_db_validation: bool = True,
        db_dialect: str = "sqlite",
        strict_columns: bool = False,
    ):
        """
        Create default SQL validation pipeline.

        Default validators (in order):
        1. EmptySQLValidator - Check SQL is not empty
        2. SQLSyntaxValidator - Check syntax using sqlparse
        3. SQLColumnValidator - Check columns exist in schema
        4. SQLDatabaseValidator - Validate with EXPLAIN (optional)

        Args:
            enable_db_validation: Whether to include database validation (default: True)
            db_dialect: Database dialect for validation (default: "sqlite")
                       Options: "sqlite", "postgres", "mysql"
            strict_columns: If True, fail on any unrecognized column (default: False)

        Returns:
            QueryValidationPipeline with default validators

        Example:
            # SQLite (default)
            pipeline = QueryValidationPipeline.default()

            # PostgreSQL
            pipeline = QueryValidationPipeline.default(db_dialect="postgres")

            # Strict column validation
            pipeline = QueryValidationPipeline.default(strict_columns=True)

            # Without database validation
            pipeline = QueryValidationPipeline.default(enable_db_validation=False)
        """
        validators = [
            EmptySQLValidator(),
            SQLSyntaxValidator(use_parser=True),
            SQLColumnValidator(strict=strict_columns),
        ]

        if enable_db_validation:
            validators.append(SQLDatabaseValidator(dialect=db_dialect))

        return cls(validators)

    def validate(
        self, sql: str, metadata: Any = None, db_connection: Optional[Any] = None
    ) -> ValidationResult:
        """
        Run SQL through validation chain.

        Each validator in the chain processes the context and adds
        errors/warnings. The chain continues until all validators
        have run or a critical error stops it.

        Args:
            sql: SQL query to validate
            metadata: Dataset metadata with schema information
            db_connection: Optional database connection for dry-run

        Returns:
            ValidationResult with validation status and details

        Example:
            result = pipeline.validate(
                sql="SELECT name FROM users",
                metadata=dataset_metadata,
                db_connection=connection
            )

            if not result.valid:
                print(f"Errors: {result.errors}")
                print(f"Type: {result.error_type}")
        """
        logger.debug(f"Starting validation pipeline with {len(self.validators)} validators")

        # Create context
        context = ValidationContext(sql=sql, metadata=metadata, db_connection=db_connection)

        # Run through validator chain
        for i, validator in enumerate(self.validators):
            validator_name = validator.__class__.__name__
            logger.debug(f"Running validator {i + 1}/{len(self.validators)}: {validator_name}")

            context = validator.validate(context)

            # Short-circuit on critical errors
            if self._has_critical_errors(context):
                logger.debug("Critical error detected, stopping validation chain")
                break

        # Build result
        error_type = self._determine_error_type(context)

        result = ValidationResult(
            valid=len(context.errors) == 0,
            errors=context.errors,
            warnings=context.warnings,
            error_type=error_type,
        )

        if result.valid:
            logger.debug("✓ Validation passed")
        else:
            logger.debug(f"✗ Validation failed: {error_type} - {len(context.errors)} error(s)")

        return result

    def add_validator(self, validator: Validator, position: Optional[int] = None):
        """
        Add validator to chain.

        Args:
            validator: Validator instance to add
            position: Optional position to insert at (default: end)

        Example:
            pipeline.add_validator(MyCustomValidator())
            pipeline.add_validator(SQLDatabaseValidator(), position=0)  # Add first
        """
        if position is None:
            self.validators.append(validator)
            logger.debug(f"Added {validator.__class__.__name__} to end of pipeline")
        else:
            self.validators.insert(position, validator)
            logger.debug(f"Inserted {validator.__class__.__name__} at position {position}")

    def remove_validator(self, validator_type: type):
        """
        Remove all validators of given type from chain.

        Args:
            validator_type: Class of validator to remove

        Example:
            pipeline.remove_validator(SQLDatabaseValidator)
        """
        original_count = len(self.validators)
        self.validators = [v for v in self.validators if not isinstance(v, validator_type)]
        removed = original_count - len(self.validators)
        if removed > 0:
            logger.debug(f"Removed {removed} {validator_type.__name__} validator(s)")

    def _has_critical_errors(self, context: ValidationContext) -> bool:
        """
        Check if context has critical errors that should stop validation.

        Critical errors are those where continuing validation makes no sense.
        For example, if SQL is empty, there's no point checking syntax.

        Args:
            context: Current validation context

        Returns:
            True if validation should stop
        """
        if not context.errors:
            return False

        # Empty SQL is critical - no point validating further
        error_text = " ".join(context.errors).lower()
        return "empty" in error_text

    def _determine_error_type(self, context: ValidationContext) -> Optional[SQLErrorType]:
        """
        Determine primary error type from context errors.

        This helps provide better error feedback for retry attempts.

        Args:
            context: Validation context with errors

        Returns:
            SQLErrorType enum or None if no errors
        """
        if not context.errors:
            return None

        error_text = " ".join(context.errors).lower()

        # Check in priority order and return immediately
        if "empty" in error_text:
            return SQLErrorType.EMPTY_SQL
        elif "syntax" in error_text:
            return SQLErrorType.SYNTAX_ERROR
        elif "column" in error_text:
            return SQLErrorType.COLUMN_ERROR
        elif "database" in error_text or "no such table" in error_text:
            return SQLErrorType.DATABASE_ERROR
        else:
            return SQLErrorType.UNKNOWN
