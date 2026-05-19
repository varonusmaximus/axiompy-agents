"""QueryAgent - AI-powered agent for intelligent query routing and execution

This module provides an agent that can intelligently route questions to appropriate
datasets and execute queries, with AI-powered planning, SQL generation, validation,
and insight generation.

REFACTORED VERSION - Uses composition with separate validation and generation components.
"""

from __future__ import annotations

from typing import Any, Optional

from axiompy.io.http import HTTPConnectionError, HTTPRequestError
from axiompy.loggers import LoggerFactory
from axiompy.reasoning.agents.feedback import ErrorFeedbackGenerator
from axiompy.reasoning.agents.sql_generator import SQLGenerator
from axiompy.reasoning.agents.validation_pipeline import QueryValidationPipeline
from axiompy.reasoning.base import BaseDatasetService
from axiompy.reasoning.client import AIClient
from axiompy.reasoning.metadata import DatasetMetadata
from axiompy.reasoning.metadata_helpers import match_keywords
from axiompy.reasoning.prompts import DynamicPromptBuilder

logger = LoggerFactory.create_logger(__name__)

_QUERY_EXECUTION_ERRORS = (
    ValueError,
    ConnectionError,
    OSError,
    RuntimeError,
    HTTPRequestError,
    HTTPConnectionError,
)
_LLM_CALL_ERRORS = (ValueError, TypeError, HTTPRequestError, HTTPConnectionError)


class QueryAgent:
    """
    AI-powered agent for intelligent query routing and execution.

    The agent follows this flow:
    1. Planning - Determine which dataset(s) to query
    2. SQL Generation - Generate SQL from the question (with validation & retry)
    3. Execution - Execute the query via the dataset service
    4. Insights - Generate AI insights from the results

    Features:
    - Automatic dataset selection
    - Natural language to SQL conversion
    - SQL validation before execution (syntax, columns, database dry-run)
    - Error recovery with intelligent retry logic
    - Insight generation from results
    - Metadata-driven decision making

    Example:
        # Setup
        agent = QueryAgent(
            ai_client=ReasoningFactory.create(ReasoningProvider.OLLAMA),
            datasets={
                "crime": crime_service,
                "housing": housing_service
            }
        )

        # Execute query
        result = agent.execute_query(
            "How many homicides in 2023?"
        )

        # Result includes: results, sql, insights, metadata
        print(result["insights"])
    """

    def __init__(
        self,
        ai_client: AIClient,
        datasets: dict[str, BaseDatasetService],
        enable_planning: bool = True,
        enable_insights: bool = True,
        enable_db_validation: bool = True,
        db_dialect: str = "sqlite",
        max_retries: int = 2,
    ):
        """
        Initialize QueryAgent.

        Args:
            ai_client: AIClient instance for LLM calls
            datasets: Mapping of dataset name to BaseDatasetService instance
            enable_planning: Whether to do dataset planning (default: True)
            enable_insights: Whether to generate insights (default: True)
            enable_db_validation: Whether to validate SQL with database EXPLAIN (default: True)
            db_dialect: Database dialect for validation (default: "sqlite")
                       Options: "sqlite", "postgres", "mysql"
            max_retries: Maximum retries on SQL generation failure (default: 2)

        Raises:
            ValueError: If no datasets provided

        Example:
            # SQLite (default)
            agent = QueryAgent(ai_client, datasets)

            # PostgreSQL
            agent = QueryAgent(ai_client, datasets, db_dialect="postgres")

            # MySQL
            agent = QueryAgent(ai_client, datasets, db_dialect="mysql")
        """
        if not datasets:
            raise ValueError("At least one dataset must be provided")

        self.ai_client = ai_client
        self.datasets = datasets
        self.enable_planning = enable_planning
        self.enable_insights = enable_insights
        self.prompt_builder = DynamicPromptBuilder()

        # Initialize validation and generation components
        self.validation_pipeline = QueryValidationPipeline.default(
            enable_db_validation=enable_db_validation, db_dialect=db_dialect
        )
        self.feedback_generator = ErrorFeedbackGenerator()
        self.sql_generator = SQLGenerator(
            ai_client=ai_client,
            validation_pipeline=self.validation_pipeline,
            feedback_generator=self.feedback_generator,
            max_retries=max_retries,
        )

        # Cache metadata for efficiency
        self._metadata_cache: dict[str, DatasetMetadata] = {}
        for name, service in self.datasets.items():
            self._metadata_cache[name] = service.get_metadata()

    def execute_query(self, question: str) -> dict[str, Any]:
        """
        Execute a natural language query.

        Args:
            question: Natural language question

        Returns:
            Dictionary with keys:
            - results: Query results (list of dicts)
            - sql: Generated SQL query
            - dataset: Dataset name used
            - insights: AI-generated insights (if enabled)
            - metadata: Selected dataset metadata

        Raises:
            ValueError: If unable to generate valid SQL after retries
            ConnectionError: If dataset service fails
        """
        # Step 1: Planning - Select appropriate dataset
        selected_dataset = self._plan_query(question)

        if not selected_dataset:
            raise ValueError(f"Could not determine appropriate dataset for question: {question}")

        dataset_service = self.datasets[selected_dataset]
        dataset_metadata = self._metadata_cache[selected_dataset]

        # Step 2: SQL Generation with Validation and Retry
        # This is now handled by SQLGenerator component
        db_connection = self._get_db_connection(dataset_service)

        sql = self.sql_generator.generate(
            question=question, metadata=dataset_metadata, db_connection=db_connection
        )

        # Step 3: Execution - Execute the validated query
        try:
            results = dataset_service.query(sql, limit=1000)
        except _QUERY_EXECUTION_ERRORS as e:
            raise ConnectionError(
                f"Failed to execute query on {selected_dataset}: {e}"
            ) from e

        # Step 4: Insights - Generate AI insights if enabled
        insights = None
        if self.enable_insights and results:
            insights = self._generate_insights(question, results, dataset_metadata)

        return {
            "results": results,
            "sql": sql,
            "dataset": selected_dataset,
            "insights": insights,
            "metadata": dataset_metadata,
        }

    def execute_multi_dataset_query(self, question: str, max_datasets: int = 3) -> dict[str, Any]:
        """
        Execute a query potentially across multiple datasets.

        This method is useful for questions that may require data from
        multiple sources, though execution will still happen on one dataset
        unless additional intelligence is added in Phase 2+.

        Args:
            question: Natural language question
            max_datasets: Maximum number of datasets to consider

        Returns:
            Dictionary with results, sql, datasets, insights
        """
        # For now, route to the best single dataset
        # Phase 2+ can enhance to support true multi-dataset queries
        return self.execute_query(question)

    def _plan_query(self, question: str) -> Optional[str]:
        """
        Determine which dataset is most appropriate for the question.

        Uses keyword matching and AI planning to select the best dataset.

        Args:
            question: Natural language question

        Returns:
            Selected dataset name or None if unable to determine
        """
        if len(self.datasets) == 1:
            # Only one dataset, no planning needed
            return list(self.datasets.keys())[0]

        if not self.enable_planning:
            # Planning disabled, return first dataset
            return list(self.datasets.keys())[0]

        # Try keyword matching first
        best_match = None
        best_score = 0.0

        for dataset_name, metadata in self._metadata_cache.items():
            scores = match_keywords(question, metadata)
            avg_score = sum(scores.values()) / len(scores) if scores else 0

            if avg_score > best_score:
                best_score = avg_score
                best_match = dataset_name

        if best_match and best_score > 0.3:
            return best_match

        # If keyword matching doesn't find a match, use AI planning
        try:
            planning_prompt = self.prompt_builder.build_planning_prompt(
                question=question,
                available_datasets=self._metadata_cache,
            )

            planning_response = self.ai_client.generate_completion(
                planning_prompt["user"],
                temperature=0.3,  # Lower temperature for planning
                use_cache=False,
            )

            # Extract dataset name from response (simple extraction)
            for dataset_name in self.datasets.keys():
                if dataset_name.lower() in planning_response.lower():
                    return dataset_name

            # If no dataset mentioned, return first one
            return list(self.datasets.keys())[0]

        except _LLM_CALL_ERRORS as e:
            logger.warning(f"AI planning failed, using keyword fallback: {e}")
            return best_match or list(self.datasets.keys())[0]

    def _generate_insights(
        self,
        question: str,
        results: list[dict[str, Any]],
        metadata: DatasetMetadata,
    ) -> str:
        """
        Generate AI insights from query results.

        Args:
            question: Original natural language question
            results: Query results
            metadata: Dataset metadata for context

        Returns:
            Generated insights text
        """
        try:
            prompt = self.prompt_builder.build_insight_generation_prompt(
                question=question,
                results=results,
                metadata=metadata,
            )

            insights = self.ai_client.generate_completion(
                prompt["user"],
                temperature=0.7,  # Higher temperature for creativity
                use_cache=False,
            )
            return insights
        except _LLM_CALL_ERRORS as e:
            logger.warning(f"Insight generation failed: {e}")
            return None

    def get_dataset_names(self) -> list[str]:
        """Get list of available dataset names."""
        return list(self.datasets.keys())

    def get_dataset_capabilities(self) -> dict[str, list[str]]:
        """Get capabilities for each dataset."""
        return {name: service.get_capabilities() for name, service in self.datasets.items()}

    def get_datasets_metadata(self) -> dict[str, DatasetMetadata]:
        """Get metadata for all datasets."""
        return self._metadata_cache.copy()

    def refresh_metadata(self) -> None:
        """Refresh cached metadata from all datasets."""
        for name, service in self.datasets.items():
            self._metadata_cache[name] = service.get_metadata()

    def _get_db_connection(self, dataset_service: BaseDatasetService) -> Optional[Any]:
        """
        Try to get database connection from dataset service.

        This is a best-effort attempt to access the underlying database
        connection for validation purposes. Returns None if not accessible.

        Args:
            dataset_service: Dataset service instance

        Returns:
            Database connection or None if not available
        """
        logger.debug("Attempting to get database connection for validation")

        # Try to get db attribute (common pattern)
        if hasattr(dataset_service, "db"):
            db = dataset_service.db
            # Check if it has _connection attribute (Database class from axiompy.io.database)
            if hasattr(db, "_connection"):
                logger.debug("Found database connection via db._connection")
                return db._connection
            # Check if it has a connection attribute
            if hasattr(db, "connection"):
                logger.debug("Found database connection via db.connection")
                return db.connection
            # Or if it is the connection itself
            logger.debug("Using db object as connection")
            return db

        # Try to get connection attribute directly
        if hasattr(dataset_service, "connection"):
            logger.debug("Found database connection via service.connection")
            return dataset_service.connection

        # Try _connection attribute directly
        if hasattr(dataset_service, "_connection"):
            logger.debug("Found database connection via service._connection")
            return dataset_service._connection

        # No connection found
        logger.warning("Could not find database connection for validation")
        return None
