"""DynamicPromptBuilder - Build prompts dynamically from metadata

This module provides utilities for constructing prompts dynamically from dataset
metadata without hardcoded templates. This enables flexible, reusable prompt
construction based on actual dataset structure and capabilities.
"""

from __future__ import annotations

from typing import Optional

from axiompy.reasoning.metadata import DatasetMetadata
from axiompy.reasoning.metadata_helpers import format_schema_for_llm


class DynamicPromptBuilder:
    """
    Build prompts dynamically from dataset metadata.

    Instead of hardcoding prompts, this class constructs prompts from
    DatasetMetadata, enabling flexible and reusable prompt generation.

    Key Features:
    - No hardcoded templates
    - All content comes from metadata
    - Configurable for different domains
    - Extensible for custom prompts

    Example:
        builder = DynamicPromptBuilder()

        # Build SQL generation prompt
        prompt = builder.build_sql_generation_prompt(
            question="How many records?",
            metadata=crime_metadata
        )

        # Build planning prompt for multi-dataset routing
        prompt = builder.build_planning_prompt(
            question="Find trends",
            available_datasets={"crime": metadata1, "housing": metadata2}
        )
    """

    @staticmethod
    def build_sql_generation_prompt(
        question: str,
        metadata: DatasetMetadata,
        additional_instructions: Optional[str] = None,
    ) -> dict[str, str]:
        """
        Build a prompt for SQL generation from a natural language question.

        Args:
            question: Natural language question
            metadata: Dataset metadata containing schema and constraints
            additional_instructions: Optional additional instructions for the LLM

        Returns:
            Dictionary with "system" and "user" keys for the LLM

        Example:
            prompt = builder.build_sql_generation_prompt(
                question="Show me crime data from 2023",
                metadata=crime_metadata
            )

            # Returns:
            # {
            #     "system": "You are a SQL expert...",
            #     "user": "Generate SQL for: Show me crime data from 2023\n\nSchema:\n..."
            # }
        """
        schema_text = format_schema_for_llm(metadata)

        system_prompt = (
            "You are an expert SQL query generator. "
            "Your task is to generate accurate, optimized SQL queries based on "
            "natural language questions. "
            "Follow these guidelines:\n"
            "1. Return ONLY the SQL query, no explanations or markdown\n"
            "2. Use the provided schema as the source of truth\n"
            "3. Respect all constraints and limitations\n"
            "4. Optimize for performance when possible\n"
            "5. If a query is not possible, explain why\n\n"
            f"Dataset: {metadata.dataset}\n"
            f"Description: {metadata.description}\n\n"
            f"{schema_text}"
        )

        # Add examples if available
        if metadata.examples:
            system_prompt += "\n\nExample Queries:\n"
            for example in metadata.examples[:5]:  # Limit to 5 examples
                system_prompt += f"\nQ: {example.question}\nSQL: {example.sql}\n"

        # Add constraints if available
        if metadata.constraints:
            system_prompt += "\n\nImportant Constraints:\n"
            for constraint in metadata.constraints:
                system_prompt += f"- {constraint}\n"

        # Add common mistakes if available
        if metadata.common_mistakes:
            system_prompt += "\n\nCommon Mistakes to Avoid:\n"
            for mistake, solution in metadata.common_mistakes.items():
                system_prompt += f"- {mistake}: {solution}\n"

        # Add any additional instructions
        if additional_instructions:
            system_prompt += f"\n\nAdditional Instructions:\n{additional_instructions}\n"

        user_prompt = f"Generate SQL for: {question}"

        return {
            "system": system_prompt,
            "user": user_prompt,
        }

    @staticmethod
    def build_planning_prompt(
        question: str,
        available_datasets: dict[str, DatasetMetadata],
        additional_instructions: Optional[str] = None,
    ) -> dict[str, str]:
        """
        Build a prompt for dataset selection and query planning.

        Used when multiple datasets are available to help the AI agent
        decide which dataset(s) to query and how to approach the question.

        Args:
            question: Natural language question
            available_datasets: Dictionary mapping dataset names to their metadata
            additional_instructions: Optional additional instructions

        Returns:
            Dictionary with "system" and "user" keys for the LLM

        Example:
            prompt = builder.build_planning_prompt(
                question="Compare crime and housing trends",
                available_datasets={
                    "crime": crime_metadata,
                    "housing": housing_metadata
                }
            )
        """
        system_prompt = (
            "You are a query planning expert. "
            "Your task is to analyze a question and determine:\n"
            "1. Which dataset(s) to query\n"
            "2. What approach to take\n"
            "3. Any data transformations needed\n"
            "4. How to combine results if multiple datasets\n\n"
            "Available Datasets:\n"
        )

        # Add metadata for each available dataset
        for dataset_name, dataset_metadata in available_datasets.items():
            system_prompt += f"\n{dataset_name}:\n"
            system_prompt += f"  Description: {dataset_metadata.description}\n"

            if dataset_metadata.capabilities:
                system_prompt += f"  Capabilities: {', '.join(dataset_metadata.capabilities)}\n"

            if dataset_metadata.scope.temporal:
                system_prompt += f"  Time Period: {dataset_metadata.scope.temporal}\n"

            if dataset_metadata.scope.domain:
                system_prompt += f"  Domain: {dataset_metadata.scope.domain}\n"

            if dataset_metadata.keywords:
                all_keywords = []
                for keywords in dataset_metadata.keywords.values():
                    all_keywords.extend(keywords)
                if all_keywords:
                    system_prompt += f"  Keywords: {', '.join(all_keywords[:10])}\n"

        # Add planning guidelines
        system_prompt += (
            "\n\nPlanning Guidelines:\n"
            "1. Select the most relevant dataset(s)\n"
            "2. Explain your reasoning\n"
            "3. Identify potential issues or limitations\n"
            "4. Suggest the best approach\n"
            "5. Format your response as:\n"
            "   Datasets: <comma-separated list>\n"
            "   Approach: <your strategy>\n"
            "   Potential Issues: <any concerns>\n"
            "   Next Steps: <what to do next>"
        )

        if additional_instructions:
            system_prompt += f"\n\nAdditional Instructions:\n{additional_instructions}\n"

        user_prompt = f"Plan how to answer this question: {question}"

        return {
            "system": system_prompt,
            "user": user_prompt,
        }

    @staticmethod
    def build_insight_generation_prompt(
        question: str,
        results: list[dict],
        metadata: DatasetMetadata,
        additional_instructions: Optional[str] = None,
    ) -> dict[str, str]:
        """
        Build a prompt for generating insights from query results.

        Args:
            question: Original natural language question
            results: Query results (list of dictionaries)
            metadata: Dataset metadata for context
            additional_instructions: Optional additional instructions

        Returns:
            Dictionary with "system" and "user" keys for the LLM

        Example:
            prompt = builder.build_insight_generation_prompt(
                question="What are the crime trends?",
                results=[{"year": 2023, "count": 1000}, ...],
                metadata=crime_metadata
            )
        """
        # Format results for context
        results_text = "Query Results:\n"
        for i, row in enumerate(results[:10]):  # Show first 10 rows
            results_text += f"{i + 1}. {row}\n"
        if len(results) > 10:
            results_text += f"... and {len(results) - 10} more rows\n"

        system_prompt = (
            "You are a data analyst expert. "
            "Your task is to analyze query results and generate meaningful insights. "
            "Guidelines:\n"
            "1. Identify patterns and trends\n"
            "2. Highlight significant findings\n"
            "3. Provide context and explanation\n"
            "4. Suggest potential implications\n"
            "5. Be concise but thorough\n\n"
            f"Dataset Context:\n"
            f"- Dataset: {metadata.dataset}\n"
            f"- Description: {metadata.description}\n"
            f"- Scope: {metadata.scope.geographic}"
        )

        if metadata.scope.temporal:
            system_prompt += f", {metadata.scope.temporal}"
        if metadata.scope.domain:
            system_prompt += f", {metadata.scope.domain}"
        system_prompt += "\n"

        if metadata.constraints:
            system_prompt += "\nImportant Constraints:\n"
            for constraint in metadata.constraints:
                system_prompt += f"- {constraint}\n"

        if additional_instructions:
            system_prompt += f"\nAdditional Instructions:\n{additional_instructions}\n"

        user_prompt = (
            f"Original question: {question}\n\n{results_text}\n\nProvide insights and analysis."
        )

        return {
            "system": system_prompt,
            "user": user_prompt,
        }

    @staticmethod
    def build_validation_prompt(
        sql: str,
        metadata: DatasetMetadata,
        additional_instructions: Optional[str] = None,
    ) -> dict[str, str]:
        """
        Build a prompt for SQL validation and improvement.

        Args:
            sql: SQL query to validate
            metadata: Dataset metadata for context
            additional_instructions: Optional additional instructions

        Returns:
            Dictionary with "system" and "user" keys for the LLM

        Example:
            prompt = builder.build_validation_prompt(
                sql="SELECT * FROM crimes WHERE year = 2023",
                metadata=crime_metadata
            )
        """
        schema_text = format_schema_for_llm(metadata)

        system_prompt = (
            "You are a SQL expert. "
            "Your task is to validate SQL queries and identify issues. "
            "Check for:\n"
            "1. Syntax correctness\n"
            "2. Column name validity\n"
            "3. Join correctness (if applicable)\n"
            "4. Performance issues\n"
            "5. Data type mismatches\n\n"
            f"{schema_text}\n\n"
            "Provide your analysis in this format:\n"
            "Valid: Yes/No\n"
            "Issues: <list of issues or 'None'>\n"
            "Suggestions: <improvement suggestions or 'None'>\n"
            "Corrected Query: <corrected query if needed>"
        )

        if additional_instructions:
            system_prompt += f"\n\nAdditional Instructions:\n{additional_instructions}\n"

        user_prompt = f"Validate this SQL query:\n\n{sql}"

        return {
            "system": system_prompt,
            "user": user_prompt,
        }
