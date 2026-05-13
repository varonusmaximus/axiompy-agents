"""JSON Publisher - Output results as JSON.

Implements ReviewPublisher for CI pipeline integration.
"""

import json
import sys
from typing import TextIO

from ...domain.results import ReviewResult


class JSONPublisher:
    """
    Output review results as JSON.

    Useful for CI pipelines that need to parse results.

    Example:
        publisher = JSONPublisher()
        publisher.publish(result, context)
    """

    def __init__(
        self,
        output: TextIO = sys.stdout,
        indent: int = 2,
        include_summary: bool = True,
    ):
        """
        Initialize JSON publisher.

        Args:
            output: Output stream (default: stdout)
            indent: JSON indentation (None for compact)
            include_summary: Whether to include markdown summary
        """
        self.output = output
        self.indent = indent
        self.include_summary = include_summary

    def publish(self, result: ReviewResult, context: dict) -> None:
        """
        Output review result as JSON.

        Args:
            result: Review result to output
            context: Additional context to include
        """
        data = {
            "score": result.score,
            "approved": result.approved,
            "review_event": result.review_event,
            "files_reviewed": result.files_reviewed,
            "rules_applied": result.rules_applied,
            "violations": {
                "total": result.violation_count,
                "critical": result.critical_count,
                "error": result.error_count,
                "warning": result.warning_count,
            },
            "details": [
                {
                    "file": v.file,
                    "line": v.line,
                    "rule_id": v.rule_id,
                    "rule_name": v.rule_name,
                    "severity": v.severity.value,
                    "message": v.message,
                    "suggestion": v.suggestion,
                }
                for v in result.violations
            ],
        }

        if self.include_summary:
            data["summary"] = result.summary

        # Include relevant context
        if "owner" in context:
            data["repository"] = {
                "owner": context.get("owner"),
                "repo": context.get("repo"),
                "pr_number": context.get("pr_number"),
            }

        json_str = json.dumps(data, indent=self.indent)
        self.output.write(json_str)
        self.output.write("\n")
        self.output.flush()
