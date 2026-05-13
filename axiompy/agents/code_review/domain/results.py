"""Domain Results - Review result types.

Defines the output of code review:
- Violation: A single rule violation
- ReviewComment: A comment to post on a line
- ReviewResult: Complete review result with score and summary
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .rules import ParsedRule, RuleSeverity


class ReviewSeverity(Enum):
    """Severity of a review comment."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @classmethod
    def from_rule_severity(cls, severity: RuleSeverity) -> "ReviewSeverity":
        """Convert RuleSeverity to ReviewSeverity."""
        mapping = {
            RuleSeverity.INFO: cls.INFO,
            RuleSeverity.WARNING: cls.WARNING,
            RuleSeverity.ERROR: cls.ERROR,
            RuleSeverity.CRITICAL: cls.CRITICAL,
        }
        return mapping.get(severity, cls.WARNING)


@dataclass
class Violation:
    """
    A single rule violation found during review.

    Attributes:
        file: Path to the file
        line: Line number (1-indexed, 0 for file-level)
        rule_id: ID of the violated rule
        rule_name: Name of the violated rule
        message: Description of the violation
        severity: How serious the violation is
        suggestion: How to fix it
        code_snippet: The offending code
    """

    file: str
    line: int
    rule_id: str
    rule_name: str
    message: str
    severity: ReviewSeverity
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None

    @property
    def is_critical(self) -> bool:
        """Check if violation is critical."""
        return self.severity == ReviewSeverity.CRITICAL

    @property
    def is_error(self) -> bool:
        """Check if violation is error or worse."""
        return self.severity in (ReviewSeverity.ERROR, ReviewSeverity.CRITICAL)

    def to_comment(self) -> "ReviewComment":
        """Convert violation to a review comment."""
        body_parts = [f"**{self.rule_name}** ({self.severity.value.upper()})"]
        body_parts.append("")
        body_parts.append(self.message)

        if self.suggestion:
            body_parts.append("")
            body_parts.append(f"💡 **Suggestion:** {self.suggestion}")

        return ReviewComment(
            path=self.file,
            line=self.line,
            body="\n".join(body_parts),
            severity=self.severity,
        )


@dataclass
class ReviewComment:
    """
    A comment to post on a specific line.

    This is the format used for posting to GitHub PR reviews.

    Attributes:
        path: File path
        line: Line number
        body: Comment body (markdown)
        severity: Comment severity
    """

    path: str
    line: int
    body: str
    severity: ReviewSeverity = ReviewSeverity.WARNING


@dataclass
class ReviewResult:
    """
    Complete result of a code review.

    Contains all violations, score, and summary.

    Attributes:
        violations: All violations found
        files_reviewed: Number of files reviewed
        rules_applied: Number of rules applied
        score: Quality score (0-100)
        summary: Markdown summary of the review
    """

    violations: List[Violation] = field(default_factory=list)
    files_reviewed: int = 0
    rules_applied: int = 0
    score: int = 100
    summary: str = ""

    @property
    def comments(self) -> List[ReviewComment]:
        """Convert violations to review comments."""
        return [v.to_comment() for v in self.violations]

    @property
    def has_critical_issues(self) -> bool:
        """Check if any critical violations exist."""
        return any(v.is_critical for v in self.violations)

    @property
    def has_errors(self) -> bool:
        """Check if any error-level violations exist."""
        return any(v.is_error for v in self.violations)

    @property
    def approved(self) -> bool:
        """Check if review passes (no errors or critical issues)."""
        return not self.has_errors and not self.has_critical_issues

    @property
    def review_event(self) -> str:
        """Get GitHub review event type.

        Note: We always use COMMENT because GitHub doesn't allow:
        - Self-approval (APPROVE on your own PR)
        - Self-request-changes (REQUEST_CHANGES on your own PR)

        The review body will indicate if there are issues that need attention.
        """
        # Always use COMMENT to avoid GitHub restrictions on self-review
        return "COMMENT"

    @property
    def violation_count(self) -> int:
        """Total number of violations."""
        return len(self.violations)

    @property
    def critical_count(self) -> int:
        """Number of critical violations."""
        return sum(1 for v in self.violations if v.is_critical)

    @property
    def error_count(self) -> int:
        """Number of error violations."""
        return sum(1 for v in self.violations if v.severity == ReviewSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        """Number of warning violations."""
        return sum(1 for v in self.violations if v.severity == ReviewSeverity.WARNING)

    @classmethod
    def from_violations(
        cls,
        violations: List[Violation],
        files_reviewed: int = 0,
        rules_applied: int = 0,
    ) -> "ReviewResult":
        """Create ReviewResult from a list of violations."""
        # If no files were reviewed, score is 0 (failed review)
        if files_reviewed == 0:
            score = 0
        else:
            # Calculate score based on violations
            score = 100
            for v in violations:
                if v.severity == ReviewSeverity.CRITICAL:
                    score -= 25
                elif v.severity == ReviewSeverity.ERROR:
                    score -= 15
                elif v.severity == ReviewSeverity.WARNING:
                    score -= 5
                elif v.severity == ReviewSeverity.INFO:
                    score -= 1

            score = max(0, score)

        # Generate summary
        summary = cls._generate_summary(violations, score, files_reviewed, rules_applied)

        return cls(
            violations=violations,
            files_reviewed=files_reviewed,
            rules_applied=rules_applied,
            score=score,
            summary=summary,
        )

    @staticmethod
    def _generate_summary(
        violations: List[Violation],
        score: int,
        files_reviewed: int,
        rules_applied: int,
    ) -> str:
        """Generate markdown summary."""
        lines = []

        # Header with score
        match (files_reviewed, score):
            case (0, _):
                emoji, verdict = "❌", "Review Failed"
            case (_, s) if s >= 90:
                emoji, verdict = "✅", "Excellent"
            case (_, s) if s >= 70:
                emoji, verdict = "👍", "Good"
            case (_, s) if s >= 50:
                emoji, verdict = "⚠️", "Needs Improvement"
            case _:
                emoji, verdict = "❌", "Needs Work"

        lines.append(f"## {emoji} Code Review: {verdict}")
        lines.append("")
        lines.append(f"**Score: {score}/100**")
        lines.append("")
        lines.append(f"Reviewed {files_reviewed} files against {rules_applied} rules.")
        lines.append("")

        if files_reviewed == 0:
            lines.append(
                "⚠️ No files were successfully reviewed. "
                "Check Ollama connection or increase timeouts."
            )
        elif not violations:
            lines.append("No issues found. Great work! 🎉")
        else:
            # Count by severity
            critical = sum(1 for v in violations if v.severity == ReviewSeverity.CRITICAL)
            errors = sum(1 for v in violations if v.severity == ReviewSeverity.ERROR)
            warnings = sum(1 for v in violations if v.severity == ReviewSeverity.WARNING)
            info = sum(1 for v in violations if v.severity == ReviewSeverity.INFO)

            lines.append("### Issues Found")
            lines.append("")
            if critical:
                lines.append(f"- 🔴 **Critical**: {critical}")
            if errors:
                lines.append(f"- 🟠 **Errors**: {errors}")
            if warnings:
                lines.append(f"- 🟡 **Warnings**: {warnings}")
            if info:
                lines.append(f"- 🔵 **Info**: {info}")

            # Top violations
            lines.append("")
            lines.append("### Details")
            lines.append("")
            for v in violations[:10]:  # Show first 10
                severity_icon = {
                    ReviewSeverity.CRITICAL: "🔴",
                    ReviewSeverity.ERROR: "🟠",
                    ReviewSeverity.WARNING: "🟡",
                    ReviewSeverity.INFO: "🔵",
                }.get(v.severity, "⚪")
                lines.append(f"- {severity_icon} **{v.file}:{v.line}** - {v.message}")

            if len(violations) > 10:
                lines.append(f"- ... and {len(violations) - 10} more")

        return "\n".join(lines)
