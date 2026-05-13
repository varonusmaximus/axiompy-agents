"""Console Publisher - Print results to terminal.

Implements ReviewPublisher for CLI output.
"""

import sys
from typing import TextIO

from ...domain.results import ReviewResult, ReviewSeverity


class ConsolePublisher:
    """
    Print review results to terminal.

    Provides colorized output for CLI use.

    Example:
        publisher = ConsolePublisher()
        publisher.publish(result, context)
    """

    def __init__(
        self,
        output: TextIO = sys.stdout,
        use_color: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize console publisher.

        Args:
            output: Output stream (default: stdout)
            use_color: Whether to use ANSI colors
            verbose: Whether to show detailed output
        """
        self.output = output
        self.use_color = use_color and hasattr(output, "isatty") and output.isatty()
        self.verbose = verbose

    def publish(self, result: ReviewResult, context: dict) -> None:
        """
        Print review result to console.

        Args:
            result: Review result to print
            context: Additional context (unused for console)
        """
        lines = []

        # Header
        lines.append("")
        lines.append(self._header(result))
        lines.append("")

        # Score
        score_color = self._score_color(result.score)
        lines.append(f"Score: {self._color(f'{result.score}/100', score_color)}")
        lines.append(f"Files reviewed: {result.files_reviewed}")
        lines.append(f"Rules applied: {result.rules_applied}")
        lines.append("")

        # Violations summary
        if result.violations:
            lines.append(self._color("Violations:", "bold"))
            lines.append(f"  Critical: {result.critical_count}")
            lines.append(f"  Errors: {result.error_count}")
            lines.append(f"  Warnings: {result.warning_count}")
            lines.append("")

            # Violation details
            if self.verbose or len(result.violations) <= 10:
                for v in result.violations:
                    icon = self._severity_icon(v.severity)
                    color = self._severity_color(v.severity)
                    location = f"{v.file}:{v.line}" if v.line else v.file
                    lines.append(
                        f"  {icon} {self._color(location, 'cyan')}: "
                        f"{self._color(v.rule_name, color)} - {v.message}"
                    )
                    if v.suggestion and self.verbose:
                        lines.append(f"      💡 {v.suggestion}")
            else:
                # Show all violations in compact format
                for v in result.violations:
                    icon = self._severity_icon(v.severity)
                    color = self._severity_color(v.severity)
                    location = f"{v.file}:{v.line}" if v.line else v.file
                    lines.append(
                        f"  {icon} {self._color(location, 'cyan')}: "
                        f"{self._color(v.rule_name, color)}"
                    )
        else:
            lines.append(self._color("✅ No violations found!", "green"))

        lines.append("")

        # Verdict
        if result.approved:
            lines.append(self._color("Review: PASSED", "green"))
        else:
            lines.append(self._color("Review: FAILED", "red"))

        lines.append("")

        # Print all
        self.output.write("\n".join(lines))
        self.output.flush()

    def _header(self, result: ReviewResult) -> str:
        """Generate header line."""
        if result.score >= 90:
            return self._color("═══ Code Review: Excellent ═══", "green")
        elif result.score >= 70:
            return self._color("═══ Code Review: Good ═══", "yellow")
        elif result.score >= 50:
            return self._color("═══ Code Review: Needs Improvement ═══", "yellow")
        else:
            return self._color("═══ Code Review: Needs Work ═══", "red")

    def _score_color(self, score: int) -> str:
        """Get color for score."""
        if score >= 90:
            return "green"
        elif score >= 70:
            return "yellow"
        else:
            return "red"

    def _severity_icon(self, severity: ReviewSeverity) -> str:
        """Get icon for severity."""
        icons = {
            ReviewSeverity.CRITICAL: "🔴",
            ReviewSeverity.ERROR: "🟠",
            ReviewSeverity.WARNING: "🟡",
            ReviewSeverity.INFO: "🔵",
        }
        return icons.get(severity, "⚪")

    def _severity_color(self, severity: ReviewSeverity) -> str:
        """Get color for severity."""
        colors = {
            ReviewSeverity.CRITICAL: "red",
            ReviewSeverity.ERROR: "red",
            ReviewSeverity.WARNING: "yellow",
            ReviewSeverity.INFO: "blue",
        }
        return colors.get(severity, "white")

    def _color(self, text: str, color: str) -> str:
        """Apply ANSI color if enabled."""
        if not self.use_color:
            return text

        codes = {
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "cyan": "\033[96m",
            "white": "\033[97m",
            "bold": "\033[1m",
            "reset": "\033[0m",
        }

        code = codes.get(color, "")
        reset = codes["reset"]

        return f"{code}{text}{reset}"
