"""Domain Rules - Business rules for code review.

Defines the structure of rules parsed from AGENTS.md:
- RuleType: Pattern, Anti-Pattern, or Code Smell
- RuleSeverity: INFO, WARNING, ERROR, CRITICAL
- RuleCategory: Specific rule categories
- ParsedRule: A single rule with examples and indicators
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class RuleType(Enum):
    """Type of rule - determines how violations are reported."""

    PATTERN = "pattern"  # Good practices to enforce
    ANTI_PATTERN = "anti_pattern"  # Bad practices to flag
    CODE_SMELL = "code_smell"  # Symptoms to review


class RuleSeverity(Enum):
    """Severity level of a rule violation."""

    INFO = "info"  # Informational, no action required
    WARNING = "warning"  # Should be addressed
    ERROR = "error"  # Must be addressed
    CRITICAL = "critical"  # Security/breaking issue


class RuleCategory(Enum):
    """Category of rule for grouping and filtering."""

    # Design Patterns
    FACTORY_PATTERN = "factory_pattern"
    SETTINGS_DATACLASS = "settings_dataclass"
    FLUENT_API = "fluent_api"
    ERROR_HIERARCHY = "error_hierarchy"
    COMPOSITION = "composition"
    MOCK_CLASSES = "mock_classes"

    # SOLID Principles
    SOLID_SRP = "solid_srp"
    SOLID_OCP = "solid_ocp"
    SOLID_LSP = "solid_lsp"
    SOLID_ISP = "solid_isp"
    SOLID_DIP = "solid_dip"

    # Design Principles
    RULE_OF_THREE = "rule_of_three"

    # Anti-Patterns
    GOD_CLASS = "god_class"
    SPECULATIVE_GENERALITY = "speculative_generality"
    INAPPROPRIATE_INTIMACY = "inappropriate_intimacy"
    SINGLETON = "singleton"

    # Code Smells
    LONG_METHOD = "long_method"
    DEEP_NESTING = "deep_nesting"
    MAGIC_NUMBERS = "magic_numbers"
    COPY_PASTE = "copy_paste"
    DEAD_CODE = "dead_code"
    GLOBAL_VARIABLES = "global_variables"
    HARDCODED_CREDENTIALS = "hardcoded_credentials"
    MISSING_ERROR_HANDLING = "missing_error_handling"
    PRIMITIVE_OBSESSION = "primitive_obsession"
    FEATURE_ENVY = "feature_envy"
    DATA_CLUMPS = "data_clumps"
    LONG_PARAMETER_LIST = "long_parameter_list"
    SHOTGUN_SURGERY = "shotgun_surgery"

    # Generic
    OTHER = "other"


@dataclass
class CodeExample:
    """A code example (good or bad) for a rule."""

    code: str
    is_good: bool
    language: str = "python"
    description: Optional[str] = None


@dataclass
class ParsedRule:
    """
    A single rule parsed from AGENTS.md.

    Rules define coding standards to enforce during review.

    Attributes:
        id: Unique identifier (snake_case)
        name: Human-readable name
        description: Detailed description
        rule_type: Pattern, Anti-Pattern, or Code Smell
        category: Rule category for grouping
        severity: How serious violations are
        is_required: Whether rule is mandatory (REQUIRED in title)
        good_examples: Examples of correct code
        bad_examples: Examples of incorrect code
        indicators: Signs that indicate a violation
        refactoring_suggestions: How to fix violations
        is_disabled: Whether rule is disabled via overrides
    """

    id: str
    name: str
    description: str
    rule_type: RuleType
    category: RuleCategory
    severity: RuleSeverity
    is_required: bool = False
    good_examples: List[CodeExample] = field(default_factory=list)
    bad_examples: List[CodeExample] = field(default_factory=list)
    indicators: List[str] = field(default_factory=list)
    refactoring_suggestions: List[str] = field(default_factory=list)
    is_disabled: bool = False

    def to_prompt_section(self) -> str:
        """Convert rule to a prompt section for AI review."""
        type_label = {
            RuleType.PATTERN: "✅ PATTERN (enforce)",
            RuleType.ANTI_PATTERN: "❌ ANTI-PATTERN (flag)",
            RuleType.CODE_SMELL: "⚠️ CODE SMELL (review)",
        }.get(self.rule_type, "RULE")

        lines = [
            f"### {self.name}",
            f"**Type**: {type_label}",
            f"**Category**: {self.category.value}",
            f"**Severity**: {self.severity.value}",
            f"**Required**: {'Yes' if self.is_required else 'No'}",
            "",
            self.description,
        ]

        if self.indicators:
            lines.append("\n**Look for:**")
            for indicator in self.indicators[:5]:
                lines.append(f"- {indicator}")

        if self.good_examples:
            lines.append("\n**Good Example:**")
            for ex in self.good_examples[:1]:
                lines.append(f"```{ex.language}")
                lines.append(ex.code[:500])
                lines.append("```")

        if self.bad_examples:
            lines.append("\n**Bad Example:**")
            for ex in self.bad_examples[:1]:
                lines.append(f"```{ex.language}")
                lines.append(ex.code[:500])
                lines.append("```")

        return "\n".join(lines)

    @property
    def is_active(self) -> bool:
        """Check if rule is active (not disabled)."""
        return not self.is_disabled

    def matches_file(self, filename: str) -> bool:
        """Check if rule applies to a file type."""
        # Most rules apply to Python files
        if filename.endswith(".py"):
            return True
        # Some rules apply to all code
        code_extensions = {".py", ".js", ".ts", ".java", ".go", ".rs", ".rb"}
        return any(filename.endswith(ext) for ext in code_extensions)


# Constants for rule detection
ANTI_PATTERN_CATEGORIES = {
    RuleCategory.GOD_CLASS,
    RuleCategory.SPECULATIVE_GENERALITY,
    RuleCategory.INAPPROPRIATE_INTIMACY,
    RuleCategory.SINGLETON,
}

CODE_SMELL_CATEGORIES = {
    RuleCategory.LONG_METHOD,
    RuleCategory.DEEP_NESTING,
    RuleCategory.MAGIC_NUMBERS,
    RuleCategory.COPY_PASTE,
    RuleCategory.DEAD_CODE,
    RuleCategory.GLOBAL_VARIABLES,
    RuleCategory.HARDCODED_CREDENTIALS,
    RuleCategory.MISSING_ERROR_HANDLING,
    RuleCategory.PRIMITIVE_OBSESSION,
    RuleCategory.FEATURE_ENVY,
    RuleCategory.DATA_CLUMPS,
    RuleCategory.LONG_PARAMETER_LIST,
    RuleCategory.SHOTGUN_SURGERY,
}
