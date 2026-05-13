"""Domain Engine - Core business logic for code review.

The RulesEngine is responsible for:
- Parsing AGENTS.md into structured rules
- Building prompts for AI analysis
- Parsing AI responses into violations

This is pure Python with no external dependencies.
"""

import re
from typing import List, Optional, Tuple

from .models import CodeFile
from .results import ReviewSeverity, Violation
from .rules import (
    ANTI_PATTERN_CATEGORIES,
    CODE_SMELL_CATEGORIES,
    CodeExample,
    ParsedRule,
    RuleCategory,
    RuleSeverity,
    RuleType,
)


class RulesEngine:
    """
    Core engine for parsing rules and building review prompts.

    This class contains all the business logic for code review,
    independent of any infrastructure concerns.

    Example:
        engine = RulesEngine()
        rules = engine.parse_rules(agents_md_content)
        prompt = engine.build_prompt(code_file, rules)
        violations = engine.parse_response(ai_response, "file.py")
    """

    # Mapping of keywords to categories
    CATEGORY_MAPPINGS = {
        # Design Patterns
        "factory": RuleCategory.FACTORY_PATTERN,
        "settings dataclass": RuleCategory.SETTINGS_DATACLASS,
        "fluent api": RuleCategory.FLUENT_API,
        "error hierarchy": RuleCategory.ERROR_HIERARCHY,
        "composition": RuleCategory.COMPOSITION,
        "mock": RuleCategory.MOCK_CLASSES,
        # SOLID
        "single responsibility": RuleCategory.SOLID_SRP,
        "srp": RuleCategory.SOLID_SRP,
        "open/closed": RuleCategory.SOLID_OCP,
        "ocp": RuleCategory.SOLID_OCP,
        "liskov": RuleCategory.SOLID_LSP,
        "lsp": RuleCategory.SOLID_LSP,
        "interface segregation": RuleCategory.SOLID_ISP,
        "isp": RuleCategory.SOLID_ISP,
        "dependency inversion": RuleCategory.SOLID_DIP,
        "dip": RuleCategory.SOLID_DIP,
        # Principles
        "rule of three": RuleCategory.RULE_OF_THREE,
        # Anti-patterns
        "god class": RuleCategory.GOD_CLASS,
        "speculative generality": RuleCategory.SPECULATIVE_GENERALITY,
        "inappropriate intimacy": RuleCategory.INAPPROPRIATE_INTIMACY,
        "singleton": RuleCategory.SINGLETON,
        # Code smells
        "long method": RuleCategory.LONG_METHOD,
        "deep nesting": RuleCategory.DEEP_NESTING,
        "magic number": RuleCategory.MAGIC_NUMBERS,
        "copy-paste": RuleCategory.COPY_PASTE,
        "dead code": RuleCategory.DEAD_CODE,
        "global variable": RuleCategory.GLOBAL_VARIABLES,
        "hardcoded credential": RuleCategory.HARDCODED_CREDENTIALS,
        "missing error handling": RuleCategory.MISSING_ERROR_HANDLING,
        "primitive obsession": RuleCategory.PRIMITIVE_OBSESSION,
        "feature envy": RuleCategory.FEATURE_ENVY,
        "data clump": RuleCategory.DATA_CLUMPS,
        "long parameter": RuleCategory.LONG_PARAMETER_LIST,
        "shotgun surgery": RuleCategory.SHOTGUN_SURGERY,
    }

    def parse_rules(
        self,
        content: str,
        overrides: Optional[str] = None,
    ) -> List[ParsedRule]:
        """
        Parse AGENTS.md content into structured rules.

        Args:
            content: AGENTS.md content
            overrides: Optional .cursorrules content for local overrides

        Returns:
            List of ParsedRule objects
        """
        rules = self._parse_content(content)

        if overrides:
            rules = self._apply_overrides(rules, overrides)

        return rules

    def filter_by_type(
        self,
        rules: List[ParsedRule],
        rule_type: RuleType,
    ) -> List[ParsedRule]:
        """
        Filter rules by type.

        Args:
            rules: All rules
            rule_type: Type to filter for

        Returns:
            Rules matching the type
        """
        return [r for r in rules if r.rule_type == rule_type]

    def filter_by_severity(
        self,
        rules: List[ParsedRule],
        min_severity: RuleSeverity,
    ) -> List[ParsedRule]:
        """
        Filter rules by minimum severity.

        Args:
            rules: All rules
            min_severity: Minimum severity to include

        Returns:
            Rules at or above minimum severity
        """
        severity_order = [
            RuleSeverity.INFO,
            RuleSeverity.WARNING,
            RuleSeverity.ERROR,
            RuleSeverity.CRITICAL,
        ]
        min_index = severity_order.index(min_severity)
        return [r for r in rules if severity_order.index(r.severity) >= min_index]

    def chunk_rules(
        self,
        rules: List[ParsedRule],
    ) -> dict:
        """
        Split rules into chunks by type for incremental review.

        Returns dict with keys: 'patterns', 'anti_patterns', 'code_smells'

        Example:
            chunks = engine.chunk_rules(rules)
            for chunk_name, chunk_rules in chunks.items():
                result = service.review_with_rules(code, chunk_rules)
        """
        return {
            "patterns": self.filter_by_type(rules, RuleType.PATTERN),
            "anti_patterns": self.filter_by_type(rules, RuleType.ANTI_PATTERN),
            "code_smells": self.filter_by_type(rules, RuleType.CODE_SMELL),
        }

    def build_prompt(
        self,
        code: CodeFile,
        rules: List[ParsedRule],
        context: Optional[str] = None,
        compact: bool = False,
    ) -> str:
        """
        Build a prompt for AI analysis.

        Args:
            code: The code file to review
            rules: Rules to enforce
            context: Optional additional context (PR description, etc.)
            compact: If True, use minimal rule descriptions (faster)

        Returns:
            Prompt string for AI
        """
        active_rules = [r for r in rules if r.is_active and r.matches_file(code.path)]

        prompt_parts = [
            "You are an expert code reviewer. Review the following code against these rules.",
            "",
            "# Rules to Enforce",
            "",
        ]

        # Add rules by type - use smaller limits for speed
        patterns = [r for r in active_rules if r.rule_type == RuleType.PATTERN]
        anti_patterns = [r for r in active_rules if r.rule_type == RuleType.ANTI_PATTERN]
        smells = [r for r in active_rules if r.rule_type == RuleType.CODE_SMELL]

        # Limits: compact mode uses fewer rules
        pattern_limit = 5 if compact else 10
        anti_pattern_limit = 5 if compact else 8
        smell_limit = 5 if compact else 8

        if patterns:
            prompt_parts.append("## Good Patterns (enforce these)")
            for rule in patterns[:pattern_limit]:
                if compact:
                    prompt_parts.append(f"- **{rule.name}** [{rule.severity.value}]")
                else:
                    prompt_parts.append(rule.to_prompt_section())
                    prompt_parts.append("")

        if anti_patterns:
            prompt_parts.append("## Anti-Patterns (flag these)")
            for rule in anti_patterns[:anti_pattern_limit]:
                if compact:
                    prompt_parts.append(
                        f"- **{rule.name}** [{rule.severity.value}]: {rule.description[:100]}"
                    )
                else:
                    prompt_parts.append(rule.to_prompt_section())
                    prompt_parts.append("")

        if smells:
            prompt_parts.append("## Code Smells (review these)")
            for rule in smells[:smell_limit]:
                if compact:
                    prompt_parts.append(
                        f"- **{rule.name}** [{rule.severity.value}]: {rule.description[:100]}"
                    )
                else:
                    prompt_parts.append(rule.to_prompt_section())
                    prompt_parts.append("")

        # Add the code
        prompt_parts.extend(
            [
                "# Code to Review",
                "",
                f"**File**: `{code.path}`",
                f"**Language**: {code.language}",
                "",
                f"```{code.language}",
                code.content[:10000],  # Limit content size
                "```",
                "",
            ]
        )

        # Add context if provided
        if context:
            prompt_parts.extend(
                [
                    "# Additional Context",
                    "",
                    context,
                    "",
                ]
            )

        # Add response format instructions
        prompt_parts.extend(
            [
                "# Response Format",
                "",
                "IMPORTANT: Only report ACTUAL violations found in the code. "
                "Do NOT list rules that are followed correctly.",
                "",
                "## Summary",
                "[1-2 sentences about what you found]",
                "",
                "## Score",
                "[Number 0-100. 100 = perfect code, 0 = many critical issues]",
                "",
                "## Violations",
                "",
                "If no violations found, write exactly: 'No violations found.'",
                "",
                "If violations ARE found, list ONLY the actual problems using this format:",
                "",
                "### [Specific Rule Name]",
                "- **Line**: [exact line number where issue occurs]",
                "- **Severity**: [CRITICAL/ERROR/WARNING/INFO]",
                "- **Message**: [what is wrong on this specific line]",
                "- **Suggestion**: [how to fix it]",
                "",
                "REMINDER: Only list rules that the code VIOLATES. "
                "A 25-line config file cannot have 'God Class' or 'Long Method'.",
            ]
        )

        return "\n".join(prompt_parts)

    def parse_response(
        self,
        response: str,
        filename: str,
    ) -> List[Violation]:
        """
        Parse AI response into violations.

        Args:
            response: AI response text
            filename: Name of the reviewed file

        Returns:
            List of Violation objects
        """
        violations = []

        # Look for violation sections
        violation_pattern = r"###\s+(.+?)\n(.*?)(?=###|\Z)"
        matches = re.findall(violation_pattern, response, re.DOTALL)

        for rule_name, content in matches:
            rule_name = rule_name.strip()

            # Skip non-violation sections
            skip_patterns = [
                "summary",
                "score",
                "no violations",
                "violations found",
                "good patterns",
                "anti-patterns",
                "code smells",
                "correction",
                "response format",
                "rules to enforce",
            ]
            if any(skip in rule_name.lower() for skip in skip_patterns):
                continue

            # Parse violation details
            line_match = re.search(r"\*\*Line\*\*:\s*(\d+)", content)
            severity_match = re.search(r"\*\*Severity\*\*:\s*(\w+)", content, re.IGNORECASE)
            message_match = re.search(r"\*\*Message\*\*:\s*(.+?)(?=\n\*\*|\Z)", content, re.DOTALL)
            suggestion_match = re.search(
                r"\*\*Suggestion\*\*:\s*(.+?)(?=\n\*\*|\Z)", content, re.DOTALL
            )

            # Require at least a line number OR a meaningful message
            line = int(line_match.group(1)) if line_match else 0
            message = message_match.group(1).strip() if message_match else ""

            # Skip if no real content (just rule name with no details)
            if not line and not message:
                continue
            if not message:
                message = content.strip()[:200]

            severity_str = severity_match.group(1).upper() if severity_match else "WARNING"
            suggestion = suggestion_match.group(1).strip() if suggestion_match else None

            # Map severity
            severity_map = {
                "INFO": ReviewSeverity.INFO,
                "WARNING": ReviewSeverity.WARNING,
                "ERROR": ReviewSeverity.ERROR,
                "CRITICAL": ReviewSeverity.CRITICAL,
            }
            severity = severity_map.get(severity_str, ReviewSeverity.WARNING)

            # Create rule ID from name
            rule_id = re.sub(r"[^a-z0-9]+", "_", rule_name.lower()).strip("_")

            violations.append(
                Violation(
                    file=filename,
                    line=line,
                    rule_id=rule_id,
                    rule_name=rule_name,
                    message=message,
                    severity=severity,
                    suggestion=suggestion,
                )
            )

        return violations

    def get_rules_summary(self, rules: List[ParsedRule]) -> str:
        """Generate a summary of loaded rules."""
        active = [r for r in rules if r.is_active]
        disabled = [r for r in rules if r.is_disabled]
        required = [r for r in active if r.is_required]

        by_type = {}
        for r in active:
            by_type.setdefault(r.rule_type.value, []).append(r)

        by_severity = {}
        for r in active:
            by_severity.setdefault(r.severity.value, []).append(r)

        lines = [
            "# Parsed Rules Summary",
            "",
            f"**Total Rules**: {len(rules)}",
            f"**Active Rules**: {len(active)}",
            f"**Disabled Rules**: {len(disabled)}",
            f"**Required Rules**: {len(required)}",
            "",
            "**By Type**:",
        ]

        for rule_type, rule_list in sorted(by_type.items()):
            emoji = {"pattern": "✅", "anti_pattern": "❌", "code_smell": "⚠️"}.get(rule_type, "")
            lines.append(f"- {emoji} {rule_type}: {len(rule_list)}")

        lines.append("")
        lines.append("**By Severity**:")
        for severity, rule_list in sorted(by_severity.items()):
            lines.append(f"- {severity}: {len(rule_list)}")

        return "\n".join(lines)

    # =========================================================================
    # Private methods for parsing
    # =========================================================================

    def _parse_content(self, content: str) -> List[ParsedRule]:
        """Parse AGENTS.md content into rules."""
        rules = []

        # Split by h2 and h3 headers
        sections = re.split(r"(?=^##\s)", content, flags=re.MULTILINE)

        for section in sections:
            if not section.strip():
                continue

            # Parse h3 subsections as individual rules
            subsections = re.split(r"(?=^###\s)", section, flags=re.MULTILINE)

            for subsection in subsections:
                if not subsection.strip():
                    continue

                rule = self._parse_section(subsection)
                if rule:
                    rules.append(rule)

        # Also parse checklist items
        checklist_rules = self._parse_checklist(content)
        rules.extend(checklist_rules)

        return rules

    def _parse_section(self, section: str) -> Optional[ParsedRule]:
        """Parse a single section into a rule."""
        lines = section.strip().split("\n")
        if not lines:
            return None

        # Extract title from first line
        title_line = lines[0]
        title_match = re.match(r"^#{2,3}\s+(.+)$", title_line)
        if not title_match:
            return None

        title = title_match.group(1).strip()

        # Skip non-rule sections
        skip_keywords = ["overview", "table of contents", "installation", "quick start"]
        if any(kw in title.lower() for kw in skip_keywords):
            return None

        # Get content after title
        content = "\n".join(lines[1:])

        # Determine category
        category = self._categorize(title)
        if category == RuleCategory.OTHER:
            # Try harder to categorize
            category = self._categorize(content[:200])

        # Determine rule type
        rule_type = self._determine_type(category, title, content)

        # Extract severity
        severity = self._extract_severity(title, content, rule_type)

        # Check if required
        is_required = "REQUIRED" in title.upper() or "(REQUIRED" in title.upper()

        # Extract description (first paragraph after title)
        description = self._extract_description(content)

        # Extract examples
        good_examples, bad_examples = self._extract_examples(content)

        # Extract indicators
        indicators = self._extract_indicators(content)

        # Generate ID
        rule_id = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        rule_id = rule_id[:50]  # Limit length

        return ParsedRule(
            id=rule_id,
            name=title.split("(")[0].strip(),  # Remove severity markers
            description=description,
            rule_type=rule_type,
            category=category,
            severity=severity,
            is_required=is_required,
            good_examples=good_examples,
            bad_examples=bad_examples,
            indicators=indicators,
        )

    def _categorize(self, text: str) -> RuleCategory:
        """Categorize text by matching keywords."""
        text_lower = text.lower()
        for keyword, category in self.CATEGORY_MAPPINGS.items():
            if keyword in text_lower:
                return category
        return RuleCategory.OTHER

    def _determine_type(
        self,
        category: RuleCategory,
        title: str,
        content: str,
    ) -> RuleType:
        """Determine if rule is pattern, anti-pattern, or code smell."""
        title_lower = title.lower()
        content_lower = content[:500].lower()

        # Check for anti-pattern markers
        if any(
            m in title_lower or m in content_lower
            for m in ["anti-pattern", "antipattern", "avoid", "don't", "bad practice"]
        ):
            return RuleType.ANTI_PATTERN

        # Check for code smell markers
        if any(m in title_lower or m in content_lower for m in ["code smell", "smell", "symptom"]):
            return RuleType.CODE_SMELL

        # Check category
        if category in ANTI_PATTERN_CATEGORIES:
            return RuleType.ANTI_PATTERN

        if category in CODE_SMELL_CATEGORIES:
            return RuleType.CODE_SMELL

        return RuleType.PATTERN

    def _extract_severity(
        self,
        title: str,
        content: str,
        rule_type: RuleType,
    ) -> RuleSeverity:
        """Extract or infer severity."""
        title_upper = title.upper()

        if "(CRITICAL)" in title_upper:
            return RuleSeverity.CRITICAL
        if "(ERROR)" in title_upper:
            return RuleSeverity.ERROR
        if "(WARNING)" in title_upper:
            return RuleSeverity.WARNING
        if "(INFO)" in title_upper:
            return RuleSeverity.INFO

        # Infer from rule type
        if rule_type == RuleType.ANTI_PATTERN:
            return RuleSeverity.ERROR
        if rule_type == RuleType.CODE_SMELL:
            return RuleSeverity.WARNING

        return RuleSeverity.WARNING

    def _extract_description(self, content: str) -> str:
        """Extract first paragraph as description."""
        # Remove code blocks
        content = re.sub(r"```[\s\S]*?```", "", content)

        # Get first paragraph
        paragraphs = content.strip().split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if para and not para.startswith(("#", "-", "*", "|", ">")):
                return para[:500]

        return ""

    def _extract_examples(
        self,
        content: str,
    ) -> Tuple[List[CodeExample], List[CodeExample]]:
        """Extract good and bad code examples."""
        good_examples = []
        bad_examples = []

        # Find code blocks
        pattern = r"```(\w*)\n(.*?)```"
        blocks = re.findall(pattern, content, re.DOTALL)

        for lang, code in blocks:
            lang = lang or "python"
            code = code.strip()

            if not code:
                continue

            # Determine if good or bad by looking at preceding text
            # Find position of this block
            block_pos = content.find(f"```{lang}\n{code}")
            preceding = content[:block_pos].lower() if block_pos > 0 else ""

            is_good = any(m in preceding[-100:] for m in ["good", "correct", "example:", "✅"])
            is_bad = any(m in preceding[-100:] for m in ["bad", "wrong", "avoid", "❌", "# bad"])

            example = CodeExample(code=code[:500], is_good=is_good, language=lang)

            if is_bad or "# bad" in code.lower():
                bad_examples.append(CodeExample(code=code[:500], is_good=False, language=lang))
            elif is_good or "# good" in code.lower():
                good_examples.append(example)

        return good_examples, bad_examples

    def _extract_indicators(self, content: str) -> List[str]:
        """Extract indicators/symptoms from content."""
        indicators = []

        # Look for bullet lists after "Indicators", "Look for", "Signs"
        pattern = r"(?:Indicators?|Look for|Signs?|Symptoms?)[:\s]*\n((?:\s*[-*]\s+.+\n?)+)"
        matches = re.findall(pattern, content, re.IGNORECASE)

        for match in matches:
            items = re.findall(r"[-*]\s+(.+)", match)
            indicators.extend(item.strip() for item in items[:10])

        return indicators

    def _parse_checklist(self, content: str) -> List[ParsedRule]:
        """Parse checklist items as rules."""
        rules = []

        # Find checklist section
        checklist_match = re.search(
            r"(?:Common Patterns Checklist|Checklist).*?\n((?:\s*[-*]\s*\[.\]\s+.+\n?)+)",
            content,
            re.IGNORECASE | re.DOTALL,
        )

        if not checklist_match:
            return rules

        checklist_content = checklist_match.group(1)
        items = re.findall(r"[-*]\s*\[.\]\s+(.+)", checklist_content)

        for item in items:
            item = item.strip()
            if not item:
                continue

            category = self._categorize(item)
            rule_id = f"checklist_{re.sub(r'[^a-z0-9]+', '_', item.lower())[:30]}"

            rules.append(
                ParsedRule(
                    id=rule_id,
                    name=item,
                    description=item,
                    rule_type=RuleType.PATTERN,
                    category=category,
                    severity=RuleSeverity.WARNING,
                    is_required=True,
                )
            )

        return rules

    def _apply_overrides(
        self,
        rules: List[ParsedRule],
        overrides: str,
    ) -> List[ParsedRule]:
        """Apply local overrides from .cursorrules."""
        # Parse disabled rules
        disabled_match = re.search(
            r"(?:Disabled Rules?|Disable)[:\s]*\n((?:\s*[-*]\s+.+\n?)+)", overrides, re.IGNORECASE
        )

        disabled_ids = set()
        if disabled_match:
            items = re.findall(r"[-*]\s+(.+)", disabled_match.group(1))
            for item in items:
                # Convert to rule ID format
                rule_id = re.sub(r"[^a-z0-9]+", "_", item.lower()).strip("_")
                disabled_ids.add(rule_id)

        # Apply disabled status
        for rule in rules:
            if rule.id in disabled_ids or rule.name.lower() in disabled_ids:
                rule.is_disabled = True

        return rules
