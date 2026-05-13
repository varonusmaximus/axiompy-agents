"""File Rules Source - Read rules from local filesystem.

Implements RulesSource for local AGENTS.md files.
"""

from pathlib import Path
from typing import Optional


class FileRulesSource:
    """
    Read rules from local filesystem.

    Example:
        source = FileRulesSource("AGENTS.md")
        content = source.get_rules()
    """

    def __init__(
        self,
        rules_path: str = "AGENTS.md",
        overrides_path: Optional[str] = ".cursorrules",
    ):
        """
        Initialize file rules source.

        Args:
            rules_path: Path to main rules file
            overrides_path: Path to local overrides file (None to skip)
        """
        self.rules_path = Path(rules_path)
        self.overrides_path = Path(overrides_path) if overrides_path else None

    def get_rules(self) -> str:
        """
        Get rules content.

        Returns:
            Rules content as string

        Raises:
            FileNotFoundError: If rules file doesn't exist
        """
        if not self.rules_path.exists():
            raise FileNotFoundError(f"Rules file not found: {self.rules_path}")

        return self.rules_path.read_text(encoding="utf-8")

    def get_local_overrides(self) -> Optional[str]:
        """
        Get local overrides if they exist.

        Returns:
            Overrides content or None
        """
        if self.overrides_path and self.overrides_path.exists():
            return self.overrides_path.read_text(encoding="utf-8")
        return None
