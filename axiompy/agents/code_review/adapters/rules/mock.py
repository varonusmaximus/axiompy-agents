"""Mock Rules Source - For testing.

Implements RulesSource with configurable content.
"""

from typing import Optional

# Sample AGENTS.md content for testing
SAMPLE_RULES = """
# Code Review Rules

## Design Patterns

### Factory Pattern (REQUIRED)
All major classes must have a corresponding Factory class.

```python
# Good
class MyServiceFactory:
    @staticmethod
    def create() -> MyService:
        return MyService()
```

### Settings Dataclass
Use dataclasses with validation.

## Anti-Patterns

### God Class (ERROR)
A class that does too much. Avoid this anti-pattern.

**Look for:**
- More than 500 lines
- More than 10 public methods

## Code Smells

### Magic Numbers (WARNING)
Hardcoded numeric values without explanation.
"""


class MockRulesSource:
    """
    Mock rules source for testing.

    Example:
        source = MockRulesSource()
        source.set_rules("# Custom Rules...")
        content = source.get_rules()
    """

    def __init__(self, rules: Optional[str] = None, overrides: Optional[str] = None):
        """
        Initialize mock rules source.

        Args:
            rules: Rules content (uses sample if not provided)
            overrides: Overrides content
        """
        self._rules = rules or SAMPLE_RULES
        self._overrides = overrides
        self.calls: list = []

    def set_rules(self, content: str) -> "MockRulesSource":
        """Set rules content."""
        self._rules = content
        return self

    def set_overrides(self, content: str) -> "MockRulesSource":
        """Set overrides content."""
        self._overrides = content
        return self

    def reset(self) -> None:
        """Reset to default state."""
        self._rules = SAMPLE_RULES
        self._overrides = None
        self.calls.clear()

    def get_rules(self) -> str:
        """Get rules content."""
        self.calls.append(("get_rules", ()))
        return self._rules

    def get_local_overrides(self) -> Optional[str]:
        """Get overrides content."""
        self.calls.append(("get_local_overrides", ()))
        return self._overrides
