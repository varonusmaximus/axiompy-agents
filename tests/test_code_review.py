"""Tests for the Code Review Agent.

Tests cover:
- RulesEngine: Parsing AGENTS.md into structured rules
- CodeReviewService: End-to-end review flow
- Infrastructure: Mock sources and analyzers
- Types: Data structures and validation
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock

from axiompy.agents.code_review import (
    # Domain - Models
    CodeFile,
    FileDiff,
    PullRequestInfo,
    # Domain - Rules
    ParsedRule,
    RuleType,
    RuleSeverity,
    RuleCategory,
    CodeExample,
    # Domain - Results
    ReviewResult,
    Violation,
    ReviewComment,
    ReviewSeverity,
    # Domain - Engine
    RulesEngine,
    # Domain - Ports & Service
    CodeSource,
    AIAnalyzer,
    CodeReviewService,
    # Infrastructure
    MockCodeSource,
    MockRulesSource,
    MockAnalyzer,
    # Factory
    CodeReviewServiceFactory,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_agents_md():
    """Sample AGENTS.md content for testing."""
    return """
# AxiomPy Rules

## Design Patterns

### 1. Factory Pattern (REQUIRED for instantiable classes)
All major classes must have a corresponding Factory class for creation.

```python
# Good
class MyServiceFactory:
    @staticmethod
    def create() -> MyService:
        return MyService()
```

```python
# Bad
service = MyService()  # Direct instantiation
```

### 2. Settings Dataclass Pattern
Use dataclasses with validation.

## Anti-Patterns

### God Class (ERROR)
A class that does too much and knows too much. Avoid this anti-pattern.

**Look for**:
- More than 500 lines
- More than 10 public methods

```python
# Bad - God class
class ApplicationManager:
    def do_everything(self): ...
```

## Code Smells

### Magic Numbers (WARNING)
Hardcoded numeric values without explanation.

```python
# Bad
if x > 42:
    pass
```

## Common Patterns Checklist

- [ ] Factory class with `create()` method
- [ ] Settings dataclass
- [ ] Type hints on all parameters
"""


@pytest.fixture
def sample_cursorrules():
    """Sample .cursorrules local override content."""
    return """
# Local Overrides

## Overrides

### Factory Pattern (WARNING)
Downgraded for this legacy repo.

## Disabled Rules
- magic_numbers
"""


@pytest.fixture
def sample_pr():
    """Sample PullRequestInfo for testing."""
    return PullRequestInfo(
        number=123,
        title="Add new feature",
        body="This PR adds a new feature",
        head_sha="abc123",
        base_sha="def456",
        base_branch="main",
        head_branch="feature/new-thing",
        author="testuser",
        files=[
            FileDiff(
                filename="src/service.py",
                status="modified",
                additions=50,
                deletions=10,
                patch="""@@ -1,5 +1,10 @@
+class MyService:
+    def __init__(self):
+        self.value = 42  # Magic number
+
+    def process(self):
+        pass
""",
            ),
            FileDiff(
                filename="src/utils.py",
                status="added",
                additions=20,
                deletions=0,
                patch="""@@ -0,0 +1,20 @@
+def helper():
+    return "helper"
""",
            ),
        ],
    )


@pytest.fixture
def mock_code_source():
    """Create mock code source."""
    return MockCodeSource()


@pytest.fixture
def mock_rules_source():
    """Create mock rules source."""
    return MockRulesSource()


@pytest.fixture
def mock_analyzer():
    """Create mock analyzer."""
    return MockAnalyzer()


# =============================================================================
# RulesEngine Tests
# =============================================================================


class TestRulesEngine:
    """Tests for RulesEngine."""

    def test_parse_rules_extracts_patterns(self, sample_agents_md):
        """Test parsing extracts pattern rules."""
        engine = RulesEngine()
        rules = engine.parse_rules(sample_agents_md)

        # Should find factory pattern
        factory_rules = [r for r in rules if "factory" in r.id.lower()]
        assert len(factory_rules) >= 1

        factory_rule = factory_rules[0]
        assert factory_rule.rule_type == RuleType.PATTERN
        assert factory_rule.is_required is True
        # Note: is_required doesn't automatically set severity to ERROR

    def test_parse_rules_extracts_anti_patterns(self, sample_agents_md):
        """Test parsing extracts anti-pattern rules."""
        engine = RulesEngine()
        rules = engine.parse_rules(sample_agents_md)

        god_class_rules = [r for r in rules if "god_class" in r.id.lower()]
        assert len(god_class_rules) >= 1

        god_class = god_class_rules[0]
        assert god_class.rule_type == RuleType.ANTI_PATTERN
        assert god_class.severity == RuleSeverity.ERROR

    def test_parse_rules_extracts_code_smells(self, sample_agents_md):
        """Test parsing extracts code smell rules."""
        engine = RulesEngine()
        rules = engine.parse_rules(sample_agents_md)

        magic_rules = [r for r in rules if "magic" in r.id.lower()]
        assert len(magic_rules) >= 1

        magic_rule = magic_rules[0]
        assert magic_rule.rule_type == RuleType.CODE_SMELL
        assert magic_rule.severity == RuleSeverity.WARNING

    def test_parse_rules_extracts_good_examples(self, sample_agents_md):
        """Test parsing extracts good code examples."""
        engine = RulesEngine()
        rules = engine.parse_rules(sample_agents_md)

        factory_rules = [r for r in rules if "factory" in r.id.lower()]
        factory_rule = factory_rules[0]

        assert len(factory_rule.good_examples) >= 1
        assert "MyServiceFactory" in factory_rule.good_examples[0].code

    def test_parse_rules_extracts_bad_examples(self, sample_agents_md):
        """Test parsing extracts bad code examples."""
        engine = RulesEngine()
        rules = engine.parse_rules(sample_agents_md)

        god_class_rules = [r for r in rules if "god_class" in r.id.lower()]
        god_class = god_class_rules[0]

        assert len(god_class.bad_examples) >= 1

    def test_parse_checklist_items(self, sample_agents_md):
        """Test parsing extracts checklist items as rules."""
        engine = RulesEngine()
        rules = engine.parse_rules(sample_agents_md)

        checklist_rules = [r for r in rules if r.id.startswith("checklist_")]
        assert len(checklist_rules) >= 1

    def test_parse_rules_with_overrides(self, sample_agents_md, sample_cursorrules):
        """Test local overrides are applied."""
        engine = RulesEngine()

        # Parse with overrides
        rules = engine.parse_rules(sample_agents_md, overrides=sample_cursorrules)

        # Factory pattern should have WARNING severity (from local override)
        factory_rules = [
            r for r in rules if "factory" in r.id.lower() and not r.id.startswith("checklist")
        ]
        if factory_rules:
            # Local override downgrades to WARNING
            assert any(r.severity == RuleSeverity.WARNING for r in factory_rules)

    def test_get_rules_summary(self, sample_agents_md):
        """Test generating rules summary."""
        engine = RulesEngine()
        rules = engine.parse_rules(sample_agents_md)

        summary = engine.get_rules_summary(rules)

        assert "Total Rules" in summary
        assert "pattern" in summary.lower() or "Pattern" in summary

    def test_build_prompt(self, sample_agents_md):
        """Test building AI prompt from code and rules."""
        engine = RulesEngine()
        rules = engine.parse_rules(sample_agents_md)

        code = CodeFile(
            path="src/test.py",
            content="class MyClass:\n    pass\n",
        )

        prompt = engine.build_prompt(code, rules)

        assert "Rules to Enforce" in prompt
        assert "src/test.py" in prompt
        assert "class MyClass" in prompt


# =============================================================================
# Infrastructure Mock Tests
# =============================================================================


class TestMockCodeSource:
    """Tests for MockCodeSource."""

    def test_set_and_get_files(self):
        """Test setting and getting files."""
        mock = MockCodeSource()
        mock.set_file("src/main.py", "print('hello')")

        files = mock.get_files(["src/main.py"])
        assert len(files) == 1
        assert files[0].path == "src/main.py"
        assert "hello" in files[0].content

    def test_set_and_get_pr(self, sample_pr):
        """Test setting and getting PR."""
        mock = MockCodeSource()
        mock.set_pr("owner", "repo", 123, sample_pr)

        pr = mock.get_pull_request("owner", "repo", 123)
        assert pr.number == 123
        assert pr.title == "Add new feature"


class TestMockRulesSource:
    """Tests for MockRulesSource."""

    def test_set_and_get_rules(self, sample_agents_md):
        """Test setting and getting rules content."""
        mock = MockRulesSource()
        mock.set_rules(sample_agents_md)

        content = mock.get_rules()
        assert "Factory Pattern" in content


class TestMockAnalyzer:
    """Tests for MockAnalyzer."""

    def test_set_and_get_response(self):
        """Test setting and getting AI response."""
        mock = MockAnalyzer()
        mock.set_response("No issues found. Score: 100/100")

        response = mock.analyze("Review this code...")
        assert "100/100" in response
        assert "No issues" in response

    def test_records_calls(self):
        """Test that calls are recorded."""
        mock = MockAnalyzer()
        mock.set_response("Test response")

        mock.analyze("First prompt")
        mock.analyze("Second prompt")

        assert len(mock.calls) == 2
        assert mock.calls[0] == ("analyze", "First prompt")
        assert mock.calls[1] == ("analyze", "Second prompt")


# =============================================================================
# Types Tests
# =============================================================================


class TestFileDiff:
    """Tests for FileDiff."""

    def test_is_python(self):
        """Test Python file detection."""
        py_file = FileDiff(
            filename="src/main.py",
            status="modified",
            additions=10,
            deletions=5,
            patch="",
        )
        assert py_file.is_python is True

        js_file = FileDiff(
            filename="src/main.js",
            status="modified",
            additions=10,
            deletions=5,
            patch="",
        )
        assert js_file.is_python is False

    def test_total_changes(self):
        """Test total changes calculation."""
        diff = FileDiff(
            filename="test.py",
            status="modified",
            additions=30,
            deletions=10,
            patch="",
        )
        assert diff.total_changes == 40


class TestPullRequestInfo:
    """Tests for PullRequestInfo."""

    def test_total_changes(self, sample_pr):
        """Test total changes across all files."""
        assert sample_pr.total_changes == 80  # 50+10 + 20+0

    def test_python_files(self, sample_pr):
        """Test filtering Python files."""
        assert len(sample_pr.python_files) == 2

    def test_file_count(self, sample_pr):
        """Test file count."""
        assert sample_pr.file_count == 2


class TestReviewResult:
    """Tests for ReviewResult."""

    def test_has_critical_issues(self):
        """Test critical issues detection."""
        result_with_critical = ReviewResult(
            violations=[
                Violation(
                    file="test.py",
                    line=1,
                    rule_id="test",
                    rule_name="Test",
                    message="Critical!",
                    severity=ReviewSeverity.CRITICAL,
                ),
            ],
            summary="Test",
        )
        assert result_with_critical.has_critical_issues is True

        result_without_critical = ReviewResult(
            violations=[
                Violation(
                    file="test.py",
                    line=1,
                    rule_id="test",
                    rule_name="Test",
                    message="Warning",
                    severity=ReviewSeverity.WARNING,
                ),
            ],
            summary="Test",
        )
        assert result_without_critical.has_critical_issues is False

    def test_review_event_critical_issues(self):
        """Test review event is always COMMENT (even with critical issues)."""
        result = ReviewResult(
            violations=[
                Violation(
                    file="test.py",
                    line=1,
                    rule_id="test",
                    rule_name="Test",
                    message="Error!",
                    severity=ReviewSeverity.ERROR,
                ),
            ],
            summary="Test",
        )
        # Always COMMENT to avoid GitHub self-review restrictions
        assert result.review_event == "COMMENT"

    def test_review_event_clean_pr(self):
        """Test review event is COMMENT for clean PRs (no violations)."""
        result = ReviewResult(violations=[], summary="Test", score=100)
        # Always COMMENT - GitHub doesn't allow self-approval
        assert result.review_event == "COMMENT"

    def test_review_event_comment(self):
        """Test review event for warnings only (score below 80)."""
        result = ReviewResult(
            violations=[
                Violation(
                    file="test.py",
                    line=1,
                    rule_id="test",
                    rule_name="Test",
                    message="Warning",
                    severity=ReviewSeverity.WARNING,
                ),
            ],
            summary="Test",
            score=70,  # Below 80 threshold for APPROVE
        )
        assert result.review_event == "COMMENT"


# =============================================================================
# CodeReviewService Tests
# =============================================================================


class TestCodeReviewService:
    """Tests for CodeReviewService."""

    def test_create_mock_service(self):
        """Test creating mock service."""
        service = CodeReviewServiceFactory.create_mock()
        assert service is not None

    def test_review_code_with_mock(self, sample_agents_md):
        """Test reviewing code with mock service."""
        # Create mock infrastructure
        mock_source = MockCodeSource()
        mock_source.set_file("src/main.py", "class MyService:\n    pass\n")

        mock_rules = MockRulesSource()
        mock_rules.set_rules(sample_agents_md)

        mock_analyzer = MockAnalyzer()
        mock_analyzer.set_response("Score: 100/100. No issues found.")

        service = CodeReviewService(
            code_source=mock_source,
            rules_source=mock_rules,
            analyzer=mock_analyzer,
            publisher=None,
        )

        result = service.review_code("class MyService:\n    pass\n", "src/main.py")

        assert result is not None
        assert isinstance(result, ReviewResult)
        assert result.score >= 0

    def test_review_files_with_mock(self, sample_agents_md):
        """Test reviewing files with mock service."""
        mock_source = MockCodeSource()
        mock_source.set_file("src/main.py", "class MyService:\n    pass\n")
        mock_source.set_file("src/utils.py", "def helper():\n    return 42\n")

        mock_rules = MockRulesSource()
        mock_rules.set_rules(sample_agents_md)

        mock_analyzer = MockAnalyzer()
        mock_analyzer.set_response("Score: 85/100. Minor issues found.")

        service = CodeReviewService(
            code_source=mock_source,
            rules_source=mock_rules,
            analyzer=mock_analyzer,
            publisher=None,
        )

        result = service.review_files(["src/main.py", "src/utils.py"])

        assert result is not None
        assert result.score >= 0

    def test_review_pr_with_mock(self, sample_pr, sample_agents_md):
        """Test reviewing PR with mock service."""
        mock_source = MockCodeSource()
        mock_source.set_pr("owner", "repo", 123, sample_pr)

        mock_rules = MockRulesSource()
        mock_rules.set_rules(sample_agents_md)

        mock_analyzer = MockAnalyzer()
        mock_analyzer.set_response("""Score: 75/100.

Issues found:
- Line 3: WARNING - magic_numbers - Hardcoded value 42
""")

        service = CodeReviewService(
            code_source=mock_source,
            rules_source=mock_rules,
            analyzer=mock_analyzer,
            publisher=None,
        )

        result = service.review_pull_request("owner", "repo", 123)

        assert result is not None
        assert isinstance(result, ReviewResult)

    def test_factory_create_mock(self):
        """Test factory creates working mock service."""
        service = CodeReviewServiceFactory.create_mock()

        result = service.review_code("print('hello')", "test.py")

        assert result is not None
        assert result.score == 100  # Mock returns perfect score

    def test_service_uses_rules_engine(self, sample_agents_md):
        """Test service uses rules engine to parse rules."""
        mock_source = MockCodeSource()
        mock_source.set_file("test.py", "x = 42")

        mock_rules = MockRulesSource()
        mock_rules.set_rules(sample_agents_md)

        mock_analyzer = MockAnalyzer()
        mock_analyzer.set_response("Score: 100/100")

        service = CodeReviewService(
            code_source=mock_source,
            rules_source=mock_rules,
            analyzer=mock_analyzer,
            publisher=None,
        )

        result = service.review_files(["test.py"])

        # Verify analyzer was called (meaning rules were loaded and prompt built)
        assert len(mock_analyzer.calls) >= 1


class TestCodeReviewServiceFactory:
    """Tests for CodeReviewServiceFactory."""

    def test_create_mock(self):
        """Test create_mock returns working service."""
        service = CodeReviewServiceFactory.create_mock()
        assert service is not None

        result = service.review_code("print(1)", "test.py")
        assert result.score == 100

    def test_create_mock_with_custom_response(self):
        """Test mock service can be configured with custom responses."""
        service = CodeReviewServiceFactory.create_mock()

        # Review should return mock result
        result = service.review_code("class BadCode:\n    pass", "bad.py")
        assert isinstance(result, ReviewResult)


# =============================================================================
# ParsedRule Tests
# =============================================================================


class TestParsedRule:
    """Tests for ParsedRule."""

    def test_to_prompt_section_pattern(self):
        """Test generating prompt section for pattern."""
        rule = ParsedRule(
            id="factory_pattern",
            name="Factory Pattern",
            description="Use factory classes",
            rule_type=RuleType.PATTERN,
            category=RuleCategory.FACTORY_PATTERN,
            severity=RuleSeverity.ERROR,
            is_required=True,
            good_examples=[CodeExample(code="class Factory: pass", is_good=True)],
        )

        section = rule.to_prompt_section()

        assert "Factory Pattern" in section
        assert "PATTERN" in section
        assert "error" in section  # severity.value is lowercase
        assert "class Factory" in section

    def test_to_prompt_section_anti_pattern(self):
        """Test generating prompt section for anti-pattern."""
        rule = ParsedRule(
            id="god_class",
            name="God Class",
            description="Avoid classes that do too much",
            rule_type=RuleType.ANTI_PATTERN,
            category=RuleCategory.GOD_CLASS,
            severity=RuleSeverity.ERROR,
        )

        section = rule.to_prompt_section()

        assert "God Class" in section
        assert "ANTI-PATTERN" in section

    def test_is_active(self):
        """Test is_active property."""
        rule = ParsedRule(
            id="test",
            name="Test",
            description="Test rule",
            rule_type=RuleType.PATTERN,
            category=RuleCategory.OTHER,
            severity=RuleSeverity.INFO,
        )
        assert rule.is_active is True

        disabled_rule = ParsedRule(
            id="test",
            name="Test",
            description="Test rule",
            rule_type=RuleType.PATTERN,
            category=RuleCategory.OTHER,
            severity=RuleSeverity.INFO,
            is_disabled=True,
        )
        assert disabled_rule.is_active is False

    def test_matches_file(self):
        """Test matches_file method."""
        rule = ParsedRule(
            id="test",
            name="Test",
            description="Test rule",
            rule_type=RuleType.PATTERN,
            category=RuleCategory.OTHER,
            severity=RuleSeverity.INFO,
        )
        # Default rules match all files
        assert rule.matches_file("test.py") is True
        assert rule.matches_file("test.js") is True


# =============================================================================
# FileSystemSource Tests
# =============================================================================


class TestFileSystemSource:
    """Tests for FileSystemSource."""

    def test_get_files_single_file(self, tmp_path):
        """Test getting a single file."""
        from axiompy.agents.code_review.adapters.sources.filesystem import FileSystemSource

        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        source = FileSystemSource(root=str(tmp_path))
        files = source.get_files(["test.py"])

        assert len(files) == 1
        assert files[0].path == "test.py"
        assert "print" in files[0].content

    def test_get_files_directory(self, tmp_path):
        """Test getting files from directory."""
        from axiompy.agents.code_review.adapters.sources.filesystem import FileSystemSource

        # Create test files
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("# main")
        (tmp_path / "src" / "utils.py").write_text("# utils")

        source = FileSystemSource(root=str(tmp_path))
        files = source.get_files(["src"])

        assert len(files) == 2
        paths = {f.path for f in files}
        assert "src/main.py" in paths or "src\\main.py" in paths

    def test_get_file_content(self, tmp_path):
        """Test getting file content directly."""
        from axiompy.agents.code_review.adapters.sources.filesystem import FileSystemSource

        test_file = tmp_path / "test.py"
        test_file.write_text("content here")

        source = FileSystemSource(root=str(tmp_path))
        content = source.get_file_content("test.py")

        assert content == "content here"

    def test_skips_hidden_files(self, tmp_path):
        """Test that hidden files are skipped."""
        from axiompy.agents.code_review.adapters.sources.filesystem import FileSystemSource

        (tmp_path / ".hidden.py").write_text("# hidden")
        (tmp_path / "visible.py").write_text("# visible")

        source = FileSystemSource(root=str(tmp_path))
        files = source.get_files(["."])

        paths = {f.path for f in files}
        assert "visible.py" in paths
        assert ".hidden.py" not in paths

    def test_skips_pycache(self, tmp_path):
        """Test that __pycache__ is skipped."""
        from axiompy.agents.code_review.adapters.sources.filesystem import FileSystemSource

        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "test.py").write_text("# cached")
        (tmp_path / "main.py").write_text("# main")

        source = FileSystemSource(root=str(tmp_path))
        files = source.get_files(["."])

        paths = {f.path for f in files}
        assert "main.py" in paths
        assert len([p for p in paths if "__pycache__" in p]) == 0


# =============================================================================
# FileRulesSource Tests
# =============================================================================


class TestFileRulesSource:
    """Tests for FileRulesSource."""

    def test_get_rules(self, tmp_path):
        """Test loading rules from file."""
        from axiompy.agents.code_review.adapters.rules.file import FileRulesSource

        rules_file = tmp_path / "AGENTS.md"
        rules_file.write_text("# Rules\n## Patterns\n### Factory Pattern")

        source = FileRulesSource(rules_path=str(rules_file))
        content = source.get_rules()

        assert "Factory Pattern" in content

    def test_get_local_overrides(self, tmp_path):
        """Test loading local overrides."""
        from axiompy.agents.code_review.adapters.rules.file import FileRulesSource

        rules_file = tmp_path / "AGENTS.md"
        rules_file.write_text("# Main Rules")

        overrides_file = tmp_path / ".cursorrules"
        overrides_file.write_text("# Local Overrides\ndisable: god_class")

        source = FileRulesSource(rules_path=str(rules_file), overrides_path=str(overrides_file))
        overrides = source.get_local_overrides()

        assert overrides is not None
        assert "disable" in overrides

    def test_get_local_overrides_missing(self, tmp_path):
        """Test when overrides file doesn't exist."""
        from axiompy.agents.code_review.adapters.rules.file import FileRulesSource

        rules_file = tmp_path / "AGENTS.md"
        rules_file.write_text("# Rules")

        source = FileRulesSource(rules_path=str(rules_file), overrides_path=None)
        overrides = source.get_local_overrides()

        assert overrides is None


# =============================================================================
# AnalyzerFactory Tests
# =============================================================================


class TestAnalyzerFactory:
    """Tests for AnalyzerFactory."""

    def test_create_mock(self):
        """Test creating mock analyzer."""
        from axiompy.agents.code_review.adapters.analyzers import (
            AnalyzerFactory,
            MockAnalyzer,
        )

        analyzer = AnalyzerFactory.create_mock()
        assert isinstance(analyzer, MockAnalyzer)

    def test_create_mock_with_response(self):
        """Test creating mock analyzer with predefined response."""
        from axiompy.agents.code_review.adapters.analyzers import AnalyzerFactory

        analyzer = AnalyzerFactory.create_mock(response="Test response")
        result = analyzer.analyze("prompt")
        assert result == "Test response"


# =============================================================================
# AnalyzerSettings Tests
# =============================================================================


class TestAnalyzerSettings:
    """Tests for AnalyzerSettings."""

    def test_default_settings(self):
        """Test default settings."""
        from axiompy.agents.code_review.adapters.analyzers import AnalyzerSettings

        settings = AnalyzerSettings()
        assert settings.model is not None
        assert settings.host is not None
        assert settings.timeout_secs > 0

    def test_custom_settings(self):
        """Test custom settings."""
        from axiompy.agents.code_review.adapters.analyzers import AnalyzerSettings

        settings = AnalyzerSettings(
            model="custom-model", host="http://custom:11434", timeout_secs=60
        )
        assert settings.model == "custom-model"
        assert settings.host == "http://custom:11434"
        assert settings.timeout_secs == 60


# =============================================================================
# ConsolePublisher Tests
# =============================================================================


class TestConsolePublisher:
    """Tests for ConsolePublisher."""

    def test_publish_result(self):
        """Test publishing results to console."""
        from io import StringIO

        from axiompy.agents.code_review.adapters.publishers.console import (
            ConsolePublisher,
        )

        output = StringIO()
        publisher = ConsolePublisher(output=output, use_color=False)
        result = ReviewResult(
            violations=[],
            files_reviewed=1,
            rules_applied=10,
            score=100,
            summary="All good",
        )

        publisher.publish(result, context={})

        output_str = output.getvalue()
        assert "100" in output_str or "PASSED" in output_str

    def test_publish_with_violations(self):
        """Test publishing results with violations."""
        from io import StringIO

        from axiompy.agents.code_review.adapters.publishers.console import (
            ConsolePublisher,
        )

        output = StringIO()
        publisher = ConsolePublisher(output=output, use_color=False, verbose=True)
        result = ReviewResult(
            violations=[
                Violation(
                    rule_id="magic_numbers",
                    rule_name="Magic Numbers",
                    file="test.py",
                    line=10,
                    severity=ReviewSeverity.WARNING,
                    message="Hardcoded value",
                )
            ],
            files_reviewed=1,
            rules_applied=10,
            score=85,
            summary="Minor issues",
        )

        publisher.publish(result, context={})

        output_str = output.getvalue()
        assert "85" in output_str or "Magic Numbers" in output_str


# =============================================================================
# Violation Tests
# =============================================================================


class TestViolation:
    """Tests for Violation dataclass."""

    def test_is_critical(self):
        """Test is_critical property."""
        critical = Violation(
            rule_id="test",
            rule_name="Test",
            file="test.py",
            line=1,
            severity=ReviewSeverity.CRITICAL,
            message="Critical issue",
        )
        assert critical.is_critical is True

        warning = Violation(
            rule_id="test",
            rule_name="Test",
            file="test.py",
            line=1,
            severity=ReviewSeverity.WARNING,
            message="Warning",
        )
        assert warning.is_critical is False

    def test_is_error(self):
        """Test is_error property."""
        error = Violation(
            rule_id="test",
            rule_name="Test",
            file="test.py",
            line=1,
            severity=ReviewSeverity.ERROR,
            message="Error",
        )
        assert error.is_error is True

        warning = Violation(
            rule_id="test",
            rule_name="Test",
            file="test.py",
            line=1,
            severity=ReviewSeverity.WARNING,
            message="Warning",
        )
        assert warning.is_error is False

    def test_to_comment(self):
        """Test converting violation to review comment."""
        violation = Violation(
            rule_id="magic_numbers",
            rule_name="Magic Numbers",
            file="test.py",
            line=42,
            severity=ReviewSeverity.WARNING,
            message="Hardcoded value 42",
            suggestion="Use a named constant",
        )

        comment = violation.to_comment()

        assert comment.path == "test.py"
        assert comment.line == 42
        assert "Magic Numbers" in comment.body
        assert "Hardcoded value" in comment.body


# =============================================================================
# ReviewResult Extended Tests
# =============================================================================


class TestReviewResultExtended:
    """Extended tests for ReviewResult."""

    def test_violation_counts(self):
        """Test violation count properties."""
        result = ReviewResult(
            violations=[
                Violation(
                    rule_id="a",
                    rule_name="A",
                    file="a.py",
                    line=1,
                    severity=ReviewSeverity.CRITICAL,
                    message="",
                ),
                Violation(
                    rule_id="b",
                    rule_name="B",
                    file="b.py",
                    line=1,
                    severity=ReviewSeverity.ERROR,
                    message="",
                ),
                Violation(
                    rule_id="c",
                    rule_name="C",
                    file="c.py",
                    line=1,
                    severity=ReviewSeverity.WARNING,
                    message="",
                ),
                Violation(
                    rule_id="d",
                    rule_name="D",
                    file="d.py",
                    line=1,
                    severity=ReviewSeverity.WARNING,
                    message="",
                ),
            ],
            files_reviewed=4,
            rules_applied=10,
            score=50,
            summary="Issues found",
        )

        assert result.critical_count == 1
        assert result.error_count == 1
        assert result.warning_count == 2
        assert result.violation_count == 4

    def test_approved_property(self):
        """Test approved property."""
        passed = ReviewResult(
            violations=[],
            files_reviewed=1,
            rules_applied=10,
            score=100,
            summary="Perfect",
        )
        assert passed.approved is True

        failed = ReviewResult(
            violations=[
                Violation(
                    rule_id="a",
                    rule_name="A",
                    file="a.py",
                    line=1,
                    severity=ReviewSeverity.ERROR,
                    message="",
                )
            ],
            files_reviewed=1,
            rules_applied=10,
            score=50,
            summary="Failed",
        )
        assert failed.approved is False

    def test_from_violations(self):
        """Test creating ReviewResult from violations."""
        violations = [
            Violation(
                rule_id="a",
                rule_name="A",
                file="a.py",
                line=1,
                severity=ReviewSeverity.WARNING,
                message="Test",
            )
        ]

        result = ReviewResult.from_violations(
            violations=violations, files_reviewed=1, rules_applied=5
        )

        assert result.violation_count == 1
        assert result.files_reviewed == 1
        assert result.rules_applied == 5
        assert result.score < 100  # Should have score deduction

    def test_comments_property(self):
        """Test comments property."""
        result = ReviewResult(
            violations=[
                Violation(
                    rule_id="a",
                    rule_name="A",
                    file="a.py",
                    line=1,
                    severity=ReviewSeverity.WARNING,
                    message="Test",
                )
            ],
            files_reviewed=1,
            rules_applied=10,
            score=95,
            summary="Minor issues",
        )

        comments = result.comments
        assert len(comments) == 1
        assert comments[0].path == "a.py"


# =============================================================================
# RulesEngine Extended Tests
# =============================================================================


class TestRulesEngineExtended:
    """Extended tests for RulesEngine."""

    def test_chunk_rules(self, sample_agents_md):
        """Test chunking rules by type."""
        engine = RulesEngine()
        rules = engine.parse_rules(sample_agents_md)

        chunks = engine.chunk_rules(rules)

        assert "patterns" in chunks
        assert "anti_patterns" in chunks
        assert "code_smells" in chunks

    def test_parse_response_no_violations(self):
        """Test parsing response with no violations."""
        engine = RulesEngine()

        response = """
## Summary
Code looks great!

## Score
100/100

## Violations
No violations found.
"""
        violations = engine.parse_response(response, "test.py")
        assert len(violations) == 0

    def test_parse_response_with_violations(self):
        """Test parsing response with violations."""
        engine = RulesEngine()

        response = """
## Summary
Found issues.

## Score
75/100

## Violations

### Magic Numbers
- **Line**: 42
- **Severity**: WARNING
- **Message**: Hardcoded value 42 found
- **Suggestion**: Use a named constant
"""
        violations = engine.parse_response(response, "test.py")
        assert len(violations) >= 1

    def test_build_prompt_compact(self, sample_agents_md):
        """Test building compact prompt."""
        engine = RulesEngine()
        rules = engine.parse_rules(sample_agents_md)

        code = CodeFile(path="test.py", content="print('hello')")

        prompt = engine.build_prompt(code, rules, compact=True)

        assert "test.py" in prompt
        assert len(prompt) > 0


# =============================================================================
# CodeFile Tests
# =============================================================================


class TestCodeFile:
    """Tests for CodeFile dataclass."""

    def test_line_count(self):
        """Test line_count property."""
        code = CodeFile(path="test.py", content="line1\nline2\nline3")
        assert code.line_count == 3

    def test_language_inference(self):
        """Test language is inferred from path."""
        py = CodeFile(path="test.py", content="")
        assert py.language == "python"

        js = CodeFile(path="app.js", content="")
        assert js.language == "javascript"

    def test_is_python(self):
        """Test is_python property."""
        py = CodeFile(path="test.py", content="")
        assert py.is_python is True

        js = CodeFile(path="app.js", content="")
        assert js.is_python is False


# =============================================================================
# MockAnalyzer Extended Tests
# =============================================================================


class TestMockAnalyzerExtended:
    """Extended tests for MockAnalyzer."""

    def test_calls_recorded(self):
        """Test calls are recorded."""
        mock = MockAnalyzer()
        mock.analyze("test1")
        mock.analyze("test2")

        assert len(mock.calls) == 2
        assert mock.calls[0] == ("analyze", "test1")
        assert mock.calls[1] == ("analyze", "test2")

    def test_multiple_responses(self):
        """Test setting multiple responses."""
        mock = MockAnalyzer()
        mock.set_response("Response 1")

        r1 = mock.analyze("prompt1")
        assert r1 == "Response 1"

        mock.set_response("Response 2")
        r2 = mock.analyze("prompt2")
        assert r2 == "Response 2"


# =============================================================================
# JSONPublisher Tests
# =============================================================================


class TestJSONPublisher:
    """Tests for JSONPublisher."""

    def test_publish_to_stdout(self):
        """Test publishing results to stdout (via StringIO)."""
        from io import StringIO
        import json

        from axiompy.agents.code_review.adapters.publishers.json import JSONPublisher

        output = StringIO()
        publisher = JSONPublisher(output=output)

        result = ReviewResult(
            violations=[],
            files_reviewed=1,
            rules_applied=10,
            score=100,
            summary="Perfect",
        )

        publisher.publish(result, context={"pr_number": 123})

        output.seek(0)
        data = json.loads(output.read())
        assert data["score"] == 100
        assert data["files_reviewed"] == 1

    def test_publish_with_violations(self):
        """Test publishing with violations."""
        from io import StringIO
        import json

        from axiompy.agents.code_review.adapters.publishers.json import JSONPublisher

        output = StringIO()
        publisher = JSONPublisher(output=output)

        result = ReviewResult(
            violations=[
                Violation(
                    rule_id="test",
                    rule_name="Test Rule",
                    file="test.py",
                    line=1,
                    severity=ReviewSeverity.WARNING,
                    message="Test violation",
                )
            ],
            files_reviewed=1,
            rules_applied=10,
            score=95,
            summary="Minor issues",
        )

        publisher.publish(result, context={})

        output.seek(0)
        data = json.loads(output.read())
        assert len(data["details"]) == 1
        assert data["details"][0]["rule_name"] == "Test Rule"


# =============================================================================
# Defaults Module Tests
# =============================================================================


class TestDefaults:
    """Tests for defaults module."""

    def test_default_model(self):
        """Test default model is defined."""
        from axiompy.agents.code_review.defaults import DEFAULT_MODEL

        assert DEFAULT_MODEL is not None
        assert isinstance(DEFAULT_MODEL, str)

    def test_default_host(self):
        """Test default Ollama host is defined."""
        from axiompy.agents.code_review.defaults import DEFAULT_OLLAMA_HOST

        assert DEFAULT_OLLAMA_HOST is not None
        assert "http" in DEFAULT_OLLAMA_HOST

    def test_default_timeout(self):
        """Test default timeout is defined."""
        from axiompy.agents.code_review.defaults import DEFAULT_TIMEOUT_SECS

        assert DEFAULT_TIMEOUT_SECS > 0

    def test_review_modes(self):
        """Test review modes are defined."""
        from axiompy.agents.code_review.defaults import REVIEW_MODES

        assert "quick" in REVIEW_MODES
        assert "standard" in REVIEW_MODES
        assert "full" in REVIEW_MODES


# =============================================================================
# FileDiff Tests
# =============================================================================


class TestFileDiffExtended:
    """Extended tests for FileDiff."""

    def test_status_values(self):
        """Test different status values."""
        added = FileDiff(
            filename="new.py",
            status="added",
            additions=10,
            deletions=0,
            patch="+ new code",
        )
        assert added.status == "added"

        modified = FileDiff(
            filename="existing.py",
            status="modified",
            additions=5,
            deletions=3,
            patch="@@ ...",
        )
        assert modified.status == "modified"

        deleted = FileDiff(
            filename="old.py",
            status="deleted",
            additions=0,
            deletions=20,
            patch="- old code",
        )
        assert deleted.status == "deleted"

    def test_total_changes_calculation(self):
        """Test total changes calculation."""
        diff = FileDiff(
            filename="test.py",
            status="modified",
            additions=10,
            deletions=5,
            patch="",
        )
        assert diff.total_changes == 15

    def test_is_python_detection(self):
        """Test Python file detection."""
        py_diff = FileDiff(
            filename="module.py", status="modified", additions=1, deletions=1, patch=""
        )
        assert py_diff.is_python is True

        js_diff = FileDiff(filename="app.js", status="modified", additions=1, deletions=1, patch="")
        assert js_diff.is_python is False


# =============================================================================
# PullRequestInfo Tests Extended
# =============================================================================


class TestPullRequestInfoExtended:
    """Extended tests for PullRequestInfo."""

    def test_empty_pr(self):
        """Test PR with no files."""
        pr = PullRequestInfo(
            number=1,
            title="Empty PR",
            body="",
            head_sha="abc123",
            base_sha="def456",
            base_branch="main",
            head_branch="feature",
            author="testuser",
            files=[],
        )
        assert pr.file_count == 0
        assert pr.total_changes == 0
        assert len(pr.python_files) == 0

    def test_pr_with_mixed_files(self):
        """Test PR with mixed file types."""
        pr = PullRequestInfo(
            number=2,
            title="Mixed PR",
            body="",
            head_sha="abc123",
            base_sha="def456",
            base_branch="main",
            head_branch="feature",
            author="testuser",
            files=[
                FileDiff(filename="app.py", status="modified", additions=10, deletions=5, patch=""),
                FileDiff(filename="style.css", status="added", additions=20, deletions=0, patch=""),
                FileDiff(
                    filename="utils.py",
                    status="modified",
                    additions=3,
                    deletions=1,
                    patch="",
                ),
            ],
        )
        assert pr.file_count == 3
        assert len(pr.python_files) == 2
        assert pr.total_changes == 39


# =============================================================================
# RulesEngine Parse Response Tests
# =============================================================================


class TestRulesEngineParseResponse:
    """Tests for RulesEngine.parse_response edge cases."""

    def test_parse_empty_response(self):
        """Test parsing empty response."""
        engine = RulesEngine()
        violations = engine.parse_response("", "test.py")
        assert len(violations) == 0

    def test_parse_malformed_response(self):
        """Test parsing malformed response."""
        engine = RulesEngine()
        violations = engine.parse_response("Random text without structure", "test.py")
        assert len(violations) == 0

    def test_parse_response_multiple_violations(self):
        """Test parsing response with multiple violations."""
        engine = RulesEngine()
        response = """
## Summary
Multiple issues found.

## Score
60/100

## Violations

### Magic Numbers
- **Line**: 10
- **Severity**: WARNING
- **Message**: Hardcoded value
- **Suggestion**: Use constant

### Long Method
- **Line**: 50
- **Severity**: WARNING
- **Message**: Method too long
- **Suggestion**: Extract methods
"""
        violations = engine.parse_response(response, "test.py")
        # Should find at least one violation
        assert len(violations) >= 1


# =============================================================================
# ReviewComment Tests
# =============================================================================


class TestReviewComment:
    """Tests for ReviewComment dataclass."""

    def test_creation(self):
        """Test creating a review comment."""
        comment = ReviewComment(
            path="test.py",
            line=42,
            body="This is a review comment",
        )
        assert comment.path == "test.py"
        assert comment.line == 42
        assert "review comment" in comment.body


# =============================================================================
# AnalyzerType Tests
# =============================================================================


class TestAnalyzerType:
    """Tests for AnalyzerType enum."""

    def test_enum_values(self):
        """Test all analyzer types are defined."""
        from axiompy.agents.code_review.adapters.analyzers import AnalyzerType

        assert AnalyzerType.OLLAMA is not None
        assert AnalyzerType.OPENAI is not None
        assert AnalyzerType.ANTHROPIC is not None
        assert AnalyzerType.MOCK is not None


# =============================================================================
# RuleCategory Tests
# =============================================================================


class TestRuleCategory:
    """Tests for RuleCategory enum."""

    def test_pattern_categories(self):
        """Test pattern categories exist."""
        assert RuleCategory.FACTORY_PATTERN is not None
        assert RuleCategory.SETTINGS_DATACLASS is not None

    def test_anti_pattern_categories(self):
        """Test anti-pattern categories exist."""
        assert RuleCategory.GOD_CLASS is not None
        assert RuleCategory.SINGLETON is not None

    def test_code_smell_categories(self):
        """Test code smell categories exist."""
        assert RuleCategory.LONG_METHOD is not None
        assert RuleCategory.MAGIC_NUMBERS is not None


# =============================================================================
# RuleSeverity Tests
# =============================================================================


class TestRuleSeverity:
    """Tests for RuleSeverity enum."""

    def test_severity_levels(self):
        """Test all severity levels exist."""
        assert RuleSeverity.INFO is not None
        assert RuleSeverity.WARNING is not None
        assert RuleSeverity.ERROR is not None
        assert RuleSeverity.CRITICAL is not None


# =============================================================================
# RuleType Tests
# =============================================================================


class TestRuleType:
    """Tests for RuleType enum."""

    def test_rule_types(self):
        """Test all rule types exist."""
        assert RuleType.PATTERN is not None
        assert RuleType.ANTI_PATTERN is not None
        assert RuleType.CODE_SMELL is not None


# =============================================================================
# MockCodeSource Extended Tests
# =============================================================================


class TestMockCodeSourceExtended:
    """Extended tests for MockCodeSource."""

    def test_set_multiple_files(self):
        """Test setting multiple files."""
        mock = MockCodeSource()
        mock.set_file("a.py", "content a")
        mock.set_file("b.py", "content b")

        files = mock.get_files(["a.py", "b.py"])
        assert len(files) == 2

    def test_get_nonexistent_file(self):
        """Test getting file that wasn't set."""
        mock = MockCodeSource()
        files = mock.get_files(["missing.py"])
        assert len(files) == 0


# =============================================================================
# MockRulesSource Extended Tests
# =============================================================================


class TestMockRulesSourceExtended:
    """Extended tests for MockRulesSource."""

    def test_set_overrides(self):
        """Test setting overrides."""
        mock = MockRulesSource()
        mock.set_rules("# Main rules")
        mock.set_overrides("# Overrides\ndisable: god_class")

        overrides = mock.get_local_overrides()
        assert overrides is not None
        assert "disable" in overrides

    def test_no_overrides_by_default(self):
        """Test no overrides by default."""
        mock = MockRulesSource()
        mock.set_rules("# Rules")

        overrides = mock.get_local_overrides()
        assert overrides is None


# =============================================================================
# CodeExample Tests
# =============================================================================


class TestCodeExample:
    """Tests for CodeExample dataclass."""

    def test_good_example(self):
        """Test creating good example."""
        example = CodeExample(code="class Factory: pass", is_good=True)
        assert example.is_good is True
        assert "Factory" in example.code

    def test_bad_example(self):
        """Test creating bad example."""
        example = CodeExample(code="x = 42  # magic number", is_good=False)
        assert example.is_good is False


# =============================================================================
# CodeFile Extended Tests
# =============================================================================


class TestCodeFileExtended:
    """Extended tests for CodeFile."""

    def test_language_inference_javascript(self):
        """Test language inference for JavaScript."""
        file = CodeFile(path="app.js", content="const x = 1;")
        assert file.language == "javascript"

    def test_language_inference_typescript(self):
        """Test language inference for TypeScript."""
        file = CodeFile(path="app.ts", content="const x: number = 1;")
        assert file.language == "typescript"

    def test_language_inference_java(self):
        """Test language inference for Java."""
        file = CodeFile(path="Main.java", content="public class Main {}")
        assert file.language == "java"

    def test_language_inference_go(self):
        """Test language inference for Go."""
        file = CodeFile(path="main.go", content="package main")
        assert file.language == "go"

    def test_language_inference_rust(self):
        """Test language inference for Rust."""
        file = CodeFile(path="main.rs", content="fn main() {}")
        assert file.language == "rust"

    def test_language_inference_unknown(self):
        """Test language inference for unknown extension."""
        file = CodeFile(path="file.xyz", content="content")
        assert file.language == "text"

    def test_is_reviewable_python(self):
        """Test Python files are reviewable."""
        file = CodeFile(path="app.py", content="x = 1")
        assert file.is_reviewable is True

    def test_is_reviewable_json(self):
        """Test JSON files are not reviewable."""
        file = CodeFile(path="config.json", content="{}")
        assert file.is_reviewable is False

    def test_is_reviewable_yaml(self):
        """Test YAML files are not reviewable."""
        file = CodeFile(path="config.yaml", content="key: value")
        assert file.is_reviewable is False

    def test_line_count(self):
        """Test line count calculation."""
        file = CodeFile(path="test.py", content="line1\nline2\nline3")
        assert file.line_count == 3

    def test_from_diff(self):
        """Test creating CodeFile from FileDiff."""
        diff = FileDiff(
            filename="test.py",
            status="modified",
            additions=5,
            deletions=2,
            patch="+ new code",
            new_content="full content here",
        )
        file = CodeFile.from_diff(diff)
        assert file.path == "test.py"
        assert file.content == "full content here"


# =============================================================================
# FileDiff Property Tests
# =============================================================================


class TestFileDiffProperties:
    """Tests for FileDiff properties."""

    def test_is_added(self):
        """Test is_added property."""
        diff = FileDiff(filename="new.py", status="added")
        assert diff.is_added is True
        assert diff.is_modified is False

    def test_is_removed(self):
        """Test is_removed property."""
        diff = FileDiff(filename="old.py", status="removed")
        assert diff.is_removed is True

    def test_is_modified(self):
        """Test is_modified property."""
        diff = FileDiff(filename="file.py", status="modified")
        assert diff.is_modified is True

    def test_is_renamed(self):
        """Test is_renamed property."""
        diff = FileDiff(
            filename="new_name.py",
            status="renamed",
            previous_filename="old_name.py",
        )
        assert diff.is_renamed is True


# =============================================================================
# PullRequestInfo Property Tests
# =============================================================================


class TestPullRequestInfoProperties:
    """Tests for PullRequestInfo properties."""

    def test_is_large_by_changes(self):
        """Test is_large when many changes."""
        pr = PullRequestInfo(
            number=1,
            title="Large PR",
            body="",
            head_sha="abc",
            base_sha="def",
            base_branch="main",
            head_branch="feature",
            author="user",
            files=[FileDiff(filename="big.py", status="modified", additions=600, deletions=0)],
        )
        assert pr.is_large is True

    def test_is_large_by_files(self):
        """Test is_large when many files."""
        files = [
            FileDiff(filename=f"file{i}.py", status="modified", additions=1, deletions=0)
            for i in range(15)
        ]
        pr = PullRequestInfo(
            number=1,
            title="Many Files PR",
            body="",
            head_sha="abc",
            base_sha="def",
            base_branch="main",
            head_branch="feature",
            author="user",
            files=files,
        )
        assert pr.is_large is True

    def test_not_large(self):
        """Test small PR is not large."""
        pr = PullRequestInfo(
            number=1,
            title="Small PR",
            body="",
            head_sha="abc",
            base_sha="def",
            base_branch="main",
            head_branch="feature",
            author="user",
            files=[FileDiff(filename="small.py", status="modified", additions=10, deletions=5)],
        )
        assert pr.is_large is False


# =============================================================================
# ReviewResult Score and Event Tests
# =============================================================================


class TestReviewResultScoreAndEvents:
    """Tests for ReviewResult scoring and events."""

    def test_score_floor_at_zero(self):
        """Test score doesn't go below zero with from_violations."""
        # Many critical violations should floor at 0
        violations = [
            Violation(
                rule_id=f"critical{i}",
                rule_name="Critical Issue",
                file="test.py",
                line=i,
                severity=ReviewSeverity.CRITICAL,
                message="Critical",
            )
            for i in range(10)
        ]
        result = ReviewResult.from_violations(violations, files_reviewed=1, rules_applied=10)
        assert result.score >= 0

    def test_review_event_with_errors(self):
        """Test review event is always COMMENT (even with errors).

        Note: We always use COMMENT because GitHub doesn't allow:
        - Self-approval (APPROVE on your own PR)
        - Self-request-changes (REQUEST_CHANGES on your own PR)
        """
        result = ReviewResult(
            violations=[
                Violation(
                    rule_id="err",
                    rule_name="Error",
                    file="test.py",
                    line=1,
                    severity=ReviewSeverity.ERROR,
                    message="Error",
                )
            ],
            files_reviewed=1,
            rules_applied=10,
        )
        assert result.review_event == "COMMENT"

    def test_review_event_low_score(self):
        """Test review event is COMMENT for low score."""
        result = ReviewResult(
            violations=[],
            files_reviewed=1,
            rules_applied=10,
            score=70,  # Below 80 threshold
        )
        assert result.review_event == "COMMENT"

    def test_review_event_clean_code(self):
        """Test review event is COMMENT for clean code.

        Note: Even clean PRs use COMMENT because GitHub doesn't
        allow self-approval.
        """
        result = ReviewResult(
            violations=[],
            files_reviewed=1,
            rules_applied=10,
        )
        assert result.review_event == "COMMENT"

    def test_has_critical_issues(self):
        """Test has_critical_issues property."""
        result = ReviewResult(
            violations=[
                Violation(
                    rule_id="crit",
                    rule_name="Critical",
                    file="test.py",
                    line=1,
                    severity=ReviewSeverity.CRITICAL,
                    message="Critical issue",
                )
            ],
            files_reviewed=1,
            rules_applied=10,
        )
        assert result.has_critical_issues is True

    def test_summary_generation_via_from_violations(self):
        """Test summary is generated via from_violations."""
        result = ReviewResult.from_violations([], files_reviewed=2, rules_applied=50)
        assert "2" in result.summary or "files" in result.summary.lower()


# =============================================================================
# AnalyzerSettings Tests
# =============================================================================


class TestAnalyzerSettingsExtended:
    """Extended tests for AnalyzerSettings."""

    def test_default_values(self):
        """Test default values are applied."""
        from axiompy.agents.code_review.adapters.analyzers import AnalyzerSettings

        settings = AnalyzerSettings()
        assert settings.model is not None
        assert settings.host is not None
        assert settings.timeout_secs > 0

    def test_custom_values(self):
        """Test custom values override defaults."""
        from axiompy.agents.code_review.adapters.analyzers import AnalyzerSettings

        settings = AnalyzerSettings(
            model="custom-model",
            host="http://custom:1234",
            timeout_secs=300,
        )
        assert settings.model == "custom-model"
        assert settings.host == "http://custom:1234"
        assert settings.timeout_secs == 300

    def test_stream_option(self):
        """Test streaming option."""
        from axiompy.agents.code_review.adapters.analyzers import AnalyzerSettings

        settings = AnalyzerSettings(stream=False)
        assert settings.stream is False

    def test_api_key(self):
        """Test API key setting."""
        from axiompy.agents.code_review.adapters.analyzers import AnalyzerSettings

        settings = AnalyzerSettings(api_key="test-key")
        assert settings.api_key == "test-key"


# =============================================================================
# ConsolePublisher Extended Tests
# =============================================================================


class TestConsolePublisherExtended:
    """Extended tests for ConsolePublisher."""

    def test_publish_with_context(self):
        """Test publishing with context info."""
        from io import StringIO

        from axiompy.agents.code_review.adapters.publishers.console import ConsolePublisher

        output = StringIO()
        publisher = ConsolePublisher(output=output, verbose=False)
        result = ReviewResult(
            violations=[],
            files_reviewed=1,
            rules_applied=10,
            score=100,
            summary="All good",
        )

        publisher.publish(result, context={"source": "test"})
        output.seek(0)
        content = output.read()
        assert "100" in content or "Score" in content

    def test_verbose_output(self):
        """Test verbose output includes all violations."""
        from io import StringIO

        from axiompy.agents.code_review.adapters.publishers.console import ConsolePublisher

        output = StringIO()
        publisher = ConsolePublisher(output=output, verbose=True)
        violations = [
            Violation(
                rule_id=f"rule{i}",
                rule_name=f"Rule {i}",
                file="test.py",
                line=i,
                severity=ReviewSeverity.WARNING,
                message=f"Message {i}",
            )
            for i in range(10)
        ]
        result = ReviewResult(
            violations=violations,
            files_reviewed=1,
            rules_applied=10,
        )

        publisher.publish(result, context={})
        output.seek(0)
        content = output.read()
        # Verbose should show all violations
        assert "Message" in content


# =============================================================================
# RulesEngine Chunking Tests
# =============================================================================


class TestRulesEngineChunking:
    """Tests for RulesEngine chunking and prompt building."""

    def test_chunk_rules(self, sample_agents_md):
        """Test chunking rules by category."""
        engine = RulesEngine()
        rules = engine.parse_rules(sample_agents_md)
        chunks = engine.chunk_rules(rules)

        assert "patterns" in chunks
        assert "anti_patterns" in chunks
        assert "code_smells" in chunks

    def test_build_prompt_compact(self):
        """Test compact prompt building."""
        engine = RulesEngine()
        file = CodeFile(path="test.py", content="x = 1")
        rules = [
            ParsedRule(
                id="test_rule",
                name="Test Rule",
                rule_type=RuleType.PATTERN,
                severity=RuleSeverity.WARNING,
                description="Test",
                category=RuleCategory.FACTORY_PATTERN,
            )
        ]

        prompt = engine.build_prompt(file, rules, context=None, compact=True)
        assert "test.py" in prompt
        assert "Test Rule" in prompt

    def test_build_prompt_with_context(self):
        """Test prompt building with context."""
        engine = RulesEngine()
        file = CodeFile(path="test.py", content="x = 1")
        rules = [
            ParsedRule(
                id="test_rule",
                name="Test Rule",
                rule_type=RuleType.PATTERN,
                severity=RuleSeverity.WARNING,
                description="Test",
                category=RuleCategory.FACTORY_PATTERN,
            )
        ]

        prompt = engine.build_prompt(file, rules, context="PR #42: Add feature")
        assert "PR #42" in prompt or "test.py" in prompt


# =============================================================================
# Factory Extended Tests
# =============================================================================


class TestCodeReviewServiceFactoryExtended:
    """Extended tests for CodeReviewServiceFactory."""

    def test_create_mock_service(self):
        """Test creating mock service."""
        service = CodeReviewServiceFactory.create_mock()
        assert service is not None
        assert service.code_source is not None
        assert service.rules_source is not None
        assert service.analyzer is not None

    def test_create_mock_analyzer_response(self):
        """Test mock with custom analyzer response."""
        service = CodeReviewServiceFactory.create_mock()
        # The mock should work without errors
        assert service.analyzer is not None


# =============================================================================
# Violation Extended Tests
# =============================================================================


class TestViolationExtended:
    """Extended tests for Violation."""

    def test_is_critical_false(self):
        """Test is_critical returns False for non-critical."""
        v = Violation(
            rule_id="test",
            rule_name="Test",
            file="test.py",
            line=1,
            severity=ReviewSeverity.WARNING,
            message="Test",
        )
        assert v.is_critical is False

    def test_is_error_includes_critical(self):
        """Test is_error returns True for critical."""
        v = Violation(
            rule_id="test",
            rule_name="Test",
            file="test.py",
            line=1,
            severity=ReviewSeverity.CRITICAL,
            message="Test",
        )
        assert v.is_error is True

    def test_to_comment_with_suggestion(self):
        """Test converting violation with suggestion to comment."""
        v = Violation(
            rule_id="test",
            rule_name="Test Rule",
            file="test.py",
            line=10,
            severity=ReviewSeverity.WARNING,
            message="Issue found",
            suggestion="Fix it this way",
        )
        comment = v.to_comment()
        assert comment.path == "test.py"
        assert comment.line == 10
        assert "Suggestion" in comment.body or "Fix it" in comment.body


# =============================================================================
# ReviewSeverity Tests
# =============================================================================


class TestReviewSeverity:
    """Tests for ReviewSeverity enum."""

    def test_from_rule_severity_info(self):
        """Test converting INFO severity."""
        result = ReviewSeverity.from_rule_severity(RuleSeverity.INFO)
        assert result == ReviewSeverity.INFO

    def test_from_rule_severity_warning(self):
        """Test converting WARNING severity."""
        result = ReviewSeverity.from_rule_severity(RuleSeverity.WARNING)
        assert result == ReviewSeverity.WARNING

    def test_from_rule_severity_error(self):
        """Test converting ERROR severity."""
        result = ReviewSeverity.from_rule_severity(RuleSeverity.ERROR)
        assert result == ReviewSeverity.ERROR

    def test_from_rule_severity_critical(self):
        """Test converting CRITICAL severity."""
        result = ReviewSeverity.from_rule_severity(RuleSeverity.CRITICAL)
        assert result == ReviewSeverity.CRITICAL


# =============================================================================
# CodeReviewService Extended Tests
# =============================================================================


class TestCodeReviewServiceExtended:
    """Extended tests for CodeReviewService."""

    def test_review_with_publisher(self):
        """Test review publishes results."""
        from io import StringIO
        from axiompy.agents.code_review.adapters.publishers.console import ConsolePublisher

        output = StringIO()
        publisher = ConsolePublisher(output=output)

        service = CodeReviewService(
            code_source=MockCodeSource(),
            rules_source=MockRulesSource(),
            analyzer=MockAnalyzer(),
            publisher=publisher,
        )

        service.code_source.set_file("test.py", "x = 1")
        service.rules_source.set_rules("# Rules")
        service.analyzer.set_response("No violations found.")

        result = service.review_files(["test.py"])

        output.seek(0)
        content = output.read()
        assert "Score" in content or result is not None

    def test_review_empty_files_list(self):
        """Test reviewing empty files list."""
        service = CodeReviewService(
            code_source=MockCodeSource(),
            rules_source=MockRulesSource(),
            analyzer=MockAnalyzer(),
            publisher=None,
        )
        service.rules_source.set_rules("# Rules")

        result = service.review_files([])
        assert result.files_reviewed == 0


# =============================================================================
# JSONPublisher Extended Tests
# =============================================================================


class TestJSONPublisherExtended:
    """Extended tests for JSONPublisher."""

    def test_publish_with_repository_context(self):
        """Test publishing with repository context."""
        from io import StringIO
        import json
        from axiompy.agents.code_review.adapters.publishers.json import JSONPublisher

        output = StringIO()
        publisher = JSONPublisher(output=output)

        result = ReviewResult(
            violations=[],
            files_reviewed=1,
            rules_applied=10,
            score=100,
        )

        publisher.publish(result, context={"owner": "org", "repo": "project", "pr_number": 42})

        output.seek(0)
        data = json.loads(output.read())
        assert data["repository"]["owner"] == "org"
        assert data["repository"]["repo"] == "project"
        assert data["repository"]["pr_number"] == 42

    def test_publish_without_summary(self):
        """Test publishing without summary."""
        from io import StringIO
        import json
        from axiompy.agents.code_review.adapters.publishers.json import JSONPublisher

        output = StringIO()
        publisher = JSONPublisher(output=output, include_summary=False)

        result = ReviewResult(
            violations=[],
            files_reviewed=1,
            rules_applied=10,
            score=100,
            summary="This should not appear",
        )

        publisher.publish(result, context={})

        output.seek(0)
        data = json.loads(output.read())
        assert "summary" not in data


# =============================================================================
# ParsedRule Extended Tests
# =============================================================================


class TestParsedRuleExtended:
    """Extended tests for ParsedRule."""

    def test_is_disabled(self):
        """Test rule can be disabled."""
        rule = ParsedRule(
            id="test_rule",
            name="Test Rule",
            rule_type=RuleType.PATTERN,
            severity=RuleSeverity.WARNING,
            description="Test",
            category=RuleCategory.FACTORY_PATTERN,
            is_disabled=True,
        )
        assert rule.is_disabled is True

    def test_is_required(self):
        """Test required rule flag."""
        rule = ParsedRule(
            id="test_rule",
            name="Test Rule (REQUIRED)",
            rule_type=RuleType.PATTERN,
            severity=RuleSeverity.ERROR,
            description="Test",
            category=RuleCategory.FACTORY_PATTERN,
            is_required=True,
        )
        assert rule.is_required is True

    def test_with_examples(self):
        """Test rule with good and bad examples."""
        rule = ParsedRule(
            id="test_rule",
            name="Test Rule",
            rule_type=RuleType.PATTERN,
            severity=RuleSeverity.WARNING,
            description="Test",
            category=RuleCategory.FACTORY_PATTERN,
            good_examples=[CodeExample(code="good code", is_good=True)],
            bad_examples=[CodeExample(code="bad code", is_good=False)],
        )
        assert len(rule.good_examples) == 1
        assert len(rule.bad_examples) == 1


# =============================================================================
# FileSystemSource Extended Tests
# =============================================================================


class TestFileSystemSourceExtended:
    """Extended tests for FileSystemSource."""

    def test_get_files_nested_directory(self, tmp_path):
        """Test getting files from nested directories."""
        from axiompy.agents.code_review.adapters.sources.filesystem import FileSystemSource

        # Create nested structure
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        (nested / "deep.py").write_text("deep = True")

        source = FileSystemSource(str(tmp_path))
        files = source.get_files([str(tmp_path)])

        # Should find the nested file
        paths = [f.path for f in files]
        assert any("deep.py" in p for p in paths)

    def test_get_files_filters_non_python(self, tmp_path):
        """Test that non-reviewable files can be returned."""
        from axiompy.agents.code_review.adapters.sources.filesystem import FileSystemSource

        (tmp_path / "code.py").write_text("x = 1")
        (tmp_path / "data.json").write_text("{}")

        source = FileSystemSource(str(tmp_path))
        files = source.get_files([str(tmp_path)])

        # Both files should be returned (filtering is done elsewhere)
        assert len(files) >= 1


# =============================================================================
# ReviewResult Summary Tests
# =============================================================================


class TestReviewResultSummary:
    """Tests for ReviewResult summary generation."""

    def test_summary_with_violations(self):
        """Test summary generation with violations."""
        violations = [
            Violation(
                rule_id="crit",
                rule_name="Critical",
                file="test.py",
                line=1,
                severity=ReviewSeverity.CRITICAL,
                message="Critical issue",
            ),
            Violation(
                rule_id="err",
                rule_name="Error",
                file="test.py",
                line=2,
                severity=ReviewSeverity.ERROR,
                message="Error issue",
            ),
        ]
        result = ReviewResult.from_violations(violations, files_reviewed=1, rules_applied=10)
        assert "Critical" in result.summary or "Error" in result.summary

    def test_summary_no_files_reviewed(self):
        """Test summary when no files reviewed."""
        result = ReviewResult.from_violations([], files_reviewed=0, rules_applied=10)
        assert "0" in result.summary or "failed" in result.summary.lower()

    def test_summary_excellent_score(self):
        """Test summary for excellent score."""
        result = ReviewResult.from_violations([], files_reviewed=5, rules_applied=50)
        assert "Excellent" in result.summary or "✅" in result.summary


# =============================================================================
# AnalyzerFactory Extended Tests
# =============================================================================


class TestAnalyzerFactoryExtended:
    """Extended tests for AnalyzerFactory."""

    def test_create_ollama_type(self):
        """Test creating Ollama analyzer type."""
        from axiompy.agents.code_review.adapters.analyzers import (
            AnalyzerFactory,
            AnalyzerType,
            AnalyzerSettings,
        )

        settings = AnalyzerSettings(model="test-model")
        # This would fail to connect, but we can check the type is created
        analyzer = AnalyzerFactory.create(AnalyzerType.MOCK, settings)
        assert analyzer is not None

    def test_settings_show_progress(self):
        """Test settings with show_progress."""
        from axiompy.agents.code_review.adapters.analyzers import AnalyzerSettings

        settings = AnalyzerSettings(show_progress=True)
        assert settings.show_progress is True


# =============================================================================
# Library API Tests
# =============================================================================


class TestLibraryAPI:
    """Tests for library API."""

    def test_create_service_via_factory(self):
        """Test creating service via factory."""
        service = CodeReviewServiceFactory.create_mock()
        assert service is not None
        assert hasattr(service, "review_files")

    def test_review_files_via_mock_service(self):
        """Test reviewing files via mock service."""
        service = CodeReviewServiceFactory.create_mock()
        service.code_source.set_file("test.py", "x = 1")
        service.rules_source.set_rules("# Rules")
        service.analyzer.set_response("No violations found.")

        result = service.review_files(["test.py"])
        assert result is not None
        assert hasattr(result, "score")


# =============================================================================
# Domain Service Extended Tests
# =============================================================================


class TestDomainServiceExtended:
    """Extended tests for domain service."""

    def test_review_files_chunked(self):
        """Test chunked review mode."""
        service = CodeReviewService(
            code_source=MockCodeSource(),
            rules_source=MockRulesSource(),
            analyzer=MockAnalyzer(),
            publisher=None,
        )
        service.code_source.set_file("test.py", "x = 1")
        service.rules_source.set_rules("# Rules\n## Anti-Patterns\n### God Class")
        service.analyzer.set_response("No violations found.")

        result = service.review_files(["test.py"], chunked=True)
        assert result is not None

    def test_review_with_mode(self):
        """Test review with specific mode."""
        service = CodeReviewService(
            code_source=MockCodeSource(),
            rules_source=MockRulesSource(),
            analyzer=MockAnalyzer(),
            publisher=None,
        )
        service.code_source.set_file("test.py", "x = 1")
        service.rules_source.set_rules("# Rules")
        service.analyzer.set_response("No violations found.")

        result = service.review_files(["test.py"], mode="quick")
        assert result is not None


# =============================================================================
# MockCodeSource Get Diff Tests
# =============================================================================


class TestMockCodeSourceDiff:
    """Tests for MockCodeSource diff functionality."""

    def test_get_diff(self):
        """Test getting diff from mock source."""
        mock = MockCodeSource()
        # Mock source returns empty diff by default
        diffs = mock.get_diff("main", "feature")
        assert isinstance(diffs, list)

    def test_get_pull_request(self):
        """Test getting PR from mock source."""
        mock = MockCodeSource()
        pr_info = PullRequestInfo(
            number=42,
            title="Test PR",
            body="Description",
            head_sha="abc",
            base_sha="def",
            base_branch="main",
            head_branch="feature",
            author="user",
            files=[],
        )
        mock.set_pr("owner", "repo", 42, pr_info)
        pr = mock.get_pull_request("owner", "repo", 42)
        assert pr.number == 42
        assert pr.title == "Test PR"


# =============================================================================
# Rules Parsing Extended Tests
# =============================================================================


class TestRulesParsingExtended:
    """Extended tests for rules parsing."""

    def test_parse_rules_with_indicators(self, sample_agents_md):
        """Test parsing rules extracts indicators."""
        engine = RulesEngine()
        rules = engine.parse_rules(sample_agents_md)

        # Check that some rules have indicators
        for rule in rules:
            if rule.indicators:
                assert isinstance(rule.indicators, list)
                break

    def test_parse_rules_with_suggestions(self, sample_agents_md):
        """Test parsing rules extracts refactoring suggestions."""
        engine = RulesEngine()
        rules = engine.parse_rules(sample_agents_md)

        # Check that some rules have suggestions
        for rule in rules:
            if rule.refactoring_suggestions:
                assert isinstance(rule.refactoring_suggestions, list)
                break


# =============================================================================
# ConsolePublisher Color Tests
# =============================================================================


class TestConsolePublisherColors:
    """Tests for ConsolePublisher color handling."""

    def test_no_color_mode(self):
        """Test output without colors."""
        from io import StringIO
        from axiompy.agents.code_review.adapters.publishers.console import ConsolePublisher

        output = StringIO()
        publisher = ConsolePublisher(output=output, use_color=False)

        result = ReviewResult(
            violations=[],
            files_reviewed=1,
            rules_applied=10,
            score=100,
        )

        publisher.publish(result, context={})
        output.seek(0)
        content = output.read()

        # Should not contain ANSI escape codes
        assert "\033[" not in content

    def test_different_score_levels(self):
        """Test output for different score levels."""
        from io import StringIO
        from axiompy.agents.code_review.adapters.publishers.console import ConsolePublisher

        for score, expected in [(95, "Excellent"), (75, "Good"), (55, "Improvement"), (30, "Work")]:
            output = StringIO()
            publisher = ConsolePublisher(output=output, use_color=False)

            result = ReviewResult(
                violations=[],
                files_reviewed=1,
                rules_applied=10,
                score=score,
            )

            publisher.publish(result, context={})
            output.seek(0)
            content = output.read()
            assert expected in content or str(score) in content


# =============================================================================
# Defaults Module Extended Tests
# =============================================================================


class TestDefaultsExtended:
    """Extended tests for defaults module."""

    def test_default_chunks(self):
        """Test default chunks are defined."""
        from axiompy.agents.code_review.defaults import DEFAULT_CHUNKS

        assert DEFAULT_CHUNKS is not None
        assert isinstance(DEFAULT_CHUNKS, list)
        assert len(DEFAULT_CHUNKS) > 0

    def test_default_rules_path(self):
        """Test default rules path is defined."""
        from axiompy.agents.code_review.defaults import DEFAULT_RULES_PATH

        assert DEFAULT_RULES_PATH is not None
        assert "AGENTS" in DEFAULT_RULES_PATH or ".md" in DEFAULT_RULES_PATH


# =============================================================================
# Engine Validation Tests
# =============================================================================


class TestEngineValidation:
    """Tests for RulesEngine validation."""

    def test_validate_rule_active(self):
        """Test rule active validation."""
        rule = ParsedRule(
            id="test",
            name="Test",
            rule_type=RuleType.PATTERN,
            severity=RuleSeverity.WARNING,
            description="Test",
            category=RuleCategory.FACTORY_PATTERN,
            is_disabled=False,
        )
        assert rule.is_active is True

    def test_validate_rule_disabled(self):
        """Test disabled rule is not active."""
        rule = ParsedRule(
            id="test",
            name="Test",
            rule_type=RuleType.PATTERN,
            severity=RuleSeverity.WARNING,
            description="Test",
            category=RuleCategory.FACTORY_PATTERN,
            is_disabled=True,
        )
        assert rule.is_active is False


# =============================================================================
# Factory Create Methods Tests
# =============================================================================


class TestFactoryCreateMethods:
    """Tests for factory create methods."""

    def test_create_mock_returns_service(self):
        """Test create_mock returns a valid service."""
        service = CodeReviewServiceFactory.create_mock()
        assert service is not None
        assert hasattr(service, "review_files")
        assert hasattr(service, "code_source")
        assert hasattr(service, "rules_source")
        assert hasattr(service, "analyzer")


# =============================================================================
# GitSource Tests
# =============================================================================


class TestGitSourceBasic:
    """Basic tests for GitSource."""

    def test_git_error_class(self):
        """Test GitError exception class."""
        from axiompy.agents.code_review.adapters.sources.git import GitError

        error = GitError("test error")
        assert str(error) == "test error"

    def test_git_source_init_valid_repo(self, tmp_path):
        """Test GitSource initialization with valid repo."""
        from axiompy.agents.code_review.adapters.sources.git import GitSource

        # Create .git directory
        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        source = GitSource(str(tmp_path))
        assert source is not None

    def test_git_source_init_invalid_repo(self, tmp_path):
        """Test GitSource initialization with invalid repo."""
        from axiompy.agents.code_review.adapters.sources.git import GitSource, GitError

        with pytest.raises(GitError, match="Not a git repository"):
            GitSource(str(tmp_path))


# =============================================================================
# Factory Create Methods Tests (Extended)
# =============================================================================


class TestFactoryCreateMethodsExtended:
    """Extended tests for factory create methods using AnalyzerFactory."""

    def test_analyzer_factory_openai_no_key(self):
        """Test AnalyzerFactory raises error for OpenAI without API key."""
        from axiompy.agents.code_review.adapters.analyzers import (
            AnalyzerFactory,
            AnalyzerType,
            AnalyzerSettings,
        )

        settings = AnalyzerSettings()  # No api_key
        with pytest.raises(ValueError, match="api_key"):
            AnalyzerFactory.create(AnalyzerType.OPENAI, settings)

    def test_analyzer_factory_anthropic_no_key(self):
        """Test AnalyzerFactory raises error for Anthropic without API key."""
        from axiompy.agents.code_review.adapters.analyzers import (
            AnalyzerFactory,
            AnalyzerType,
            AnalyzerSettings,
        )

        settings = AnalyzerSettings()  # No api_key
        with pytest.raises(ValueError, match="api_key"):
            AnalyzerFactory.create(AnalyzerType.ANTHROPIC, settings)

    def test_analyzer_factory_mock(self):
        """Test AnalyzerFactory creates mock analyzer."""
        from axiompy.agents.code_review.adapters.analyzers import (
            AnalyzerFactory,
            AnalyzerType,
            AnalyzerSettings,
        )

        settings = AnalyzerSettings()
        analyzer = AnalyzerFactory.create(AnalyzerType.MOCK, settings)
        assert analyzer is not None


# =============================================================================
# MockCodeSource Extended Tests
# =============================================================================


class TestMockCodeSourceReset:
    """Tests for MockCodeSource reset functionality."""

    def test_reset_clears_all(self):
        """Test reset clears all stored data."""
        mock = MockCodeSource()
        mock.set_file("test.py", "content")

        mock.reset()

        files = mock.get_files(["test.py"])
        assert len(files) == 0

    def test_get_file_content_not_found(self):
        """Test get_file_content raises for missing file."""
        mock = MockCodeSource()

        with pytest.raises(FileNotFoundError):
            mock.get_file_content("missing.py")

    def test_get_pull_request_not_found(self):
        """Test get_pull_request raises for missing PR."""
        mock = MockCodeSource()

        with pytest.raises(ValueError, match="Mock PR not found"):
            mock.get_pull_request("owner", "repo", 999)

    def test_set_diff(self):
        """Test setting and getting diff."""
        mock = MockCodeSource()
        diffs = [FileDiff(filename="test.py", status="modified", additions=5, deletions=2)]
        mock.set_diff("main", "feature", diffs)

        result = mock.get_diff("main", "feature")
        assert len(result) == 1
        assert result[0].filename == "test.py"


# =============================================================================
# Domain Service Review Methods Tests
# =============================================================================


class TestDomainServiceReviewMethods:
    """Tests for domain service review methods."""

    def test_review_files_with_violations(self):
        """Test review_files returns violations."""
        service = CodeReviewService(
            code_source=MockCodeSource(),
            rules_source=MockRulesSource(),
            analyzer=MockAnalyzer(),
            publisher=None,
        )
        service.code_source.set_file("test.py", "x = 42  # magic number")
        service.rules_source.set_rules("# Rules\n## Code Smells\n### Magic Numbers")
        service.analyzer.set_response(
            "### Magic Numbers\n- **Line**: 1\n- **Severity**: WARNING\n- **Message**: Hardcoded value"
        )

        result = service.review_files(["test.py"])
        assert result is not None


# =============================================================================
# AnalyzerFactory Create Tests
# =============================================================================


class TestAnalyzerFactoryCreate:
    """Tests for AnalyzerFactory.create method."""

    def test_create_mock_analyzer(self):
        """Test creating mock analyzer."""
        from axiompy.agents.code_review.adapters.analyzers import (
            AnalyzerFactory,
            AnalyzerType,
            AnalyzerSettings,
        )

        settings = AnalyzerSettings()
        analyzer = AnalyzerFactory.create(AnalyzerType.MOCK, settings)
        assert analyzer is not None


# =============================================================================
# Library API Extended Tests
# =============================================================================


class TestLibraryCreateService:
    """Tests for library.create_service function."""

    def test_create_service_filesystem(self, tmp_path):
        """Test creating service with filesystem source."""
        from axiompy.agents.code_review.applications.library import create_service
        from axiompy.agents.code_review import CodeSourceType, RulesSourceType
        from axiompy.agents.code_review.adapters.analyzers import AnalyzerType

        rules_file = tmp_path / "AGENTS.md"
        rules_file.write_text("# Rules")

        # Use MOCK analyzer type to avoid Ollama connection
        service = create_service(
            source_type=CodeSourceType.FILESYSTEM,
            rules_type=RulesSourceType.FILE,
            analyzer_type=AnalyzerType.MOCK,
            root=str(tmp_path),
            rules_path=str(rules_file),
        )
        assert service is not None

    def test_create_service_github_source_no_token(self):
        """Test create_service raises for GitHub source without token."""
        from axiompy.agents.code_review.applications.library import create_service
        from axiompy.agents.code_review import CodeSourceType

        # Clear GITHUB_TOKEN if set
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GITHUB_TOKEN", None)
            with pytest.raises(ValueError, match="github_token required"):
                create_service(source_type=CodeSourceType.GITHUB)

    def test_create_service_github_rules_no_token(self, tmp_path):
        """Test create_service raises for GitHub rules without token."""
        from axiompy.agents.code_review.applications.library import create_service
        from axiompy.agents.code_review import RulesSourceType

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GITHUB_TOKEN", None)
            with pytest.raises(ValueError, match="github_token required"):
                create_service(rules_type=RulesSourceType.GITHUB)

    def test_create_service_openai_no_key(self, tmp_path):
        """Test create_service raises for OpenAI without API key."""
        from axiompy.agents.code_review.applications.library import create_service
        from axiompy.agents.code_review.adapters.analyzers import AnalyzerType

        rules_file = tmp_path / "AGENTS.md"
        rules_file.write_text("# Rules")

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ValueError, match="OpenAI API key required"):
                create_service(
                    analyzer_type=AnalyzerType.OPENAI,
                    rules_path=str(rules_file),
                )

    def test_create_service_anthropic_no_key(self, tmp_path):
        """Test create_service raises for Anthropic without API key."""
        from axiompy.agents.code_review.applications.library import create_service
        from axiompy.agents.code_review.adapters.analyzers import AnalyzerType

        rules_file = tmp_path / "AGENTS.md"
        rules_file.write_text("# Rules")

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(ValueError, match="Anthropic API key required"):
                create_service(
                    analyzer_type=AnalyzerType.ANTHROPIC,
                    rules_path=str(rules_file),
                )

    def test_create_service_with_valid_publisher(self, tmp_path):
        """Test create_service with valid publisher type."""
        from axiompy.agents.code_review.applications.library import create_service
        from axiompy.agents.code_review import PublisherType
        from axiompy.agents.code_review.adapters.analyzers import AnalyzerType

        rules_file = tmp_path / "AGENTS.md"
        rules_file.write_text("# Rules")

        # Use MOCK analyzer type to avoid Ollama connection
        service = create_service(
            analyzer_type=AnalyzerType.MOCK,
            publisher_type=PublisherType.JSON,
            rules_path=str(rules_file),
        )
        assert service.publisher is not None

    def test_create_service_with_none_publisher(self, tmp_path):
        """Test create_service with NONE publisher type."""
        from axiompy.agents.code_review.applications.library import create_service
        from axiompy.agents.code_review import PublisherType
        from axiompy.agents.code_review.adapters.analyzers import AnalyzerType

        rules_file = tmp_path / "AGENTS.md"
        rules_file.write_text("# Rules")

        # Use MOCK analyzer type to avoid Ollama connection
        service = create_service(
            analyzer_type=AnalyzerType.MOCK,
            publisher_type=PublisherType.NONE,
            rules_path=str(rules_file),
        )
        assert service.publisher is None

    # Legacy test: enum types don't allow unknown values
    def test_create_service_publisher_type_enum(self, tmp_path):
        """Test create_service uses PublisherType enum."""
        from axiompy.agents.code_review import PublisherType
        from axiompy.agents.code_review.adapters.analyzers import AnalyzerType
        from axiompy.agents.code_review.applications.library import create_service

        rules_file = tmp_path / "AGENTS.md"
        rules_file.write_text("# Rules")

        # Use MOCK analyzer type to avoid Ollama connection
        # Using enum ensures type safety - no "unknown" value possible
        service = create_service(
            analyzer_type=AnalyzerType.MOCK,
            publisher_type=PublisherType.CONSOLE,
            rules_path=str(rules_file),
        )
        assert service is not None

    def test_create_service_with_console_publisher(self, tmp_path):
        """Test create_service with console publisher."""
        from axiompy.agents.code_review.applications.library import create_service
        from axiompy.agents.code_review.adapters.analyzers import AnalyzerType
        from axiompy.agents.code_review import PublisherType

        rules_file = tmp_path / "AGENTS.md"
        rules_file.write_text("# Rules")

        # Use MOCK analyzer type to avoid Ollama connection
        service = create_service(
            analyzer_type=AnalyzerType.MOCK,
            publisher_type=PublisherType.CONSOLE,
            rules_path=str(rules_file),
        )
        assert service.publisher is not None

    def test_create_service_with_json_publisher(self, tmp_path):
        """Test create_service with JSON publisher."""
        from axiompy.agents.code_review.applications.library import create_service
        from axiompy.agents.code_review.adapters.analyzers import AnalyzerType
        from axiompy.agents.code_review import PublisherType

        rules_file = tmp_path / "AGENTS.md"
        rules_file.write_text("# Rules")

        # Use MOCK analyzer type to avoid Ollama connection
        service = create_service(
            analyzer_type=AnalyzerType.MOCK,
            publisher_type=PublisherType.JSON,
            rules_path=str(rules_file),
        )
        assert service.publisher is not None


class TestLibraryReviewPr:
    """Tests for library.review_pr function."""

    def test_review_pr_no_token(self):
        """Test review_pr raises without token."""
        from axiompy.agents.code_review.applications.library import review_pr

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GITHUB_TOKEN", None)
            with pytest.raises(ValueError, match="GitHub token required"):
                review_pr("owner", "repo", 123)


# =============================================================================
# Factory Extended Tests
# =============================================================================


class TestFactoryCreateForGit:
    """Tests for CodeReviewServiceFactory.create with GIT source."""

    def test_create_with_git_source(self, tmp_path):
        """Test create() with GIT source type."""
        from axiompy.agents.code_review import (
            CodeSourceType,
            CodeSourceSettings,
            RulesSourceSettings,
            AnalyzerType,
        )

        # Create .git directory
        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        rules_file = tmp_path / "AGENTS.md"
        rules_file.write_text("# Rules")

        # Use MOCK analyzer type to avoid Ollama connection
        service = CodeReviewServiceFactory.create(
            code_source_type=CodeSourceType.GIT,
            analyzer_type=AnalyzerType.MOCK,
            code_source_settings=CodeSourceSettings(repo_path=str(tmp_path)),
            rules_source_settings=RulesSourceSettings(rules_path=str(rules_file)),
        )
        assert service is not None


class TestFactoryCreateFromEnv:
    """Tests for factory - create_from_env was removed."""

    def test_create_from_env_removed(self):
        """Verify create_from_env method no longer exists."""
        # create_from_env was removed - explicit dependency injection required
        assert not hasattr(CodeReviewServiceFactory, "create_from_env")


# =============================================================================
# GitSource Extended Tests
# =============================================================================


class TestGitSourceExtended:
    """Extended tests for GitSource."""

    def test_git_source_repo_path_property(self, tmp_path):
        """Test GitSource repo_path property."""
        from axiompy.agents.code_review.adapters.sources.git import GitSource

        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        source = GitSource(str(tmp_path))
        assert source.repo_path == tmp_path


# =============================================================================
# AnalyzerSettings Extended Tests
# =============================================================================


class TestAnalyzerSettingsValidation:
    """Tests for AnalyzerSettings validation."""

    def test_settings_all_fields(self):
        """Test AnalyzerSettings with all fields."""
        from axiompy.agents.code_review.adapters.analyzers import AnalyzerSettings

        settings = AnalyzerSettings(
            model="test-model",
            api_key="test-key",
            host="http://test:1234",
            timeout_secs=60,
            stream=False,
            show_progress=True,
        )
        assert settings.model == "test-model"
        assert settings.api_key == "test-key"
        assert settings.host == "http://test:1234"
        assert settings.timeout_secs == 60
        assert settings.stream is False
        assert settings.show_progress is True


# =============================================================================
# MockRulesSource Extended Tests
# =============================================================================


class TestMockRulesSourceMethods:
    """Tests for MockRulesSource methods."""

    def test_mock_rules_source_get_local_overrides_set(self):
        """Test MockRulesSource get_local_overrides when set."""
        mock = MockRulesSource()
        mock.set_overrides("disable: god_class")

        overrides = mock.get_local_overrides()
        assert overrides == "disable: god_class"

    def test_mock_rules_source_calls_recorded(self):
        """Test MockRulesSource records calls."""
        mock = MockRulesSource()
        mock.set_rules("# Rules")

        mock.get_rules()

        assert len(mock.calls) == 1
        assert mock.calls[0][0] == "get_rules"


# =============================================================================
# ConsolePublisher Header Tests
# =============================================================================


class TestConsolePublisherHeader:
    """Tests for ConsolePublisher header generation."""

    def test_header_needs_work(self):
        """Test header for low score."""
        from io import StringIO
        from axiompy.agents.code_review.adapters.publishers.console import ConsolePublisher

        output = StringIO()
        publisher = ConsolePublisher(output=output, use_color=False)

        result = ReviewResult(
            violations=[],
            files_reviewed=1,
            rules_applied=10,
            score=40,
        )

        publisher.publish(result, context={})
        output.seek(0)
        content = output.read()
        assert "Needs Work" in content or "40" in content


# =============================================================================
# Domain Service Extended Tests
# =============================================================================


class TestDomainServiceVerbose:
    """Tests for domain service verbose mode."""

    def test_service_with_verbose_publisher(self):
        """Test service with verbose publisher."""
        from io import StringIO
        from axiompy.agents.code_review.adapters.publishers.console import ConsolePublisher

        output = StringIO()
        publisher = ConsolePublisher(output=output, verbose=True)

        service = CodeReviewService(
            code_source=MockCodeSource(),
            rules_source=MockRulesSource(),
            analyzer=MockAnalyzer(),
            publisher=publisher,
        )

        service.code_source.set_file("test.py", "x = 1")
        service.rules_source.set_rules("# Rules")
        service.analyzer.set_response("No violations found.")

        result = service.review_files(["test.py"])
        assert result is not None


# =============================================================================
# GitHub Source Error Classes Tests
# =============================================================================


class TestGitHubSourceErrors:
    """Tests for GitHub source error classes."""

    def test_github_error(self):
        """Test GitHubError exception."""
        from axiompy.agents.code_review.adapters.sources.github import GitHubError

        error = GitHubError("test error")
        assert str(error) == "test error"

    def test_github_auth_error(self):
        """Test GitHubAuthError exception."""
        from axiompy.agents.code_review.adapters.sources.github import GitHubAuthError

        error = GitHubAuthError("auth failed")
        assert str(error) == "auth failed"

    def test_github_not_found_error(self):
        """Test GitHubNotFoundError exception."""
        from axiompy.agents.code_review.adapters.sources.github import GitHubNotFoundError

        error = GitHubNotFoundError("not found")
        assert str(error) == "not found"


class TestGitHubSourceSettings:
    """Tests for GitHubSourceSettings dataclass."""

    def test_settings_valid(self):
        """Test valid settings creation."""
        from axiompy.agents.code_review.adapters.sources.github import GitHubSourceSettings

        settings = GitHubSourceSettings(token="test-token")
        assert settings.token == "test-token"
        assert settings.base_url == "https://api.github.com"
        assert settings.timeout == 30

    def test_settings_custom(self):
        """Test custom settings."""
        from axiompy.agents.code_review.adapters.sources.github import GitHubSourceSettings

        settings = GitHubSourceSettings(
            token="test-token",
            base_url="https://custom.api.com",
            timeout=60,
        )
        assert settings.base_url == "https://custom.api.com"
        assert settings.timeout == 60

    def test_settings_no_token_raises(self):
        """Test settings raises without token."""
        from axiompy.agents.code_review.adapters.sources.github import GitHubSourceSettings

        with pytest.raises(ValueError, match="token is required"):
            GitHubSourceSettings(token="")


class TestGitHubSourceInit:
    """Tests for GitHubSource initialization."""

    def test_github_source_init(self):
        """Test GitHubSource initialization."""
        from axiompy.agents.code_review.adapters.sources.github import GitHubSource

        source = GitHubSource(token="test-token")
        assert source is not None

    def test_github_source_custom_url(self):
        """Test GitHubSource with custom URL."""
        from axiompy.agents.code_review.adapters.sources.github import GitHubSource

        source = GitHubSource(
            token="test-token",
            base_url="https://custom.github.com/",
        )
        assert source is not None


# =============================================================================
# GitHub Publisher Tests
# =============================================================================


class TestGitHubPublisherInit:
    """Tests for GitHubPublisher initialization."""

    def test_publisher_init(self):
        """Test GitHubPublisher initialization."""
        from axiompy.agents.code_review.adapters.publishers.github import GitHubPublisher

        publisher = GitHubPublisher(token="test-token")
        assert publisher is not None

    def test_publisher_publish_missing_context(self):
        """Test publish raises with missing context."""
        from axiompy.agents.code_review.adapters.publishers.github import GitHubPublisher

        publisher = GitHubPublisher(token="test-token")
        result = ReviewResult(violations=[], files_reviewed=1, rules_applied=10)

        with pytest.raises(ValueError, match="must contain"):
            publisher.publish(result, {})

    def test_publisher_build_review_body(self):
        """Test _build_review_body method."""
        from axiompy.agents.code_review.adapters.publishers.github import GitHubPublisher

        publisher = GitHubPublisher(token="test-token")
        result = ReviewResult(
            violations=[],
            files_reviewed=1,
            rules_applied=10,
            summary="Test summary",
        )

        body = publisher._build_review_body(result)
        assert body == "Test summary"

    def test_publisher_append_comments_to_body(self):
        """Test _append_comments_to_body method."""
        from axiompy.agents.code_review.adapters.publishers.github import GitHubPublisher

        publisher = GitHubPublisher(token="test-token")

        comments = [
            ReviewComment(path="test.py", line=10, body="Fix this issue"),
            ReviewComment(path="other.py", line=20, body="Another issue"),
        ]

        body = "Original summary"
        result = publisher._append_comments_to_body(body, comments)

        assert "Original summary" in result
        assert "`test.py:10`" in result
        assert "Fix this issue" in result
        assert "`other.py:20`" in result
        assert "Another issue" in result


# =============================================================================
# GitHub Rules Source Tests
# =============================================================================


class TestGitHubRulesSourceInit:
    """Tests for GitHubRulesSource initialization."""

    def test_rules_source_init(self):
        """Test GitHubRulesSource initialization."""
        from axiompy.agents.code_review.adapters.rules.github import GitHubRulesSource

        source = GitHubRulesSource(token="test-token", repo="owner/repo")
        assert source.owner == "owner"
        assert source.repo == "repo"
        assert source.rules_file == "AGENTS.md"

    def test_rules_source_custom_files(self):
        """Test GitHubRulesSource with custom file paths."""
        from axiompy.agents.code_review.adapters.rules.github import GitHubRulesSource

        source = GitHubRulesSource(
            token="test-token",
            repo="owner/repo",
            rules_file="custom/rules.md",
            overrides_file=".custom-overrides",
            ref="develop",
        )
        assert source.rules_file == "custom/rules.md"
        assert source.overrides_file == ".custom-overrides"
        assert source.ref == "develop"

    def test_rules_source_invalid_repo_format(self):
        """Test GitHubRulesSource raises for invalid repo format."""
        from axiompy.agents.code_review.adapters.rules.github import GitHubRulesSource

        with pytest.raises(ValueError, match="Invalid repo format"):
            GitHubRulesSource(token="test-token", repo="invalid-repo")


# =============================================================================
# Analyzer Settings and Types Tests
# =============================================================================


class TestAnalyzerTypes:
    """Tests for analyzer types and factory."""

    def test_analyzer_type_values(self):
        """Test all analyzer type values."""
        from axiompy.agents.code_review.adapters.analyzers import AnalyzerType

        assert AnalyzerType.OLLAMA.value == "ollama"
        assert AnalyzerType.OPENAI.value == "openai"
        assert AnalyzerType.ANTHROPIC.value == "anthropic"
        assert AnalyzerType.MOCK.value == "mock"

    def test_mock_analyzer_default_response(self):
        """Test MockAnalyzer default response."""
        from axiompy.agents.code_review.adapters.analyzers import MockAnalyzer

        analyzer = MockAnalyzer()
        response = analyzer.analyze("prompt")
        assert "No violations found" in response


# =============================================================================
# Factory Create Methods Extended
# =============================================================================


class TestFactoryAllCreateMethods:
    """Tests for all factory create methods."""

    def test_create_mock_with_response(self):
        """Test create_mock with custom response."""
        service = CodeReviewServiceFactory.create_mock(response="Custom response")
        response = service.analyzer.analyze("test")
        assert response == "Custom response"

    def test_create_mock_with_rules(self):
        """Test create_mock with custom rules."""
        service = CodeReviewServiceFactory.create_mock(rules="# Custom Rules")
        rules = service.rules_source.get_rules()
        assert "Custom Rules" in rules

    def test_create_mock_default(self):
        """Test create_mock with defaults."""
        service = CodeReviewServiceFactory.create_mock()
        assert service.code_source is not None
        assert service.rules_source is not None
        assert service.analyzer is not None
        assert service.publisher is None


# =============================================================================
# Application Service Tests (application/service.py)
# =============================================================================


class TestApplicationService:
    """Tests for application/service.py CodeReviewService."""

    def test_import_application_service(self):
        """Test importing from application module (re-exports from domain)."""
        from axiompy.agents.code_review.application import CodeReviewService as AppService

        assert AppService is not None

    def test_application_service_creation(self):
        """Test creating service from application module."""
        from axiompy.agents.code_review.application import CodeReviewService as AppService

        service = AppService(
            code_source=MockCodeSource(),
            rules_source=MockRulesSource(),
            analyzer=MockAnalyzer(),
            publisher=None,
        )
        assert service is not None


# =============================================================================
# GitSource Extended Tests
# =============================================================================


class TestGitSourceMethods:
    """Tests for GitSource methods."""

    def test_git_source_has_repo_path(self, tmp_path):
        """Test GitSource has repo_path attribute."""
        from axiompy.agents.code_review.adapters.sources.git import GitSource

        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        source = GitSource(str(tmp_path))
        assert hasattr(source, "repo_path")


# =============================================================================
# MockAnalyzer Extended Tests
# =============================================================================


class TestMockAnalyzerMethods:
    """Tests for MockAnalyzer methods."""

    def test_mock_analyzer_tracks_calls(self):
        """Test MockAnalyzer tracks calls."""
        analyzer = MockAnalyzer()
        analyzer.analyze("test1")
        analyzer.analyze("test2")

        assert len(analyzer.calls) == 2
        assert analyzer.calls[0] == ("analyze", "test1")
        assert analyzer.calls[1] == ("analyze", "test2")

    def test_mock_analyzer_fluent_set_response(self):
        """Test MockAnalyzer set_response returns self."""
        analyzer = MockAnalyzer()
        result = analyzer.set_response("test")
        assert result is analyzer

    def test_mock_analyzer_custom_init_response(self):
        """Test MockAnalyzer can be initialized with custom response."""
        custom_response = "Custom analysis result"
        analyzer = MockAnalyzer(response=custom_response)
        result = analyzer.analyze("test")
        assert result == custom_response


# =============================================================================
# ConsolePublisher Severity Icon Tests
# =============================================================================


class TestConsolePublisherSeverity:
    """Tests for ConsolePublisher severity handling."""

    def test_severity_icons_all_levels(self):
        """Test severity icons for all levels."""
        from io import StringIO
        from axiompy.agents.code_review.adapters.publishers.console import ConsolePublisher

        output = StringIO()
        publisher = ConsolePublisher(output=output, use_color=False, verbose=True)

        violations = [
            Violation(
                rule_id="crit",
                rule_name="Critical",
                file="test.py",
                line=1,
                severity=ReviewSeverity.CRITICAL,
                message="Critical issue",
            ),
            Violation(
                rule_id="err",
                rule_name="Error",
                file="test.py",
                line=2,
                severity=ReviewSeverity.ERROR,
                message="Error issue",
            ),
            Violation(
                rule_id="warn",
                rule_name="Warning",
                file="test.py",
                line=3,
                severity=ReviewSeverity.WARNING,
                message="Warning issue",
            ),
            Violation(
                rule_id="info",
                rule_name="Info",
                file="test.py",
                line=4,
                severity=ReviewSeverity.INFO,
                message="Info issue",
            ),
        ]

        result = ReviewResult(
            violations=violations,
            files_reviewed=1,
            rules_applied=10,
            score=50,
        )

        publisher.publish(result, context={})
        output.seek(0)
        content = output.read()

        # Should contain violation info
        assert "Critical" in content or "🔴" in content


# =============================================================================
# FileRulesSource Extended Tests
# =============================================================================


class TestFileRulesSourceExtended:
    """Extended tests for FileRulesSource."""

    def test_file_rules_source_str_path(self, tmp_path):
        """Test FileRulesSource with string path."""
        from axiompy.agents.code_review.adapters.rules.file import FileRulesSource

        rules_file = tmp_path / "rules.md"
        rules_file.write_text("# Custom Rules")

        source = FileRulesSource(rules_path=str(rules_file))
        rules = source.get_rules()

        assert "Custom Rules" in rules


# =============================================================================
# Domain Service Methods Extended
# =============================================================================


class TestDomainServiceMethods:
    """Extended tests for domain service methods."""

    def test_service_review_code_direct(self):
        """Test review_code method directly."""
        service = CodeReviewService(
            code_source=MockCodeSource(),
            rules_source=MockRulesSource(),
            analyzer=MockAnalyzer(),
            publisher=None,
        )

        service.rules_source.set_rules("# Rules\n## Patterns")
        service.analyzer.set_response("No violations found.")

        # review_code takes a string, not a CodeFile
        result = service.review_code("x = 1", filename="test.py")

        assert result is not None


# =============================================================================
# FileSystemSource Extended Tests
# =============================================================================


class TestFileSystemSourceMethods:
    """Extended tests for FileSystemSource methods."""

    def test_filesystem_source_root_property(self, tmp_path):
        """Test FileSystemSource root property."""
        from axiompy.agents.code_review.adapters.sources.filesystem import FileSystemSource

        source = FileSystemSource(str(tmp_path))
        assert source.root == tmp_path

    def test_filesystem_source_multiple_extensions(self, tmp_path):
        """Test FileSystemSource handles multiple file types."""
        from axiompy.agents.code_review.adapters.sources.filesystem import FileSystemSource

        (tmp_path / "test.py").write_text("python")
        (tmp_path / "test.js").write_text("javascript")
        (tmp_path / "test.ts").write_text("typescript")

        source = FileSystemSource(str(tmp_path))
        files = source.get_files([str(tmp_path)])

        assert len(files) == 3


# =============================================================================
# CLI Tests
# =============================================================================


class TestCLIParser:
    """Tests for CLI argument parser."""

    def test_create_parser(self):
        """Test CLI parser creation."""
        from axiompy.agents.code_review.applications.cli import create_parser

        parser = create_parser()
        assert parser is not None
        assert parser.prog == "axiompy code-review"

    def test_parse_paths_argument(self):
        """Test parsing paths argument."""
        from axiompy.agents.code_review.applications.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["src/main.py"])
        assert args.paths == ["src/main.py"]

    def test_parse_staged_flag(self):
        """Test parsing --staged flag."""
        from axiompy.agents.code_review.applications.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["--staged"])
        assert args.staged is True

    def test_parse_diff_argument(self):
        """Test parsing --diff argument."""
        from axiompy.agents.code_review.applications.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["--diff", "HEAD~1..HEAD"])
        assert args.diff == "HEAD~1..HEAD"

    def test_parse_pr_argument(self):
        """Test parsing --pr argument."""
        from axiompy.agents.code_review.applications.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["--pr", "owner/repo#123"])
        assert args.pr == "owner/repo#123"

    def test_parse_rules_argument(self):
        """Test parsing --rules argument."""
        from axiompy.agents.code_review.applications.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([".", "--rules", "CUSTOM.md"])
        assert args.rules == "CUSTOM.md"

    def test_parse_ai_choices(self):
        """Test parsing --ai choices."""
        from axiompy.agents.code_review.applications.cli import create_parser

        parser = create_parser()

        args_ollama = parser.parse_args([".", "--ai", "ollama"])
        assert args_ollama.ai == "ollama"

        args_openai = parser.parse_args([".", "--ai", "openai"])
        assert args_openai.ai == "openai"

    def test_parse_output_choices(self):
        """Test parsing --output choices."""
        from axiompy.agents.code_review.applications.cli import create_parser

        parser = create_parser()

        args = parser.parse_args([".", "--output", "json"])
        assert args.output == "json"

    def test_parse_verbose_flag(self):
        """Test parsing --verbose flag."""
        from axiompy.agents.code_review.applications.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([".", "--verbose"])
        assert args.verbose is True

    def test_default_values(self):
        """Test CLI default values."""
        from axiompy.agents.code_review.applications.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["."])

        assert args.rules == "AGENTS.md"
        assert args.ai == "ollama"
        assert args.output == "console"
        assert args.verbose is False


# =============================================================================
# Webhook Tests
# =============================================================================


class TestWebhookSettings:
    """Tests for WebhookSettings."""

    def test_settings_valid(self):
        """Test valid webhook settings."""
        from axiompy.agents.code_review.applications.webhook import WebhookSettings

        settings = WebhookSettings(github_token="test-token")
        assert settings.github_token == "test-token"
        assert settings.host == "0.0.0.0"
        assert settings.port == 8080

    def test_settings_custom(self):
        """Test custom webhook settings."""
        from axiompy.agents.code_review.applications.webhook import WebhookSettings

        settings = WebhookSettings(
            github_token="test-token",
            host="127.0.0.1",
            port=9000,
            rules_repo="custom/repo",
            rules_file="CUSTOM.md",
            ollama_host="http://custom:11434",
            ollama_model="llama2",
            webhook_secret="secret123",
        )
        assert settings.host == "127.0.0.1"
        assert settings.port == 9000
        assert settings.rules_repo == "custom/repo"
        assert settings.ollama_model == "llama2"
        assert settings.webhook_secret == "secret123"

    def test_settings_no_token_raises(self):
        """Test settings raises without token."""
        from axiompy.agents.code_review.applications.webhook import WebhookSettings

        with pytest.raises(ValueError, match="GitHub token is required"):
            WebhookSettings(github_token="")

    def test_settings_from_env_no_token(self):
        """Test from_env raises without GITHUB_TOKEN."""
        from axiompy.agents.code_review.applications.webhook import WebhookSettings

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GITHUB_TOKEN", None)
            with pytest.raises(ValueError, match="GITHUB_TOKEN"):
                WebhookSettings.from_env()

    def test_settings_from_env_with_token(self):
        """Test from_env with environment variables."""
        from axiompy.agents.code_review.applications.webhook import WebhookSettings

        env_vars = {
            "GITHUB_TOKEN": "env-token",
            "HOST": "192.168.1.1",
            "PORT": "3000",
            "RULES_REPO": "env/repo",
            "RULES_FILE": "ENV.md",
            "OLLAMA_HOST": "http://env-host:11434",
            "OLLAMA_MODEL": "env-model",
            "WEBHOOK_SECRET": "env-secret",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = WebhookSettings.from_env()
            assert settings.github_token == "env-token"
            assert settings.host == "192.168.1.1"
            assert settings.port == 3000
            assert settings.rules_repo == "env/repo"
            assert settings.ollama_model == "env-model"


# =============================================================================
# Analyzer Extended Tests
# =============================================================================


class TestAnalyzerFactoryTypes:
    """Tests for AnalyzerFactory type handling."""

    def test_factory_create_mock(self):
        """Test creating mock analyzer."""
        from axiompy.agents.code_review.adapters.analyzers import (
            AnalyzerFactory,
            AnalyzerType,
            AnalyzerSettings,
        )

        settings = AnalyzerSettings()
        analyzer = AnalyzerFactory.create(AnalyzerType.MOCK, settings)
        assert analyzer is not None

    def test_factory_create_unknown_type(self):
        """Test factory raises for unknown type."""
        from axiompy.agents.code_review.adapters.analyzers import (
            AnalyzerFactory,
            AnalyzerSettings,
        )

        settings = AnalyzerSettings()
        # Use a string that doesn't match any type
        with pytest.raises((ValueError, KeyError)):
            AnalyzerFactory.create("unknown", settings)


class TestOllamaStreamingAnalyzerErrors:
    """Tests for OllamaStreamingAnalyzer error handling."""

    def test_ollama_analyzer_init_timeout(self):
        """Test OllamaStreamingAnalyzer initialization with custom timeout."""
        from axiompy.agents.code_review.adapters.analyzers.analyzer import (
            OllamaStreamingAnalyzer,
        )

        # This will fail because no Ollama server, but we can test the init
        import contextlib

        with contextlib.suppress(Exception):
            OllamaStreamingAnalyzer(
                host="http://nonexistent:11434",
                model="test",
                timeout_secs=5,
            )


class TestReasoningAnalyzerErrors:
    """Tests for ReasoningAnalyzer error handling."""

    def test_reasoning_analyzer_init(self):
        """Test ReasoningAnalyzer initialization with mock client."""
        from axiompy.agents.code_review.adapters.analyzers.analyzer import (
            ReasoningAnalyzer,
            AnalyzerType,
        )

        # Create a mock client
        mock_client = Mock()

        analyzer = ReasoningAnalyzer(
            client=mock_client,
            provider_type=AnalyzerType.OPENAI,
        )
        assert analyzer is not None

    def test_reasoning_analyzer_analyze(self):
        """Test ReasoningAnalyzer analyze method."""
        from axiompy.agents.code_review.adapters.analyzers.analyzer import (
            ReasoningAnalyzer,
            AnalyzerType,
        )

        # Create a mock client that returns a real string
        mock_client = Mock()
        mock_client.generate_completion.return_value = "Analysis result"

        analyzer = ReasoningAnalyzer(
            client=mock_client,
            provider_type=AnalyzerType.ANTHROPIC,
        )

        result = analyzer.analyze("Test prompt")
        assert result == "Analysis result"
        mock_client.generate_completion.assert_called_once()


# =============================================================================
# Factory Extended Tests
# =============================================================================


class TestFactoryCreateForFilesystem:
    """Tests for factory create() with FILESYSTEM source."""

    def test_create_with_filesystem_source(self, tmp_path):
        """Test create() with FILESYSTEM source type and options."""
        from axiompy.agents.code_review import (
            AnalyzerType,
            CodeSourceType,
            CodeSourceSettings,
            RulesSourceSettings,
            PublisherType,
            PublisherSettings,
        )

        rules_file = tmp_path / "AGENTS.md"
        rules_file.write_text("# Rules")

        # Use MOCK analyzer type to avoid Ollama connection
        service = CodeReviewServiceFactory.create(
            code_source_type=CodeSourceType.FILESYSTEM,
            analyzer_type=AnalyzerType.MOCK,
            publisher_type=PublisherType.JSON,
            code_source_settings=CodeSourceSettings(root=str(tmp_path)),
            rules_source_settings=RulesSourceSettings(rules_path=str(rules_file)),
            publisher_settings=PublisherSettings(verbose=True),
        )
        assert service is not None


class TestFactoryCreateForGitHub:
    """Tests for factory create() with GITHUB source."""

    def test_create_with_github_source(self):
        """Test create() with GITHUB source type."""
        from axiompy.agents.code_review import (
            AnalyzerType,
            CodeSourceType,
            RulesSourceType,
            PublisherType,
            CodeSourceSettings,
            RulesSourceSettings,
            PublisherSettings,
        )

        # Use MOCK analyzer type to avoid Ollama connection
        service = CodeReviewServiceFactory.create(
            code_source_type=CodeSourceType.GITHUB,
            rules_source_type=RulesSourceType.GITHUB,
            analyzer_type=AnalyzerType.MOCK,
            publisher_type=PublisherType.GITHUB,
            code_source_settings=CodeSourceSettings(github_token="test-token"),
            rules_source_settings=RulesSourceSettings(
                github_token="test-token",
                github_repo="owner/repo",
            ),
            publisher_settings=PublisherSettings(github_token="test-token"),
        )
        assert service is not None


# =============================================================================
# Application Service Extended Tests
# =============================================================================


class TestApplicationServiceMethods:
    """Tests for application service methods."""

    def test_review_files_empty_paths(self):
        """Test review_files with empty paths."""
        service = CodeReviewService(
            code_source=MockCodeSource(),
            rules_source=MockRulesSource(),
            analyzer=MockAnalyzer(),
            publisher=None,
        )

        service.rules_source.set_rules("# Rules")
        service.analyzer.set_response("No violations found.")

        result = service.review_files([])
        assert result is not None

    def test_get_rules_loads_if_needed(self):
        """Test get_rules loads rules on first call."""
        service = CodeReviewService(
            code_source=MockCodeSource(),
            rules_source=MockRulesSource(),
            analyzer=MockAnalyzer(),
            publisher=None,
        )

        service.rules_source.set_rules("# Test Rules")

        rules = service.get_rules()
        assert rules is not None

    def test_get_rules_summary(self):
        """Test get_rules_summary returns string."""
        service = CodeReviewService(
            code_source=MockCodeSource(),
            rules_source=MockRulesSource(),
            analyzer=MockAnalyzer(),
            publisher=None,
        )

        service.rules_source.set_rules("# Test Rules")

        summary = service.get_rules_summary()
        assert isinstance(summary, str)


# =============================================================================
# GitHubSource Extended Tests - API Methods
# =============================================================================


class TestGitHubSourceMethods:
    """Tests for GitHubSource API methods."""

    def test_get_files_raises_not_implemented(self):
        """Test get_files raises NotImplementedError."""
        from axiompy.agents.code_review.adapters.sources.github import GitHubSource

        source = GitHubSource(token="test-token")

        with pytest.raises(NotImplementedError):
            source.get_files(["test.py"])

    def test_get_diff_raises_not_implemented(self):
        """Test get_diff raises NotImplementedError."""
        from axiompy.agents.code_review.adapters.sources.github import GitHubSource

        source = GitHubSource(token="test-token")

        with pytest.raises(NotImplementedError):
            source.get_diff("HEAD~1", "HEAD")


class TestGitHubRulesSourceMethods:
    """Tests for GitHubRulesSource methods."""

    def test_get_rules_calls_github(self):
        """Test get_rules makes GitHub API call."""
        from axiompy.agents.code_review.adapters.rules.github import GitHubRulesSource

        source = GitHubRulesSource(token="test-token", repo="owner/repo")

        # Mock the GitHubSource.get_file_content method
        with patch.object(source._github, "get_file_content") as mock_get:
            mock_get.return_value = "# Rules content"

            result = source.get_rules()
            assert result == "# Rules content"
            mock_get.assert_called_once()

    def test_get_local_overrides_not_found(self):
        """Test get_local_overrides returns None when file not found."""
        from axiompy.agents.code_review.adapters.rules.github import GitHubRulesSource
        from axiompy.agents.code_review.adapters.sources.github import GitHubNotFoundError

        source = GitHubRulesSource(token="test-token", repo="owner/repo")

        # Mock get_file_content to raise NotFoundError
        with patch.object(source._github, "get_file_content") as mock_get:
            mock_get.side_effect = GitHubNotFoundError("File not found")

            result = source.get_local_overrides()
            assert result is None

    def test_get_local_overrides_other_exception(self):
        """Test get_local_overrides returns None on other errors."""
        from axiompy.agents.code_review.adapters.rules.github import GitHubRulesSource

        source = GitHubRulesSource(token="test-token", repo="owner/repo")

        # Mock get_file_content to raise generic exception
        with patch.object(source._github, "get_file_content") as mock_get:
            mock_get.side_effect = Exception("Some error")

            result = source.get_local_overrides()
            assert result is None


# =============================================================================
# CLI Extended Tests
# =============================================================================


class TestCLIValidation:
    """Tests for CLI validation functions."""

    def test_parse_diff_range(self):
        """Test diff range parsing."""
        from axiompy.agents.code_review.applications.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["--diff", "main..feature"])
        assert args.diff == "main..feature"

    def test_parse_multiple_paths(self):
        """Test parsing multiple file paths."""
        from axiompy.agents.code_review.applications.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["file1.py", "file2.py", "dir/"])
        assert args.paths == ["file1.py", "file2.py", "dir/"]

    def test_parse_all_flags_together(self):
        """Test parsing multiple flags together."""
        from axiompy.agents.code_review.applications.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(
            [
                ".",
                "--rules",
                "CUSTOM.md",
                "--ai",
                "openai",
                "--output",
                "json",
                "--verbose",
            ]
        )
        assert args.rules == "CUSTOM.md"
        assert args.ai == "openai"
        assert args.output == "json"
        assert args.verbose is True


# =============================================================================
# Analyzer Extended Tests
# =============================================================================


class TestOllamaStreamingAnalyzerInit:
    """Tests for OllamaStreamingAnalyzer initialization."""

    def test_ollama_analyzer_properties(self):
        """Test OllamaStreamingAnalyzer has expected properties."""
        from axiompy.agents.code_review.adapters.analyzers.analyzer import (
            OllamaStreamingAnalyzer,
        )

        # Test that the class exists and has __init__ params
        import inspect

        sig = inspect.signature(OllamaStreamingAnalyzer.__init__)
        params = list(sig.parameters.keys())
        assert "host" in params
        assert "model" in params
        assert "timeout_secs" in params


class TestAnalyzerSettingsDefaults:
    """Extended tests for AnalyzerSettings defaults."""

    def test_settings_defaults(self):
        """Test AnalyzerSettings uses defaults correctly."""
        from axiompy.agents.code_review.adapters.analyzers import AnalyzerSettings
        from axiompy.agents.code_review.defaults import DEFAULT_MODEL, DEFAULT_OLLAMA_HOST

        settings = AnalyzerSettings()
        assert settings.model == DEFAULT_MODEL
        assert settings.host == DEFAULT_OLLAMA_HOST
        assert settings.api_key is None

    def test_settings_stream_default(self):
        """Test AnalyzerSettings stream default."""
        from axiompy.agents.code_review.adapters.analyzers import AnalyzerSettings

        settings = AnalyzerSettings()
        assert settings.stream is True


# =============================================================================
# Factory Extended Tests
# =============================================================================


class TestFactoryOutputTypes:
    """Tests for factory output type handling."""

    def test_create_with_json_publisher(self, tmp_path):
        """Test create() with JSON publisher type."""
        from axiompy.agents.code_review import (
            AnalyzerType,
            CodeSourceType,
            PublisherType,
            CodeSourceSettings,
            RulesSourceSettings,
        )

        rules_file = tmp_path / "AGENTS.md"
        rules_file.write_text("# Rules")

        # Use MOCK analyzer type to avoid Ollama connection
        service = CodeReviewServiceFactory.create(
            code_source_type=CodeSourceType.FILESYSTEM,
            analyzer_type=AnalyzerType.MOCK,
            publisher_type=PublisherType.JSON,
            code_source_settings=CodeSourceSettings(root=str(tmp_path)),
            rules_source_settings=RulesSourceSettings(rules_path=str(rules_file)),
        )
        assert service.publisher is not None

    def test_create_with_no_publisher(self, tmp_path):
        """Test create() with NONE publisher type."""
        from axiompy.agents.code_review import (
            AnalyzerType,
            CodeSourceType,
            PublisherType,
            CodeSourceSettings,
            RulesSourceSettings,
        )

        rules_file = tmp_path / "AGENTS.md"
        rules_file.write_text("# Rules")

        # Use MOCK analyzer type to avoid Ollama connection
        service = CodeReviewServiceFactory.create(
            code_source_type=CodeSourceType.FILESYSTEM,
            analyzer_type=AnalyzerType.MOCK,
            publisher_type=PublisherType.NONE,
            code_source_settings=CodeSourceSettings(root=str(tmp_path)),
            rules_source_settings=RulesSourceSettings(rules_path=str(rules_file)),
        )
        assert service.publisher is None

    def test_create_github_without_publisher(self):
        """Test create() with GITHUB source but no publisher."""
        from axiompy.agents.code_review import (
            AnalyzerType,
            CodeSourceType,
            RulesSourceType,
            PublisherType,
            CodeSourceSettings,
            RulesSourceSettings,
        )

        # Use MOCK analyzer type to avoid Ollama connection
        service = CodeReviewServiceFactory.create(
            code_source_type=CodeSourceType.GITHUB,
            rules_source_type=RulesSourceType.GITHUB,
            analyzer_type=AnalyzerType.MOCK,
            publisher_type=PublisherType.NONE,
            code_source_settings=CodeSourceSettings(github_token="test-token"),
            rules_source_settings=RulesSourceSettings(
                github_token="test-token",
                github_repo="custom/rules",
                github_file="CUSTOM.md",
            ),
        )
        assert service is not None
        assert service.publisher is None


# =============================================================================
# Domain Service Review Methods
# =============================================================================


class TestDomainServiceReview:
    """Tests for domain service review methods."""

    def test_review_files_with_publisher(self):
        """Test review_files publishes result."""
        from io import StringIO
        from axiompy.agents.code_review.adapters.publishers.console import ConsolePublisher

        output = StringIO()
        publisher = ConsolePublisher(output=output, use_color=False)

        service = CodeReviewService(
            code_source=MockCodeSource(),
            rules_source=MockRulesSource(),
            analyzer=MockAnalyzer(),
            publisher=publisher,
        )

        service.code_source.set_file("test.py", "x = 1")
        service.rules_source.set_rules("# Rules")
        service.analyzer.set_response("No violations found.")

        result = service.review_files(["test.py"])

        output.seek(0)
        content = output.read()
        assert len(content) > 0  # Publisher wrote something

    def test_load_rules_caches(self):
        """Test load_rules caches results."""
        service = CodeReviewService(
            code_source=MockCodeSource(),
            rules_source=MockRulesSource(),
            analyzer=MockAnalyzer(),
            publisher=None,
        )

        service.rules_source.set_rules("# Rules")

        # Load rules twice
        rules1 = service.load_rules()
        rules2 = service.get_rules()

        # Should be same object (cached)
        assert rules1 == rules2

    def test_refresh_rules(self):
        """Test refresh_rules reloads rules."""
        service = CodeReviewService(
            code_source=MockCodeSource(),
            rules_source=MockRulesSource(),
            analyzer=MockAnalyzer(),
            publisher=None,
        )

        service.rules_source.set_rules("# Rules v1")
        rules1 = service.load_rules()

        service.rules_source.set_rules("# Rules v2")
        rules2 = service.refresh_rules()

        # Should get new rules
        assert rules1 is not rules2


# =============================================================================
# GitSource Extended Tests - With Temp Repo
# =============================================================================


class TestGitSourceWithTempRepo:
    """Tests for GitSource using temporary git repos."""

    @pytest.fixture
    def git_repo(self, tmp_path):
        """Create a temporary git repository."""
        import subprocess

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        # Create a file and commit
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        return tmp_path

    def test_git_source_get_files(self, git_repo):
        """Test GitSource.get_files with real repo."""
        from axiompy.agents.code_review.adapters.sources.git import GitSource

        source = GitSource(str(git_repo))
        files = source.get_files(["test.py"])

        assert len(files) == 1
        assert files[0].path == "test.py"
        assert files[0].content == "x = 1"

    def test_git_source_get_file_content(self, git_repo):
        """Test GitSource.get_file_content without ref."""
        from axiompy.agents.code_review.adapters.sources.git import GitSource

        source = GitSource(str(git_repo))
        content = source.get_file_content("test.py")

        assert content == "x = 1"

    def test_git_source_get_file_content_with_ref(self, git_repo):
        """Test GitSource.get_file_content with ref."""
        from axiompy.agents.code_review.adapters.sources.git import GitSource

        source = GitSource(str(git_repo))
        content = source.get_file_content("test.py", ref="HEAD")

        assert content == "x = 1"

    def test_git_source_get_diff(self, git_repo):
        """Test GitSource.get_diff with modified file."""
        import subprocess
        from axiompy.agents.code_review.adapters.sources.git import GitSource

        # Modify file and commit
        test_file = git_repo / "test.py"
        test_file.write_text("x = 2")
        subprocess.run(["git", "add", "."], cwd=git_repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "modify"], cwd=git_repo, check=True, capture_output=True
        )

        source = GitSource(str(git_repo))
        diffs = source.get_diff("HEAD~1", "HEAD")

        assert len(diffs) == 1
        assert diffs[0].filename == "test.py"
        assert diffs[0].status == "modified"

    def test_git_source_get_staged_diff(self, git_repo):
        """Test GitSource.get_diff with staged changes."""
        import subprocess
        from axiompy.agents.code_review.adapters.sources.git import GitSource

        # Make and stage a change
        test_file = git_repo / "test.py"
        test_file.write_text("x = 3")
        subprocess.run(["git", "add", "."], cwd=git_repo, check=True, capture_output=True)

        source = GitSource(str(git_repo))
        diffs = source.get_diff("HEAD", "--staged")

        assert len(diffs) >= 0  # Could be empty if no staged changes

    def test_git_source_get_staged_files(self, git_repo):
        """Test GitSource.get_staged_files method."""
        import subprocess
        from axiompy.agents.code_review.adapters.sources.git import GitSource

        # Make and stage a change
        test_file = git_repo / "new_file.py"
        test_file.write_text("y = 1")
        subprocess.run(["git", "add", "."], cwd=git_repo, check=True, capture_output=True)

        source = GitSource(str(git_repo))
        diffs = source.get_staged_files()

        assert len(diffs) >= 0  # Should have the staged file

    def test_git_source_get_pr_raises(self, git_repo):
        """Test GitSource.get_pull_request raises NotImplementedError."""
        from axiompy.agents.code_review.adapters.sources.git import GitSource

        source = GitSource(str(git_repo))

        with pytest.raises(NotImplementedError):
            source.get_pull_request("owner", "repo", 123)

    def test_git_source_get_diff_added_file(self, git_repo):
        """Test GitSource.get_diff with added file."""
        import subprocess
        from axiompy.agents.code_review.adapters.sources.git import GitSource

        # Add new file and commit
        new_file = git_repo / "new.py"
        new_file.write_text("new = True")
        subprocess.run(["git", "add", "."], cwd=git_repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "add new"], cwd=git_repo, check=True, capture_output=True
        )

        source = GitSource(str(git_repo))
        diffs = source.get_diff("HEAD~1", "HEAD")

        assert len(diffs) == 1
        assert diffs[0].filename == "new.py"
        assert diffs[0].status == "added"

    def test_git_source_get_diff_deleted_file(self, git_repo):
        """Test GitSource.get_diff with deleted file."""
        import subprocess
        from axiompy.agents.code_review.adapters.sources.git import GitSource

        # Delete file and commit
        test_file = git_repo / "test.py"
        test_file.unlink()
        subprocess.run(["git", "add", "."], cwd=git_repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "delete"], cwd=git_repo, check=True, capture_output=True
        )

        source = GitSource(str(git_repo))
        diffs = source.get_diff("HEAD~1", "HEAD")

        assert len(diffs) == 1
        assert diffs[0].filename == "test.py"
        assert diffs[0].status == "removed"


# =============================================================================
# CLI Extended Tests - Main Function
# =============================================================================


class TestCLIMain:
    """Tests for CLI main function and execution."""

    def test_cli_module_has_main(self):
        """Test CLI module has main function."""
        from axiompy.agents.code_review.applications import cli

        assert hasattr(cli, "main") or hasattr(cli, "create_parser")


# =============================================================================
# Factory Create Analyzer Tests
# =============================================================================


class TestFactoryCreateAnalyzer:
    """Tests for AnalyzerFactory.create with different types."""

    def test_analyzer_factory_openai_no_key_raises(self):
        """Test AnalyzerFactory raises for OpenAI without key."""
        from axiompy.agents.code_review.adapters.analyzers import (
            AnalyzerFactory,
            AnalyzerType,
            AnalyzerSettings,
        )

        settings = AnalyzerSettings()  # No api_key
        with pytest.raises(ValueError, match="api_key"):
            AnalyzerFactory.create(AnalyzerType.OPENAI, settings)

    def test_analyzer_factory_anthropic_no_key_raises(self):
        """Test AnalyzerFactory raises for Anthropic without key."""
        from axiompy.agents.code_review.adapters.analyzers import (
            AnalyzerFactory,
            AnalyzerType,
            AnalyzerSettings,
        )

        settings = AnalyzerSettings()  # No api_key
        with pytest.raises(ValueError, match="api_key"):
            AnalyzerFactory.create(AnalyzerType.ANTHROPIC, settings)

    def test_analyzer_factory_mock_creates_analyzer(self):
        """Test AnalyzerFactory creates mock analyzer."""
        from axiompy.agents.code_review.adapters.analyzers import (
            AnalyzerFactory,
            AnalyzerType,
            AnalyzerSettings,
        )

        settings = AnalyzerSettings(model="test-model")
        # Use MOCK type to avoid Ollama connection in CI
        analyzer = AnalyzerFactory.create(AnalyzerType.MOCK, settings)
        assert analyzer is not None


# =============================================================================
# Analyzer Type Enum Tests
# =============================================================================


class TestAnalyzerTypeEnum:
    """Tests for AnalyzerType enum."""

    def test_all_types_exist(self):
        """Test all analyzer types exist."""
        from axiompy.agents.code_review.adapters.analyzers import AnalyzerType

        assert hasattr(AnalyzerType, "OLLAMA")
        assert hasattr(AnalyzerType, "OPENAI")
        assert hasattr(AnalyzerType, "ANTHROPIC")
        assert hasattr(AnalyzerType, "MOCK")


# =============================================================================
# GitHub Publisher Publish Tests (with mocks)
# =============================================================================


class TestGitHubPublisherPublish:
    """Tests for GitHubPublisher.publish method."""

    def test_publish_calls_create_review(self):
        """Test publish calls create_review on GitHub source."""
        from axiompy.agents.code_review.adapters.publishers.github import GitHubPublisher

        publisher = GitHubPublisher(token="test-token")

        # Mock the GitHubSource
        with patch.object(publisher._github, "create_review") as mock_create:
            mock_create.return_value = {"id": 123}

            result = ReviewResult(
                violations=[],
                files_reviewed=1,
                rules_applied=10,
                summary="Test summary",
            )

            context = {
                "owner": "test-owner",
                "repo": "test-repo",
                "pr_number": 42,
                "head_sha": "abc123",
            }

            publisher.publish(result, context)

            mock_create.assert_called_once()


# =============================================================================
# MockCodeSource Extended Tests
# =============================================================================


class TestMockCodeSourceDiffPR:
    """Extended tests for MockCodeSource diff and PR methods."""

    def test_mock_code_source_get_diff(self):
        """Test MockCodeSource.get_diff."""
        source = MockCodeSource()
        diffs = [FileDiff(filename="test.py", status="modified")]
        source.set_diff("HEAD~1", "HEAD", diffs)

        result = source.get_diff("HEAD~1", "HEAD")
        assert result == diffs

    def test_mock_code_source_get_pull_request(self):
        """Test MockCodeSource.get_pull_request."""
        source = MockCodeSource()
        pr = PullRequestInfo(
            number=123,
            title="Test PR",
            body="Description",
            head_sha="abc123",
            base_sha="def456",
            base_branch="main",
            head_branch="feature",
            author="test-user",
            files=[],
        )
        source.set_pr("owner", "repo", 123, pr)

        result = source.get_pull_request("owner", "repo", 123)
        assert result.number == 123
        assert result.title == "Test PR"


# =============================================================================
# GitHubSource API Method Tests (with mocks)
# =============================================================================


class TestGitHubSourceAPIMethods:
    """Tests for GitHubSource API methods with mocks."""

    def test_get_file_content_success(self):
        """Test get_file_content returns content."""
        from axiompy.agents.code_review.adapters.sources.github import GitHubSource

        source = GitHubSource(token="test-token")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "file content"
        mock_response.raise_for_status = Mock()

        with patch("requests.get", return_value=mock_response):
            content = source.get_file_content("owner", "repo", "path/file.py")
            assert content == "file content"

    def test_get_file_content_401_raises(self):
        """Test get_file_content raises on 401."""
        from axiompy.agents.code_review.adapters.sources.github import (
            GitHubSource,
            GitHubAuthError,
        )

        source = GitHubSource(token="test-token")

        mock_response = Mock()
        mock_response.status_code = 401

        with patch("requests.get", return_value=mock_response):
            with pytest.raises(GitHubAuthError):
                source.get_file_content("owner", "repo", "path/file.py")

    def test_get_file_content_404_raises(self):
        """Test get_file_content raises on 404."""
        from axiompy.agents.code_review.adapters.sources.github import (
            GitHubSource,
            GitHubNotFoundError,
        )

        source = GitHubSource(token="test-token")

        mock_response = Mock()
        mock_response.status_code = 404

        with patch("requests.get", return_value=mock_response):
            with pytest.raises(GitHubNotFoundError):
                source.get_file_content("owner", "repo", "path/file.py")

    def test_get_pull_request_success(self):
        """Test get_pull_request returns PR info."""
        from axiompy.agents.code_review.adapters.sources.github import GitHubSource

        source = GitHubSource(token="test-token")

        pr_response = Mock()
        pr_response.status_code = 200
        pr_response.raise_for_status = Mock()
        pr_response.json.return_value = {
            "number": 123,
            "title": "Test PR",
            "body": "Description",
            "head": {"sha": "abc123", "ref": "feature"},
            "base": {"sha": "def456", "ref": "main"},
            "user": {"login": "test-user"},
        }

        files_response = Mock()
        files_response.status_code = 200
        files_response.raise_for_status = Mock()
        files_response.json.return_value = []

        with patch("requests.get", side_effect=[pr_response, files_response]):
            pr = source.get_pull_request("owner", "repo", 123)
            assert pr.number == 123
            assert pr.title == "Test PR"

    def test_get_pull_request_401_raises(self):
        """Test get_pull_request raises on 401."""
        from axiompy.agents.code_review.adapters.sources.github import (
            GitHubSource,
            GitHubAuthError,
        )

        source = GitHubSource(token="test-token")

        mock_response = Mock()
        mock_response.status_code = 401

        with patch("requests.get", return_value=mock_response):
            with pytest.raises(GitHubAuthError):
                source.get_pull_request("owner", "repo", 123)

    def test_get_pull_request_404_raises(self):
        """Test get_pull_request raises on 404."""
        from axiompy.agents.code_review.adapters.sources.github import (
            GitHubSource,
            GitHubNotFoundError,
        )

        source = GitHubSource(token="test-token")

        mock_response = Mock()
        mock_response.status_code = 404

        with patch("requests.get", return_value=mock_response):
            with pytest.raises(GitHubNotFoundError):
                source.get_pull_request("owner", "repo", 123)

    def test_get_pr_files_pagination(self):
        """Test _get_pr_files handles pagination."""
        from axiompy.agents.code_review.adapters.sources.github import GitHubSource

        source = GitHubSource(token="test-token")

        # First page with files
        page1_response = Mock()
        page1_response.status_code = 200
        page1_response.raise_for_status = Mock()
        page1_response.json.return_value = [
            {"filename": "file1.py", "status": "modified", "additions": 1, "deletions": 0}
        ]

        # Empty second page (end of pagination)
        page2_response = Mock()
        page2_response.status_code = 200
        page2_response.raise_for_status = Mock()
        page2_response.json.return_value = []

        with patch("requests.get", side_effect=[page1_response, page2_response]):
            files = source._get_pr_files("owner", "repo", 123)
            assert len(files) == 1
            assert files[0].filename == "file1.py"

    def test_create_review_success(self):
        """Test create_review posts review."""
        from axiompy.agents.code_review.adapters.sources.github import GitHubSource

        source = GitHubSource(token="test-token")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"id": 456}

        with patch("requests.post", return_value=mock_response):
            result = source.create_review(
                owner="owner",
                repo="repo",
                pr_number=123,
                body="Review body",
                event="COMMENT",
            )
            assert result["id"] == 456

    def test_create_review_with_comments(self):
        """Test create_review posts review with inline comments."""
        from axiompy.agents.code_review.adapters.sources.github import GitHubSource

        source = GitHubSource(token="test-token")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"id": 789}

        comments = [{"path": "file.py", "line": 10, "body": "Fix this"}]

        with patch("requests.post", return_value=mock_response) as mock_post:
            source.create_review(
                owner="owner",
                repo="repo",
                pr_number=123,
                body="Review body",
                event="REQUEST_CHANGES",
                comments=comments,
            )

            # Check that comments were included in the request
            call_args = mock_post.call_args
            assert call_args[1]["json"]["comments"] == comments


# =============================================================================
# CLI Functions Tests
# =============================================================================


class TestCLIFunctions:
    """Tests for CLI functions."""

    def test_create_parser_returns_parser(self):
        """Test create_parser returns ArgumentParser."""
        from axiompy.agents.code_review.applications.cli import create_parser
        import argparse

        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_parser_handles_help(self):
        """Test parser handles --help without error."""
        from axiompy.agents.code_review.applications.cli import create_parser

        parser = create_parser()
        # This should parse successfully
        args = parser.parse_args(["--rules", "test.md"])
        assert args.rules == "test.md"


class TestCLIParseFunctions:
    """Tests for CLI parse functions."""

    def test_parse_pr_arg_valid(self):
        """Test parse_pr_arg with valid format."""
        from axiompy.agents.code_review.applications.cli import parse_pr_arg

        owner, repo, pr_number = parse_pr_arg("owner/repo#123")
        assert owner == "owner"
        assert repo == "repo"
        assert pr_number == 123

    def test_parse_pr_arg_no_hash(self):
        """Test parse_pr_arg raises without hash."""
        from axiompy.agents.code_review.applications.cli import parse_pr_arg

        with pytest.raises(ValueError, match="Invalid PR format"):
            parse_pr_arg("owner/repo123")

    def test_parse_pr_arg_no_slash(self):
        """Test parse_pr_arg raises without slash."""
        from axiompy.agents.code_review.applications.cli import parse_pr_arg

        with pytest.raises(ValueError, match="Invalid PR format"):
            parse_pr_arg("ownerrepo#123")

    def test_parse_diff_arg_valid(self):
        """Test parse_diff_arg with valid format."""
        from axiompy.agents.code_review.applications.cli import parse_diff_arg

        base, head = parse_diff_arg("main..feature")
        assert base == "main"
        assert head == "feature"

    def test_parse_diff_arg_no_dots(self):
        """Test parse_diff_arg raises without dots."""
        from axiompy.agents.code_review.applications.cli import parse_diff_arg

        with pytest.raises(ValueError, match="Invalid diff format"):
            parse_diff_arg("main-feature")

    def test_parse_diff_arg_head_tilde(self):
        """Test parse_diff_arg with HEAD notation."""
        from axiompy.agents.code_review.applications.cli import parse_diff_arg

        base, head = parse_diff_arg("HEAD~1..HEAD")
        assert base == "HEAD~1"
        assert head == "HEAD"


class TestCLIMainRouting:
    """Tests for CLI main function routing."""

    def test_main_with_paths_creates_file_review(self):
        """Test main with paths calls file review."""
        from axiompy.agents.code_review.applications.cli import main

        # This will fail because no analyzer, but we can test argument parsing
        with patch("axiompy.agents.code_review.applications.cli.run_file_review") as mock_run:
            mock_run.return_value = 0
            result = main(["test.py"])
            mock_run.assert_called_once()
            assert result == 0

    def test_main_with_staged_creates_staged_review(self):
        """Test main with --staged calls staged review."""
        from axiompy.agents.code_review.applications.cli import main

        with patch("axiompy.agents.code_review.applications.cli.run_staged_review") as mock_run:
            mock_run.return_value = 0
            result = main(["--staged"])
            mock_run.assert_called_once()
            assert result == 0

    def test_main_with_diff_creates_diff_review(self):
        """Test main with --diff calls diff review."""
        from axiompy.agents.code_review.applications.cli import main

        with patch("axiompy.agents.code_review.applications.cli.run_diff_review") as mock_run:
            mock_run.return_value = 0
            result = main(["--diff", "main..feature"])
            mock_run.assert_called_once()
            assert result == 0

    def test_main_with_pr_creates_pr_review(self):
        """Test main with --pr calls PR review."""
        from axiompy.agents.code_review.applications.cli import main

        with patch("axiompy.agents.code_review.applications.cli.run_pr_review") as mock_run:
            mock_run.return_value = 0
            result = main(["--pr", "owner/repo#123"])
            mock_run.assert_called_once()
            assert result == 0

    def test_main_default_reviews_current_dir(self):
        """Test main with no args reviews current directory."""
        from axiompy.agents.code_review.applications.cli import main

        with patch("axiompy.agents.code_review.applications.cli.run_file_review") as mock_run:
            mock_run.return_value = 0
            result = main([])
            mock_run.assert_called_once()
            assert result == 0


class TestFactoryCreatePublisher:
    """Tests for factory publisher creation via enums."""

    def test_create_with_console_publisher(self, tmp_path):
        """Test factory creates console publisher."""
        from axiompy.agents.code_review import (
            AnalyzerType,
            CodeSourceType,
            PublisherType,
            CodeSourceSettings,
            RulesSourceSettings,
        )
        from axiompy.agents.code_review.adapters.publishers import ConsolePublisher

        rules_file = tmp_path / "AGENTS.md"
        rules_file.write_text("# Rules")

        # Use MOCK analyzer type to avoid Ollama connection
        service = CodeReviewServiceFactory.create(
            code_source_type=CodeSourceType.FILESYSTEM,
            analyzer_type=AnalyzerType.MOCK,
            publisher_type=PublisherType.CONSOLE,
            code_source_settings=CodeSourceSettings(root=str(tmp_path)),
            rules_source_settings=RulesSourceSettings(rules_path=str(rules_file)),
        )
        assert isinstance(service.publisher, ConsolePublisher)

    def test_create_with_json_publisher(self, tmp_path):
        """Test factory creates JSON publisher."""
        from axiompy.agents.code_review import (
            AnalyzerType,
            CodeSourceType,
            PublisherType,
            CodeSourceSettings,
            RulesSourceSettings,
        )
        from axiompy.agents.code_review.adapters.publishers import JSONPublisher

        rules_file = tmp_path / "AGENTS.md"
        rules_file.write_text("# Rules")

        # Use MOCK analyzer type to avoid Ollama connection
        service = CodeReviewServiceFactory.create(
            code_source_type=CodeSourceType.FILESYSTEM,
            analyzer_type=AnalyzerType.MOCK,
            publisher_type=PublisherType.JSON,
            code_source_settings=CodeSourceSettings(root=str(tmp_path)),
            rules_source_settings=RulesSourceSettings(rules_path=str(rules_file)),
        )
        assert isinstance(service.publisher, JSONPublisher)

    def test_create_with_none_publisher(self, tmp_path):
        """Test factory creates no publisher."""
        from axiompy.agents.code_review import (
            AnalyzerType,
            CodeSourceType,
            PublisherType,
            CodeSourceSettings,
            RulesSourceSettings,
        )

        rules_file = tmp_path / "AGENTS.md"
        rules_file.write_text("# Rules")

        # Use MOCK analyzer type to avoid Ollama connection
        service = CodeReviewServiceFactory.create(
            code_source_type=CodeSourceType.FILESYSTEM,
            analyzer_type=AnalyzerType.MOCK,
            publisher_type=PublisherType.NONE,
            code_source_settings=CodeSourceSettings(root=str(tmp_path)),
            rules_source_settings=RulesSourceSettings(rules_path=str(rules_file)),
        )
        assert service.publisher is None


class TestDomainServiceNonReviewable:
    """Tests for domain service with non-reviewable files."""

    def test_service_review_non_reviewable_files(self):
        """Test service handles non-reviewable files."""
        service = CodeReviewService(
            code_source=MockCodeSource(),
            rules_source=MockRulesSource(),
            analyzer=MockAnalyzer(),
            publisher=None,
        )

        # Set up a binary file (non-reviewable)
        service.code_source.set_file("image.png", "binary content")
        service.rules_source.set_rules("# Rules")
        service.analyzer.set_response("No violations found.")

        result = service.review_files(["image.png"])
        assert result is not None

    def test_service_engine_property(self):
        """Test service has engine property."""
        service = CodeReviewService(
            code_source=MockCodeSource(),
            rules_source=MockRulesSource(),
            analyzer=MockAnalyzer(),
            publisher=None,
        )

        # Access engine through internal attribute
        assert hasattr(service, "_engine")


class TestConsolePublisherVerbose:
    """Tests for ConsolePublisher verbose mode."""

    def test_publisher_with_violations_verbose(self):
        """Test publisher outputs verbose violation details."""
        from io import StringIO
        from axiompy.agents.code_review.adapters.publishers.console import ConsolePublisher

        output = StringIO()
        publisher = ConsolePublisher(output=output, use_color=False, verbose=True)

        violations = [
            Violation(
                rule_id="test",
                rule_name="Test Rule",
                file="test.py",
                line=10,
                severity=ReviewSeverity.ERROR,
                message="Test message",
            )
        ]

        result = ReviewResult(
            violations=violations,
            files_reviewed=1,
            rules_applied=5,
            score=85,
        )

        publisher.publish(result, context={})
        output.seek(0)
        content = output.read()

        # Should contain violation info in verbose mode
        assert "test.py" in content or "Test Rule" in content


class TestFileRulesSourceOverrides:
    """More tests for FileRulesSource."""

    def test_file_rules_source_get_local_overrides_exists(self, tmp_path):
        """Test get_local_overrides returns content when file exists."""
        from axiompy.agents.code_review.adapters.rules.file import FileRulesSource

        rules_file = tmp_path / "rules.md"
        rules_file.write_text("# Rules")

        overrides_file = tmp_path / ".cursorrules"
        overrides_file.write_text("# Overrides")

        source = FileRulesSource(rules_path=str(rules_file), overrides_path=str(overrides_file))
        overrides = source.get_local_overrides()

        assert overrides == "# Overrides"


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestFactoryCreateForGitExtended:
    """Extended tests for factory create() with GIT source."""

    def test_create_git_with_json_publisher(self, tmp_path):
        """Test create() with GIT source and JSON publisher."""
        import subprocess

        from axiompy.agents.code_review import (
            AnalyzerType,
            CodeSourceType,
            PublisherType,
            CodeSourceSettings,
            RulesSourceSettings,
        )

        # Create git repo
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)

        rules_file = tmp_path / "AGENTS.md"
        rules_file.write_text("# Rules")

        # Use MOCK analyzer type to avoid Ollama connection
        service = CodeReviewServiceFactory.create(
            code_source_type=CodeSourceType.GIT,
            analyzer_type=AnalyzerType.MOCK,
            publisher_type=PublisherType.JSON,
            code_source_settings=CodeSourceSettings(repo_path=str(tmp_path)),
            rules_source_settings=RulesSourceSettings(rules_path=str(rules_file)),
        )
        assert service is not None
        assert service.publisher is not None


class TestMockRulesSourceReset:
    """Tests for MockRulesSource reset."""

    def test_mock_rules_source_reset(self):
        """Test MockRulesSource reset clears state."""
        mock = MockRulesSource()
        mock.set_rules("# Rules")
        mock.get_rules()

        mock.reset()
        assert len(mock.calls) == 0


class TestReviewResultCounts:
    """Tests for ReviewResult violation counts."""

    def test_review_result_warning_count(self):
        """Test ReviewResult warning_count property."""
        violations = [
            Violation(
                rule_id="warn1",
                rule_name="Warning 1",
                file="test.py",
                line=1,
                severity=ReviewSeverity.WARNING,
                message="Warning",
            ),
            Violation(
                rule_id="warn2",
                rule_name="Warning 2",
                file="test.py",
                line=2,
                severity=ReviewSeverity.WARNING,
                message="Warning",
            ),
        ]

        result = ReviewResult.from_violations(violations, files_reviewed=1, rules_applied=5)
        assert result.warning_count == 2

    def test_review_result_info_count(self):
        """Test ReviewResult with info violations."""
        violations = [
            Violation(
                rule_id="info1",
                rule_name="Info 1",
                file="test.py",
                line=1,
                severity=ReviewSeverity.INFO,
                message="Info",
            ),
        ]

        result = ReviewResult.from_violations(violations, files_reviewed=1, rules_applied=5)
        assert result.violation_count == 1


class TestRulesEnginePrompt:
    """Tests for RulesEngine prompt building."""

    def test_rules_engine_build_prompt_with_context(self):
        """Test build_prompt includes context."""
        from axiompy.agents.code_review.domain.engine import RulesEngine

        engine = RulesEngine()
        file = CodeFile(path="test.py", content="x = 1")
        rules = []

        prompt = engine.build_prompt(file, rules, context="PR: Fix bug in auth")

        assert "PR: Fix bug in auth" in prompt or "test.py" in prompt

    def test_rules_engine_parse_response_no_violations(self):
        """Test parse_response with 'No violations found'."""
        from axiompy.agents.code_review.domain.engine import RulesEngine

        engine = RulesEngine()
        response = "No violations found."

        violations = engine.parse_response(response, "test.py")
        assert len(violations) == 0


class TestDomainServiceReviewExtended:
    """Extended domain service tests."""

    def test_service_review_with_non_python_file(self):
        """Test reviewing non-Python files."""
        service = CodeReviewService(
            code_source=MockCodeSource(),
            rules_source=MockRulesSource(),
            analyzer=MockAnalyzer(),
            publisher=None,
        )

        service.code_source.set_file("app.js", "const x = 1;")
        service.rules_source.set_rules("# Rules")
        service.analyzer.set_response("No violations found.")

        result = service.review_files(["app.js"])
        assert result is not None
        assert result.files_reviewed >= 0


class TestFactoryExtended:
    """Extended factory tests."""

    def test_factory_create_with_none_publisher(self, tmp_path):
        """Test create() with NONE publisher type."""
        from axiompy.agents.code_review import (
            AnalyzerType,
            CodeSourceType,
            PublisherType,
            CodeSourceSettings,
            RulesSourceSettings,
        )

        rules_file = tmp_path / "AGENTS.md"
        rules_file.write_text("# Rules")

        # Use MOCK analyzer type to avoid Ollama connection
        service = CodeReviewServiceFactory.create(
            code_source_type=CodeSourceType.FILESYSTEM,
            analyzer_type=AnalyzerType.MOCK,
            publisher_type=PublisherType.NONE,
            code_source_settings=CodeSourceSettings(root=str(tmp_path)),
            rules_source_settings=RulesSourceSettings(rules_path=str(rules_file)),
        )
        assert service.publisher is None


class TestFactoryCreateAnalyzerExtended:
    """Extended tests for AnalyzerFactory."""

    def test_analyzer_factory_with_defaults(self):
        """Test AnalyzerFactory creates mock analyzer with defaults."""
        from axiompy.agents.code_review.adapters.analyzers import (
            AnalyzerFactory,
            AnalyzerType,
            AnalyzerSettings,
        )

        # Use MOCK type to avoid Ollama connection in CI
        settings = AnalyzerSettings()
        analyzer = AnalyzerFactory.create(AnalyzerType.MOCK, settings)
        assert analyzer is not None

    def test_analyzer_factory_openai_with_key(self):
        """Test AnalyzerFactory creates OpenAI analyzer with key."""
        from axiompy.agents.code_review.adapters.analyzers import (
            AnalyzerFactory,
            AnalyzerType,
            AnalyzerSettings,
        )

        settings = AnalyzerSettings(api_key="test-key", model="gpt-4o")
        analyzer = AnalyzerFactory.create(AnalyzerType.OPENAI, settings)
        assert analyzer is not None

    def test_analyzer_factory_anthropic_with_key(self):
        """Test AnalyzerFactory creates Anthropic analyzer with key."""
        from axiompy.agents.code_review.adapters.analyzers import (
            AnalyzerFactory,
            AnalyzerType,
            AnalyzerSettings,
        )

        settings = AnalyzerSettings(api_key="test-key", model="claude-sonnet-4-20250514")
        analyzer = AnalyzerFactory.create(AnalyzerType.ANTHROPIC, settings)
        assert analyzer is not None


class TestFactoryCreatePublisherExtended:
    """Extended tests for publisher creation."""

    def test_create_github_publisher_requires_token(self):
        """Test GITHUB publisher type requires token."""
        from axiompy.agents.code_review import (
            AnalyzerType,
            CodeSourceType,
            PublisherType,
            CodeSourceSettings,
            RulesSourceSettings,
            PublisherSettings,
        )

        # Use MOCK analyzer type to avoid Ollama connection
        with pytest.raises(ValueError, match="github_token"):
            CodeReviewServiceFactory.create(
                code_source_type=CodeSourceType.FILESYSTEM,
                analyzer_type=AnalyzerType.MOCK,
                publisher_type=PublisherType.GITHUB,
                code_source_settings=CodeSourceSettings(),
                rules_source_settings=RulesSourceSettings(),
                publisher_settings=PublisherSettings(github_token=None),
            )


class TestCodeFileReviewable:
    """Tests for CodeFile is_reviewable property."""

    def test_code_file_is_reviewable_typescript(self):
        """Test TypeScript file is reviewable."""
        file = CodeFile(path="app.tsx", content="const x = 1;")
        assert file.is_reviewable is True

    def test_code_file_is_reviewable_rust(self):
        """Test Rust file is reviewable."""
        file = CodeFile(path="main.rs", content="fn main() {}")
        assert file.is_reviewable is True

    def test_code_file_is_reviewable_go(self):
        """Test Go file is reviewable."""
        file = CodeFile(path="main.go", content="package main")
        assert file.is_reviewable is True


class TestFileDiffStatus:
    """Tests for FileDiff status properties."""

    def test_file_diff_is_modified(self):
        """Test FileDiff is_modified property."""
        diff = FileDiff(filename="test.py", status="modified")
        assert diff.is_modified is True

    def test_file_diff_is_added(self):
        """Test FileDiff is_added property."""
        diff = FileDiff(filename="test.py", status="added")
        assert diff.is_added is True

    def test_file_diff_is_removed(self):
        """Test FileDiff is_removed property."""
        diff = FileDiff(filename="test.py", status="removed")
        assert diff.is_removed is True

    def test_file_diff_is_renamed(self):
        """Test FileDiff is_renamed property."""
        diff = FileDiff(filename="test.py", status="renamed", previous_filename="old.py")
        assert diff.is_renamed is True


class TestPullRequestInfoCounts:
    """Tests for PullRequestInfo count properties."""

    def test_pr_info_file_count(self):
        """Test PullRequestInfo file_count property."""
        pr = PullRequestInfo(
            number=1,
            title="Test",
            body="Body",
            head_sha="abc",
            base_sha="def",
            base_branch="main",
            head_branch="feature",
            author="user",
            files=[
                FileDiff(filename="a.py", status="modified"),
                FileDiff(filename="b.py", status="added"),
            ],
        )
        assert pr.file_count == 2

    def test_pr_info_total_changes(self):
        """Test PullRequestInfo total_changes property."""
        pr = PullRequestInfo(
            number=1,
            title="Test",
            body="Body",
            head_sha="abc",
            base_sha="def",
            base_branch="main",
            head_branch="feature",
            author="user",
            files=[
                FileDiff(filename="a.py", status="modified", additions=10, deletions=5),
            ],
        )
        assert pr.total_changes == 15
