"""Application Service - Main use case orchestration.

The CodeReviewService orchestrates the code review process:
1. Get code from source
2. Load rules
3. Build prompt and analyze with AI
4. Parse response into violations
5. Publish results

It depends ONLY on ports (protocols), enabling flexible dependency injection.
"""
# pylint: disable=assignment-from-no-return,not-an-iterable
# Note: These are false positives - Protocol methods have return types but pylint doesn't see them

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from .engine import RulesEngine
from .models import CodeFile, PullRequestInfo
from .ports import AIAnalyzer, CodeSource, ReviewPublisher, RulesSource
from .results import ReviewResult, Violation
from .rules import ParsedRule

logger = logging.getLogger(__name__)


@dataclass
class CodeReviewService:
    """
    Core service for code review.

    Orchestrates the review process using injected dependencies.
    All dependencies are ports (protocols), not concrete implementations.

    Example:
        service = CodeReviewService(
            code_source=FileSystemSource("."),
            rules_source=FileRulesSource("AGENTS.md"),
            analyzer=OllamaAnalyzer(),
            publisher=ConsolePublisher(),
        )

        result = service.review_files(["src/main.py"])
    """

    code_source: CodeSource
    rules_source: RulesSource
    analyzer: AIAnalyzer
    publisher: Optional[ReviewPublisher] = None

    # Internal state
    _engine: RulesEngine = field(default_factory=RulesEngine, init=False)
    _rules: List[ParsedRule] = field(default_factory=list, init=False)
    _rules_loaded: bool = field(default=False, init=False)

    def load_rules(self) -> List[ParsedRule]:
        """
        Load rules from source.

        Rules are cached and can be refreshed with refresh_rules().

        Returns:
            List of parsed rules
        """
        logger.info("Loading rules from source")

        content = self.rules_source.get_rules()
        overrides = self.rules_source.get_local_overrides()

        self._rules = self._engine.parse_rules(content, overrides)
        self._rules_loaded = True

        active = [r for r in self._rules if r.is_active]
        logger.info(f"Loaded {len(active)} active rules")

        return self._rules

    def refresh_rules(self) -> List[ParsedRule]:
        """Force reload rules from source."""
        self._rules_loaded = False
        return self.load_rules()

    def get_rules(self) -> List[ParsedRule]:
        """Get current rules, loading if necessary."""
        if not self._rules_loaded:
            self.load_rules()
        return self._rules

    def review_files(
        self,
        paths: List[str],
        chunked: bool = True,
        mode: Optional[str] = None,
    ) -> ReviewResult:
        """
        Review specific files.

        Use case: CLI reviewing local files.

        Args:
            paths: List of file paths to review
            chunked: Use chunked review for faster processing (default: True)
            mode: Review mode - "quick", "standard", or "full" (default: standard)

        Returns:
            ReviewResult with violations and score

        Example:
            result = service.review_files(["src/main.py", "src/utils.py"])
            result = service.review_files(["src/main.py"], mode="quick")  # Fast
            print(f"Score: {result.score}")
        """
        logger.info(f"Reviewing {len(paths)} files (chunked={chunked}, mode={mode})")

        # Use chunked review by default for speed
        if chunked:
            return self.review_files_chunked(paths, mode=mode)

        # Full review (all rules at once) - slower but more thorough
        rules = self.get_rules()
        files = self.code_source.get_files(paths)
        result = self._review_files(files, rules)

        # Publish if configured
        if self.publisher:
            self.publisher.publish(result, {"paths": paths})

        return result

    def review_diff(self, base: str = "HEAD~1", head: str = "HEAD") -> ReviewResult:
        """
        Review changes between two git refs.

        Use case: CLI reviewing staged changes or commits.

        Args:
            base: Base ref (default: HEAD~1)
            head: Head ref (default: HEAD)

        Returns:
            ReviewResult with violations and score

        Example:
            # Review last commit
            result = service.review_diff("HEAD~1", "HEAD")

            # Review staged changes
            result = service.review_diff("HEAD", "staged")
        """
        logger.info(f"Reviewing diff {base}..{head}")

        rules = self.get_rules()

        # Get diffs from source
        diffs = self.code_source.get_diff(base, head)

        # Convert to CodeFiles
        files = [CodeFile.from_diff(d) for d in diffs]

        # Run review
        result = self._review_files(files, rules)

        if self.publisher:
            self.publisher.publish(result, {"base": base, "head": head})

        return result

    def review_pull_request(
        self,
        owner: str,
        repo: str,
        pr_number: int,
    ) -> ReviewResult:
        """
        Review a GitHub pull request.

        Use case: Webhook or GitHub Action.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: PR number

        Returns:
            ReviewResult with violations and score

        Example:
            result = service.review_pull_request("owner", "repo", 123)
            if result.has_errors:
                print("PR has issues!")
        """
        logger.info(f"Reviewing PR {owner}/{repo}#{pr_number}")

        rules = self.get_rules()

        # Get PR info from source
        pr = self.code_source.get_pull_request(owner, repo, pr_number)

        # Warn about large PRs
        if pr.is_large:
            logger.warning(f"Large PR: {pr.file_count} files, {pr.total_changes} lines changed")

        # Convert to CodeFiles
        files = [CodeFile.from_diff(f) for f in pr.files]

        # Run review with PR context
        context = f"PR Title: {pr.title}\n\nPR Description:\n{pr.body}"
        result = self._review_files(files, rules, context=context)

        # Publish if configured
        if self.publisher:
            self.publisher.publish(
                result,
                {
                    "owner": owner,
                    "repo": repo,
                    "pr_number": pr_number,
                    "head_sha": pr.head_sha,
                },
            )

        return result

    def review_code(self, code: str, filename: str = "code.py") -> ReviewResult:
        """
        Review a code string directly.

        Use case: Quick review of code snippet.

        Args:
            code: Code content to review
            filename: Virtual filename (for language detection)

        Returns:
            ReviewResult with violations and score
        """
        logger.info(f"Reviewing code snippet ({len(code)} chars)")

        rules = self.get_rules()
        file = CodeFile(path=filename, content=code)

        return self._review_files([file], rules)

    def review_files_chunked(
        self,
        paths: List[str],
        chunks: Optional[List[str]] = None,
        mode: Optional[str] = None,
    ) -> ReviewResult:
        """
        Review files in chunks to avoid prompt size issues.

        Splits rules into categories and reviews each separately,
        then merges the results. This keeps prompts small and fast.

        Args:
            paths: List of file paths to review
            chunks: Which chunks to run (overrides mode if provided)
                   Options: "patterns", "anti_patterns", "code_smells"
            mode: Review mode - "quick", "standard", or "full"
                  quick: anti_patterns only (fastest)
                  standard: anti_patterns + code_smells (default)
                  full: all chunks

        Returns:
            Merged ReviewResult from all chunks

        Example:
            result = service.review_files_chunked(["src/main.py"], mode="quick")
            result = service.review_files_chunked(["src/main.py"], mode="standard")
            result = service.review_files_chunked(["src/main.py"], mode="full")
        """
        from axiompy.agents.code_review.defaults import DEFAULT_REVIEW_MODE, REVIEW_MODES

        # Determine chunks from mode or explicit list
        if chunks is None:
            effective_mode = mode or DEFAULT_REVIEW_MODE
            chunks = REVIEW_MODES.get(effective_mode, REVIEW_MODES[DEFAULT_REVIEW_MODE])
            logger.info(f"Using review mode: {effective_mode}")

        logger.info(f"Chunked review of {len(paths)} files with chunks: {chunks}")

        # Load rules and chunk them
        rules = self.get_rules()
        rule_chunks = self._engine.chunk_rules(rules)

        # Get files once
        files = self.code_source.get_files(paths)

        # Review each chunk and collect violations
        all_violations: List[Violation] = []
        files_reviewed = 0
        rules_applied = 0

        for chunk_name in chunks:
            if chunk_name not in rule_chunks:
                logger.warning(f"Unknown chunk: {chunk_name}")
                continue

            chunk_rules = rule_chunks[chunk_name]
            if not chunk_rules:
                continue

            logger.info(f"Reviewing with {chunk_name} ({len(chunk_rules)} rules)")

            # Review this chunk
            chunk_result = self._review_files(files, chunk_rules)

            # Collect results
            all_violations.extend(chunk_result.violations)
            files_reviewed = max(files_reviewed, chunk_result.files_reviewed)
            rules_applied += len(chunk_rules)

        # Merge into final result
        result = ReviewResult.from_violations(
            violations=all_violations,
            files_reviewed=files_reviewed,
            rules_applied=rules_applied,
        )

        # Publish if configured
        if self.publisher:
            self.publisher.publish(result, {"paths": paths, "chunks": chunks})

        return result

    def _review_files(
        self,
        files: List[CodeFile],
        rules: List[ParsedRule],
        context: Optional[str] = None,
    ) -> ReviewResult:
        """
        Internal: Run review on a list of files.

        Args:
            files: Files to review
            rules: Rules to apply
            context: Optional context (PR description, etc.)

        Returns:
            ReviewResult
        """
        all_violations: List[Violation] = []
        reviewed_count = 0

        # Filter to reviewable files
        reviewable = [f for f in files if f.is_reviewable]

        if not reviewable:
            logger.info("No reviewable files found")
            return ReviewResult.from_violations(
                violations=[],
                files_reviewed=0,
                rules_applied=len([r for r in rules if r.is_active]),
            )

        # Review each file
        for file in reviewable:
            try:
                violations = self._review_single_file(file, rules, context)
                all_violations.extend(violations)
                reviewed_count += 1
            except Exception as e:
                logger.warning(f"Failed to review {file.path}: {e}")

        active_rules = len([r for r in rules if r.is_active])

        result = ReviewResult.from_violations(
            violations=all_violations,
            files_reviewed=reviewed_count,
            rules_applied=active_rules,
        )

        logger.info(
            f"Review complete: {result.violation_count} violations, score: {result.score}/100"
        )

        return result

    def _review_single_file(
        self,
        file: CodeFile,
        rules: List[ParsedRule],
        context: Optional[str] = None,
    ) -> List[Violation]:
        """
        Review a single file.

        Args:
            file: File to review
            rules: Rules to apply
            context: Optional context

        Returns:
            List of violations found
        """
        logger.debug(f"Reviewing {file.path} ({file.line_count} lines)")

        # Build prompt
        prompt = self._engine.build_prompt(file, rules, context)

        # Send to AI
        response = self.analyzer.analyze(prompt)

        # Parse response
        violations = self._engine.parse_response(response, file.path)

        # Validate violations against actual file
        validated = self._validate_violations(violations, file)

        logger.debug(
            f"Found {len(validated)} violations in {file.path} (filtered from {len(violations)})"
        )

        return validated

    def _validate_violations(
        self,
        violations: List[Violation],
        file: CodeFile,
    ) -> List[Violation]:
        """
        Filter out hallucinated violations that don't make sense.

        Args:
            violations: Raw violations from AI
            file: The actual file being reviewed

        Returns:
            Validated violations
        """
        validated = []

        for v in violations:
            # Reject violations with line numbers beyond file length
            if v.line > file.line_count:
                logger.debug(
                    f"Rejected '{v.rule_name}': line {v.line} > file length {file.line_count}"
                )
                continue

            # Reject "Long Method" for small files (< 50 lines)
            if "long method" in v.rule_name.lower() and file.line_count < 50:
                logger.debug(f"Rejected 'Long Method': file only has {file.line_count} lines")
                continue

            # Reject "Deep Nesting" for small files (< 30 lines)
            if "deep nesting" in v.rule_name.lower() and file.line_count < 30:
                logger.debug(f"Rejected 'Deep Nesting': file only has {file.line_count} lines")
                continue

            # Reject "God Class" for small files (< 100 lines)
            if "god class" in v.rule_name.lower() and file.line_count < 100:
                logger.debug(f"Rejected 'God Class': file only has {file.line_count} lines")
                continue

            validated.append(v)

        return validated

    def get_rules_summary(self) -> str:
        """Get a summary of loaded rules."""
        rules = self.get_rules()
        return self._engine.get_rules_summary(rules)
