"""CLI Adapter - Command-line interface for code review.

Provides the `axiompy code-review` command.

Usage:
    axiompy code-review .                    # Review current directory
    axiompy code-review src/main.py          # Review specific file
    axiompy code-review --staged             # Review staged git changes
    axiompy code-review --diff HEAD~1..HEAD  # Review commit range
    axiompy code-review --pr owner/repo#123  # Review GitHub PR
"""

import argparse
import os
import sys
from typing import List, Optional

from ..adapters.analyzers import AnalyzerFactory, AnalyzerSettings, AnalyzerType
from ..adapters.publishers import ConsolePublisher, GitHubPublisher, JSONPublisher
from ..adapters.rules import FileRulesSource, GitHubRulesSource
from ..adapters.sources import FileSystemSource, GitHubSource, GitSource
from ..defaults import DEFAULT_MODEL, DEFAULT_OLLAMA_HOST
from ..domain.service import CodeReviewService


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="axiompy code-review",
        description="AI-powered code review using AGENTS.md rules.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  axiompy code-review .                    Review current directory
  axiompy code-review src/                 Review src directory
  axiompy code-review main.py utils.py     Review specific files
  axiompy code-review --staged             Review staged git changes
  axiompy code-review --diff HEAD~1..HEAD  Review last commit
  axiompy code-review --pr owner/repo#123  Review GitHub PR
  axiompy code-review . --output json      Output as JSON
        """,
    )

    # Input source (positional or flags)
    parser.add_argument(
        "paths",
        nargs="*",
        help="Files or directories to review",
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Review staged git changes",
    )
    parser.add_argument(
        "--diff",
        metavar="BASE..HEAD",
        help="Review git diff (e.g., HEAD~1..HEAD)",
    )
    parser.add_argument(
        "--pr",
        metavar="OWNER/REPO#NUM",
        help="Review GitHub PR (e.g., varonusmaximus/axiompy#123)",
    )

    # Rules configuration
    parser.add_argument(
        "--rules",
        default="AGENTS.md",
        help="Path to rules file (default: AGENTS.md)",
    )
    parser.add_argument(
        "--rules-repo",
        default="varonusmaximus/axiompy",
        help="GitHub repo for rules (default: varonusmaximus/axiompy)",
    )

    # AI configuration
    parser.add_argument(
        "--ai",
        choices=["ollama", "openai"],
        default="ollama",
        help="AI provider (default: ollama)",
    )
    parser.add_argument(
        "--model",
        help="AI model (default: codellama for ollama, gpt-4o for openai)",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Ollama server URL",
    )

    # Output configuration
    parser.add_argument(
        "--output",
        choices=["console", "json", "github"],
        default="console",
        help="Output format (default: console)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Quiet output (only errors)",
    )

    # Exit behavior
    parser.add_argument(
        "--fail-on",
        choices=["error", "warning", "none"],
        default="error",
        help="Exit with error code on (default: error)",
    )

    return parser


def parse_pr_arg(pr_arg: str) -> tuple:
    """Parse PR argument (owner/repo#num)."""
    if "#" not in pr_arg:
        raise ValueError(f"Invalid PR format: {pr_arg}. Expected: owner/repo#num")

    repo_part, num_str = pr_arg.rsplit("#", 1)

    if "/" not in repo_part:
        raise ValueError(f"Invalid PR format: {pr_arg}. Expected: owner/repo#num")

    owner, repo = repo_part.split("/", 1)
    pr_number = int(num_str)

    return owner, repo, pr_number


def parse_diff_arg(diff_arg: str) -> tuple:
    """Parse diff argument (base..head)."""
    if ".." not in diff_arg:
        raise ValueError(f"Invalid diff format: {diff_arg}. Expected: base..head")

    base, head = diff_arg.split("..", 1)
    return base, head


def main(args: Optional[List[str]] = None) -> int:
    """
    Main CLI entry point.

    Args:
        args: Command-line arguments (uses sys.argv if None)

    Returns:
        Exit code (0 for success)
    """
    parser = create_parser()
    opts = parser.parse_args(args)

    # Determine mode
    if opts.pr:
        return run_pr_review(opts)
    elif opts.staged:
        return run_staged_review(opts)
    elif opts.diff:
        return run_diff_review(opts)
    elif opts.paths:
        return run_file_review(opts)
    else:
        # Default: review current directory
        opts.paths = ["."]
        return run_file_review(opts)


def run_file_review(opts) -> int:
    """Review local files."""
    service = create_service(opts, source_type="filesystem")
    result = service.review_files(opts.paths)
    return check_exit(result, opts)


def run_staged_review(opts) -> int:
    """Review staged git changes."""
    service = create_service(opts, source_type="git")
    result = service.review_diff("HEAD", "staged")
    return check_exit(result, opts)


def run_diff_review(opts) -> int:
    """Review git diff."""
    base, head = parse_diff_arg(opts.diff)
    service = create_service(opts, source_type="git")
    result = service.review_diff(base, head)
    return check_exit(result, opts)


def run_pr_review(opts) -> int:
    """Review GitHub PR."""
    owner, repo, pr_number = parse_pr_arg(opts.pr)

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN environment variable required", file=sys.stderr)
        return 1

    service = create_service(opts, source_type="github", github_token=token)
    result = service.review_pull_request(owner, repo, pr_number)
    return check_exit(result, opts)


def create_service(opts, source_type: str, github_token: str = None) -> CodeReviewService:
    """Create service based on CLI options."""
    # Code source
    if source_type == "filesystem":
        code_source = FileSystemSource(root=".")
    elif source_type == "git":
        code_source = GitSource(repo_path=".")
    elif source_type == "github":
        code_source = GitHubSource(token=github_token)
    else:
        raise ValueError(f"Unknown source type: {source_type}")

    # Rules source
    if source_type == "github" or not os.path.exists(opts.rules):
        # Use GitHub rules for PR review or when local rules don't exist
        token = github_token or os.environ.get("GITHUB_TOKEN")
        if token:
            rules_source = GitHubRulesSource(
                token=token,
                repo=opts.rules_repo,
            )
        else:
            rules_source = FileRulesSource(opts.rules)
    else:
        rules_source = FileRulesSource(opts.rules)

    # Analyzer
    match opts.ai:
        case "ollama":
            model = opts.model or DEFAULT_MODEL
            settings = AnalyzerSettings(
                host=opts.ollama_host,
                model=model,
                show_progress=True,  # Real-time progress for CLI
            )
            analyzer = AnalyzerFactory.create(AnalyzerType.OLLAMA, settings)
        case "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("Error: OPENAI_API_KEY required for --ai openai", file=sys.stderr)
                sys.exit(1)
            model = opts.model or "gpt-4o"
            settings = AnalyzerSettings(api_key=api_key, model=model)
            analyzer = AnalyzerFactory.create(AnalyzerType.OPENAI, settings)
        case "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                print("Error: ANTHROPIC_API_KEY required for --ai anthropic", file=sys.stderr)
                sys.exit(1)
            model = opts.model or "claude-sonnet-4-20250514"
            settings = AnalyzerSettings(api_key=api_key, model=model)
            analyzer = AnalyzerFactory.create(AnalyzerType.ANTHROPIC, settings)
        case _:
            print(f"Error: Unknown AI provider: {opts.ai}", file=sys.stderr)
            sys.exit(1)

    # Publisher
    if opts.output == "json":
        publisher = JSONPublisher()
    elif opts.output == "github":
        token = github_token or os.environ.get("GITHUB_TOKEN")
        if not token:
            print("Error: GITHUB_TOKEN required for --output github", file=sys.stderr)
            sys.exit(1)
        publisher = GitHubPublisher(token=token)
    else:
        publisher = ConsolePublisher(
            verbose=opts.verbose,
            use_color=not opts.quiet,
        )

    return CodeReviewService(
        code_source=code_source,
        rules_source=rules_source,
        analyzer=analyzer,
        publisher=publisher,
    )


def check_exit(result, opts) -> int:
    """Check result and return exit code."""
    if opts.fail_on == "error" and result.has_errors:
        return 1
    if opts.fail_on == "warning" and result.violations:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
