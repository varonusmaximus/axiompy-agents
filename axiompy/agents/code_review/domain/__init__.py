"""Domain Layer - Core business logic for code review.

This layer contains:
- Models (CodeFile, FileDiff, PullRequestInfo)
- Rules (ParsedRule, RuleType, RuleSeverity)
- Results (ReviewResult, Violation, ReviewComment)
- Engine (RulesEngine for parsing and prompt building)
- Ports (interfaces the domain needs from infrastructure)
- Service (CodeReviewService - the core use case)

The domain defines WHAT it needs (ports), infrastructure provides HOW.
"""

from .engine import RulesEngine
from .models import CodeFile, FileDiff, PullRequestInfo
from .ports import AIAnalyzer, CodeSource, ReviewPublisher, RulesSource
from .results import ReviewComment, ReviewResult, ReviewSeverity, Violation
from .rules import CodeExample, ParsedRule, RuleCategory, RuleSeverity, RuleType
from .service import CodeReviewService

__all__ = [
    # Models
    "CodeFile",
    "FileDiff",
    "PullRequestInfo",
    # Rules
    "ParsedRule",
    "RuleType",
    "RuleSeverity",
    "RuleCategory",
    "CodeExample",
    # Results
    "ReviewResult",
    "Violation",
    "ReviewComment",
    "ReviewSeverity",
    # Engine
    "RulesEngine",
    # Ports
    "CodeSource",
    "RulesSource",
    "AIAnalyzer",
    "ReviewPublisher",
    # Service
    "CodeReviewService",
]
