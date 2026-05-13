"""Applications Layer - Executable entry points for the code review agent.

Different applications that run the code review domain:
- CLI: Command-line interface (`axiompy code-review`)
- Webhook: FastAPI webhook handler for GitHub events
- Library: Python API for in-process use
"""

from .library import create_service, review_diff, review_files, review_pr

__all__ = [
    # Library functions
    "review_files",
    "review_diff",
    "review_pr",
    "create_service",
]
