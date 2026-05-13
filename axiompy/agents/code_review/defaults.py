"""
Default configuration values for the code review agent.

Single source of truth for all defaults - change here, changes everywhere.
"""

# Model defaults
DEFAULT_MODEL = "llama3.2:1b"  # Faster than qwen2.5-coder:1.5b in practice
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# Timeout configuration (seconds)
DEFAULT_TIMEOUT_SECS = 120  # Overall connection timeout
DEFAULT_FIRST_TOKEN_TIMEOUT = 120  # Max wait for first token (model loading + prompt processing)
DEFAULT_STREAM_TIMEOUT_SECS = 300  # Max total streaming time (5 min)
DEFAULT_IDLE_TIMEOUT_SECS = 45  # Max silence between tokens

# Rules defaults
DEFAULT_RULES_PATH = "AGENTS.md"
DEFAULT_RULES_URL = "https://raw.githubusercontent.com/user/repo/main/AGENTS.md"

# Chunk configuration
DEFAULT_CHUNKS = ["anti_patterns", "code_smells", "patterns"]

# Review modes - which chunks to include
REVIEW_MODES = {
    "quick": ["anti_patterns"],  # Just anti-patterns (fastest)
    "standard": ["anti_patterns", "code_smells"],  # + code smells
    "full": ["anti_patterns", "code_smells", "patterns"],  # + design principles
}
DEFAULT_REVIEW_MODE = "quick"  # Fast by default, use --mode=standard or --mode=full for more

# Limits
MAX_CODE_LENGTH = 10000
MAX_PATTERNS = 15
MAX_ANTI_PATTERNS = 10
MAX_CODE_SMELLS = 10
