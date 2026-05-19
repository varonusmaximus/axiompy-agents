"""FileSystem Document Source.

Load documents from the local filesystem.

Supports:
- Text files (.txt, .md, .rst)
- Code files (.py, .js, .ts, .java, .go, etc.)
- Recursive directory scanning
- Glob patterns

Example:
    from axiompy.agents.io.documents import FileSystemSource

    source = FileSystemSource()

    # Load single file
    doc = source.load_document("./README.md")

    # Load multiple files
    docs = source.load_documents(["./docs/", "./src/*.py"])
"""

import glob
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

from axiompy.agents.io.types import Document, DocumentMetadata
from axiompy.agents.io.errors import AgentIOIngestionError
from axiompy.loggers import LoggerFactory

logger = LoggerFactory.create_logger(__name__)

# Default supported extensions
DEFAULT_EXTENSIONS = {
    # Text
    ".txt",
    ".md",
    ".rst",
    ".text",
    # Code
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".swift",
    ".kt",
    ".scala",
    # Config
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    # Web
    ".html",
    ".css",
    ".xml",
    # Shell
    ".sh",
    ".bash",
    ".zsh",
    # SQL
    ".sql",
}


class FileSystemSource:
    """
    Document source for loading files from the local filesystem.

    Supports text files, code files, and recursive directory scanning.

    Attributes:
        extensions: Set of supported file extensions
        encoding: File encoding (default: utf-8)

    Example:
        source = FileSystemSource()
        docs = source.load_documents(["./docs/", "*.md"])
    """

    def __init__(
        self,
        extensions: Optional[Set[str]] = None,
        encoding: str = "utf-8",
        ignore_patterns: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize filesystem source.

        Args:
            extensions: Allowed file extensions (default: common text/code files)
            encoding: File encoding for reading (default: utf-8)
            ignore_patterns: Glob patterns to ignore (e.g., ["**/node_modules/**"])
        """
        self._extensions = extensions or DEFAULT_EXTENSIONS
        self._encoding = encoding
        self._ignore_patterns = ignore_patterns or [
            "**/node_modules/**",
            "**/__pycache__/**",
            "**/.git/**",
            "**/venv/**",
            "**/.venv/**",
            "**/dist/**",
            "**/build/**",
            "**/*.egg-info/**",
        ]
        logger.info(
            f"FileSystemSource initialized: {len(self._extensions)} extensions, encoding={encoding}"
        )

    def load_documents(self, paths: List[str]) -> List[Document]:
        """
        Load multiple documents from paths.

        Args:
            paths: List of file paths, directory paths, or glob patterns

        Returns:
            List of loaded documents

        Raises:
            AgentIOIngestionError: If loading fails
        """
        documents: List[Document] = []
        seen_paths: Set[str] = set()

        for path in paths:
            try:
                # Expand path (handle globs, directories, files)
                file_paths = self._expand_path(path)

                for file_path in file_paths:
                    # Skip duplicates
                    abs_path = os.path.abspath(file_path)
                    if abs_path in seen_paths:
                        continue
                    seen_paths.add(abs_path)

                    # Skip ignored patterns
                    if self._should_ignore(file_path):
                        logger.debug(f"Ignoring: {file_path}")
                        continue

                    # Load document
                    try:
                        doc = self._load_file(file_path)
                        documents.append(doc)
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path}: {e}")

            except Exception as e:
                logger.error(f"Error processing path {path}: {e}")
                raise AgentIOIngestionError(f"Failed to load from {path}: {e}") from e

        logger.info(f"Loaded {len(documents)} documents from {len(paths)} paths")
        return documents

    def load_document(self, path: str) -> Document:
        """
        Load a single document.

        Args:
            path: File path

        Returns:
            Loaded document

        Raises:
            AgentIOIngestionError: If loading fails
        """
        if not os.path.isfile(path):
            raise AgentIOIngestionError(f"File not found: {path}")

        return self._load_file(path)

    def _expand_path(self, path: str) -> List[str]:
        """
        Expand a path to a list of file paths.

        Handles:
        - Glob patterns (*.md, **/*.py)
        - Directories (recursively finds files)
        - Single files

        Args:
            path: Path to expand

        Returns:
            List of file paths
        """
        # Handle glob patterns
        if "*" in path or "?" in path:
            matches = glob.glob(path, recursive=True)
            return [m for m in matches if os.path.isfile(m) and self._has_valid_extension(m)]

        # Handle directories
        if os.path.isdir(path):
            files = []
            for root, _, filenames in os.walk(path):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    if self._has_valid_extension(file_path):
                        files.append(file_path)
            return files

        # Handle single file
        if os.path.isfile(path):
            if self._has_valid_extension(path):
                return [path]
            else:
                logger.warning(f"Skipping unsupported file type: {path}")
                return []

        # Path doesn't exist
        logger.warning(f"Path not found: {path}")
        return []

    def _has_valid_extension(self, path: str) -> bool:
        """Check if file has a supported extension."""
        ext = os.path.splitext(path)[1].lower()
        return ext in self._extensions

    def _should_ignore(self, path: str) -> bool:
        """Check if path matches any ignore patterns."""
        for pattern in self._ignore_patterns:
            if glob.fnmatch.fnmatch(path, pattern):
                return True
            # Also check with absolute path
            if glob.fnmatch.fnmatch(os.path.abspath(path), pattern):
                return True
        return False

    def _load_file(self, path: str) -> Document:
        """
        Load a single file into a Document.

        Args:
            path: File path

        Returns:
            Document object

        Raises:
            AgentIOIngestionError: If reading fails
        """
        try:
            abs_path = os.path.abspath(path)
            stat = os.stat(abs_path)

            with open(abs_path, encoding=self._encoding) as f:
                content = f.read()

            # Extract metadata
            path_obj = Path(abs_path)
            file_ext = path_obj.suffix.lstrip(".") or "txt"
            content_type = self._get_content_type(file_ext)
            metadata = DocumentMetadata(
                source=abs_path,
                title=path_obj.name,
                content_type=content_type,
                created_at=datetime.fromtimestamp(stat.st_ctime),
                extra={
                    "size_bytes": stat.st_size,
                    "relative_path": path,
                    "file_extension": file_ext,
                    "updated_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                },
            )

            doc = Document(
                id=abs_path,
                content=content,
                metadata=metadata,
            )

            logger.debug(f"Loaded: {path} ({len(content)} chars)")
            return doc

        except UnicodeDecodeError as e:
            raise AgentIOIngestionError(
                f"Failed to decode {path} with {self._encoding} encoding: {e}"
            ) from e
        except Exception as e:
            raise AgentIOIngestionError(f"Failed to read {path}: {e}") from e

    def _get_content_type(self, extension: str) -> str:
        """Map file extension to MIME content type."""
        content_types = {
            "txt": "text/plain",
            "md": "text/markdown",
            "rst": "text/x-rst",
            "py": "text/x-python",
            "js": "text/javascript",
            "ts": "text/typescript",
            "json": "application/json",
            "yaml": "text/yaml",
            "yml": "text/yaml",
            "html": "text/html",
            "css": "text/css",
            "xml": "application/xml",
            "sql": "application/sql",
        }
        return content_types.get(extension, "text/plain")

    def __repr__(self) -> str:
        return f"FileSystemSource(extensions={len(self._extensions)}, encoding={self._encoding})"
