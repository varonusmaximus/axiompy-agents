"""URL Document Source.

Load documents from web URLs using axiompy HTTPClientFactory.

Supports:
- HTML pages (converted to text)
- Plain text files
- Markdown files
- JSON responses

Example:
    from axiompy.agents.io.documents import URLSource

    source = URLSource()

    # Load single URL
    doc = source.load_document("https://example.com/page.html")

    # Load multiple URLs
    docs = source.load_documents([
        "https://example.com/docs/intro",
        "https://example.com/docs/guide",
    ])
"""

import re
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urlparse

from axiompy.agents.io.types import Document, DocumentMetadata
from axiompy.agents.io.errors import AgentIOIngestionError
from axiompy.io.http import HTTPClientFactory
from axiompy.loggers import LoggerFactory

logger = LoggerFactory.create_logger(__name__)


def _strip_html_tags(html: str) -> str:
    """Remove HTML tags and decode entities."""
    # Remove script and style elements
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", html)

    # Decode common HTML entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)

    return text.strip()


def _extract_title(html: str) -> Optional[str]:
    """Extract title from HTML."""
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if match:
        return _strip_html_tags(match.group(1)).strip()
    return None


class URLSource:
    """
    Document source for loading content from web URLs.

    Uses axiompy HTTPClientFactory for HTTP requests.

    Attributes:
        timeout_secs: Request timeout in seconds
        user_agent: User-Agent header for requests

    Example:
        source = URLSource(timeout_secs=30)
        docs = source.load_documents(["https://example.com/page"])
    """

    def __init__(
        self,
        timeout_secs: int = 30,
        user_agent: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initialize URL source.

        Args:
            timeout_secs: Request timeout in seconds (default: 30)
            user_agent: Custom User-Agent header
            headers: Additional headers to include in requests
        """
        self._timeout = timeout_secs
        self._user_agent = user_agent or "axiompy-rag/1.0"
        self._headers = headers or {}

        # Create HTTP client
        self._client = HTTPClientFactory.create(timeout_secs=timeout_secs).add_header(
            "User-Agent", self._user_agent
        )

        for key, value in self._headers.items():
            self._client.add_header(key, value)

        logger.debug(f"URLSource initialized: timeout={timeout_secs}s")

    def load_document(self, url: str) -> Document:
        """
        Load a single document from a URL.

        Args:
            url: URL to fetch

        Returns:
            Document with content and metadata

        Raises:
            AgentIOIngestionError: If fetch fails
        """
        try:
            response = self._client.get(url)

            if response.status_code != 200:
                raise AgentIOIngestionError(f"Failed to fetch {url}: HTTP {response.status_code}")

            content_type = response.headers.get("content-type", "").lower()
            content = response.text

            # Extract title and clean content based on content type
            title = None
            if "text/html" in content_type:
                title = _extract_title(content)
                content = _strip_html_tags(content)
            elif "application/json" in content_type:
                # Keep JSON as-is for now
                pass

            # Parse URL for metadata
            parsed = urlparse(url)

            # Generate document ID from URL
            doc_id = f"url:{parsed.netloc}{parsed.path}".replace("/", "_")

            metadata = DocumentMetadata(
                source=url,
                title=title,
                content_type=content_type.split(";")[0] if content_type else "text/html",
                extra={
                    "domain": parsed.netloc,
                    "path": parsed.path,
                    "fetched_at": datetime.now().isoformat(),
                },
            )

            logger.debug(f"Loaded document from URL: {url} ({len(content)} chars)")

            return Document(id=doc_id, content=content, metadata=metadata)

        except AgentIOIngestionError:
            raise
        except Exception as e:
            raise AgentIOIngestionError(f"Failed to load URL {url}: {e}") from e

    def load_documents(self, urls: List[str]) -> List[Document]:
        """
        Load documents from multiple URLs.

        Args:
            urls: List of URLs to fetch

        Returns:
            List of Document objects

        Note:
            Failed URLs are logged and skipped, not raised as errors.
        """
        documents = []

        for url in urls:
            try:
                doc = self.load_document(url)
                documents.append(doc)
            except AgentIOIngestionError as e:
                logger.warning(f"Skipping URL {url}: {e}")
                continue

        logger.info(f"Loaded {len(documents)} documents from {len(urls)} URLs")
        return documents

    def __repr__(self) -> str:
        return f"URLSource(timeout={self._timeout}s)"
