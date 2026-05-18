"""PDF Document Source.

Load documents from PDF files using pypdf.

Supports:
- Text extraction from PDF pages
- Metadata extraction (title, author, etc.)
- Page-level or full-document loading
- Directory scanning for PDF files

Example:
    from axiompy.agents.io.documents import PDFSource

    source = PDFSource()

    # Load single PDF
    doc = source.load_document("./report.pdf")

    # Load multiple PDFs
    docs = source.load_documents(["./docs/", "./reports/*.pdf"])

Note:
    Requires pypdf: pip install pypdf
"""

import glob
from datetime import datetime
from pathlib import Path
from typing import List

from axiompy.agents.io.types import Document, DocumentMetadata
from axiompy.agents.io.errors import RAGIngestionError
from axiompy.loggers import LoggerFactory

logger = LoggerFactory.create_logger(__name__)

# Check for pypdf availability
try:
    from pypdf import PdfReader

    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    PdfReader = None  # type: ignore


class PDFSource:
    """
    Document source for loading content from PDF files.

    Uses pypdf for PDF text extraction.

    Attributes:
        pages_as_documents: If True, each page becomes a separate document
        include_metadata: If True, include PDF metadata in document metadata

    Example:
        source = PDFSource()
        docs = source.load_documents(["./docs/"])
    """

    def __init__(
        self,
        pages_as_documents: bool = False,
        include_metadata: bool = True,
    ) -> None:
        """
        Initialize PDF source.

        Args:
            pages_as_documents: If True, each page is a separate document
            include_metadata: If True, extract PDF metadata (title, author, etc.)

        Raises:
            RAGIngestionError: If pypdf is not installed
        """
        if not PYPDF_AVAILABLE:
            raise RAGIngestionError(
                "pypdf is required for PDFSource. Install with: pip install pypdf"
            )

        self._pages_as_documents = pages_as_documents
        self._include_metadata = include_metadata

        logger.debug(f"PDFSource initialized: pages_as_documents={pages_as_documents}")

    def load_document(self, path: str) -> Document:
        """
        Load a single PDF document.

        Args:
            path: Path to PDF file

        Returns:
            Document with extracted text and metadata

        Raises:
            RAGIngestionError: If file not found or extraction fails
        """
        file_path = Path(path).resolve()

        if not file_path.exists():
            raise RAGIngestionError(f"PDF file not found: {path}")

        if file_path.suffix.lower() != ".pdf":
            raise RAGIngestionError(f"Not a PDF file: {path}")

        try:
            reader = PdfReader(str(file_path))

            # Extract text from all pages
            text_parts = []
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

            content = "\n\n".join(text_parts)

            if not content.strip():
                logger.warning(f"No text extracted from PDF: {path}")

            # Extract metadata
            pdf_metadata = reader.metadata or {}

            # Get file info
            file_stat = file_path.stat()

            # Build document metadata
            title = None
            if self._include_metadata and pdf_metadata:
                title = pdf_metadata.get("/Title") or pdf_metadata.get("title")

            extra = {
                "file_size": file_stat.st_size,
                "page_count": len(reader.pages),
            }

            if self._include_metadata and pdf_metadata:
                # Common PDF metadata fields
                if pdf_metadata.get("/Author"):
                    extra["author"] = pdf_metadata.get("/Author")
                if pdf_metadata.get("/Subject"):
                    extra["subject"] = pdf_metadata.get("/Subject")
                if pdf_metadata.get("/Creator"):
                    extra["creator"] = pdf_metadata.get("/Creator")
                if pdf_metadata.get("/Producer"):
                    extra["producer"] = pdf_metadata.get("/Producer")
                if pdf_metadata.get("/CreationDate"):
                    extra["creation_date"] = str(pdf_metadata.get("/CreationDate"))

            # Generate document ID
            doc_id = f"pdf:{file_path.name}".replace(" ", "_")

            metadata = DocumentMetadata(
                source=str(file_path),
                title=title or file_path.stem,
                content_type="application/pdf",
                created_at=datetime.fromtimestamp(file_stat.st_mtime),
                extra=extra,
            )

            logger.debug(f"Loaded PDF: {path} ({len(reader.pages)} pages, {len(content)} chars)")

            return Document(id=doc_id, content=content, metadata=metadata)

        except RAGIngestionError:
            raise
        except Exception as e:
            raise RAGIngestionError(f"Failed to load PDF {path}: {e}") from e

    def load_document_pages(self, path: str) -> List[Document]:
        """
        Load a PDF as separate documents per page.

        Args:
            path: Path to PDF file

        Returns:
            List of Document objects, one per page

        Raises:
            RAGIngestionError: If file not found or extraction fails
        """
        file_path = Path(path).resolve()

        if not file_path.exists():
            raise RAGIngestionError(f"PDF file not found: {path}")

        try:
            reader = PdfReader(str(file_path))
            documents = []

            pdf_metadata = reader.metadata or {}
            title_base = None
            if self._include_metadata and pdf_metadata:
                title_base = pdf_metadata.get("/Title") or pdf_metadata.get("title")
            title_base = title_base or file_path.stem

            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()
                if not page_text or not page_text.strip():
                    continue

                doc_id = f"pdf:{file_path.name}:page_{page_num}".replace(" ", "_")

                metadata = DocumentMetadata(
                    source=str(file_path),
                    title=f"{title_base} - Page {page_num}",
                    content_type="application/pdf",
                    extra={
                        "page_number": page_num,
                        "total_pages": len(reader.pages),
                    },
                )

                documents.append(Document(id=doc_id, content=page_text, metadata=metadata))

            logger.debug(f"Loaded {len(documents)} pages from PDF: {path}")
            return documents

        except RAGIngestionError:
            raise
        except Exception as e:
            raise RAGIngestionError(f"Failed to load PDF pages {path}: {e}") from e

    def load_documents(self, paths: List[str]) -> List[Document]:
        """
        Load documents from multiple PDF files or directories.

        Args:
            paths: List of file paths, directories, or glob patterns

        Returns:
            List of Document objects

        Note:
            Failed PDFs are logged and skipped.
        """
        documents = []
        pdf_files = set()

        for path in paths:
            path_obj = Path(path)

            if path_obj.is_file():
                if path_obj.suffix.lower() == ".pdf":
                    pdf_files.add(str(path_obj.resolve()))
            elif path_obj.is_dir():
                # Scan directory for PDFs
                for pdf_path in path_obj.rglob("*.pdf"):
                    pdf_files.add(str(pdf_path.resolve()))
            elif "*" in path or "?" in path:
                # Glob pattern
                for match in glob.glob(path, recursive=True):
                    if match.lower().endswith(".pdf"):
                        pdf_files.add(str(Path(match).resolve()))
            else:
                logger.warning(f"Path not found: {path}")

        for pdf_file in sorted(pdf_files):
            try:
                if self._pages_as_documents:
                    docs = self.load_document_pages(pdf_file)
                    documents.extend(docs)
                else:
                    doc = self.load_document(pdf_file)
                    documents.append(doc)
            except RAGIngestionError as e:
                logger.warning(f"Skipping PDF {pdf_file}: {e}")
                continue

        logger.info(f"Loaded {len(documents)} documents from {len(pdf_files)} PDFs")
        return documents

    def __repr__(self) -> str:
        return f"PDFSource(pages_as_documents={self._pages_as_documents})"
