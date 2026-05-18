"""Document Chunking Implementations.

Provides strategies for splitting documents into chunks for embedding.

Available chunkers:
- FixedSizeChunker: Fixed character-based chunking with overlap
- SentenceChunker: Sentence-aware chunking (groups sentences up to target size)
- ParagraphChunker: Paragraph-aware chunking (splits on double newlines)

Example:
    chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_document(document)

    sentence_chunker = SentenceChunker(target_size=500)
    chunks = sentence_chunker.chunk_document(document)
"""

import re
from dataclasses import dataclass

from axiompy.agents.io.defaults import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from axiompy.agents.io.types import Document, DocumentChunk


@dataclass
class FixedSizeChunker:
    """
    Fixed-size character-based document chunker.

    Splits documents into chunks of fixed character size with configurable overlap.
    Simple and predictable, good for most use cases.

    Attributes:
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of overlapping characters between chunks
    """

    _chunk_size: int = DEFAULT_CHUNK_SIZE
    _chunk_overlap: int = DEFAULT_CHUNK_OVERLAP

    def __post_init__(self) -> None:
        """Validate chunker configuration."""
        if self._chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if self._chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self._chunk_overlap >= self._chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

    @property
    def chunk_size(self) -> int:
        """Get the target chunk size."""
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        """Get the overlap between chunks."""
        return self._chunk_overlap

    def chunk_document(self, document: Document) -> list[DocumentChunk]:
        """
        Split a document into fixed-size chunks.

        Args:
            document: Document to chunk

        Returns:
            List of document chunks with positions tracked
        """
        content = document.content
        chunks: list[DocumentChunk] = []

        if not content:
            return chunks

        # Calculate step size (chunk_size - overlap)
        step = self._chunk_size - self._chunk_overlap

        start = 0
        chunk_index = 0

        while start < len(content):
            end = min(start + self._chunk_size, len(content))
            chunk_content = content[start:end]

            # Skip empty chunks
            if chunk_content.strip():
                chunk = DocumentChunk(
                    id=f"{document.id}_chunk_{chunk_index}",
                    document_id=document.id,
                    content=chunk_content,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    metadata={
                        "source": document.metadata.source,
                        "chunk_size": len(chunk_content),
                    },
                )
                chunks.append(chunk)
                chunk_index += 1

            start += step

            # Prevent infinite loop for very small content
            if start >= len(content):
                break

        return chunks


@dataclass
class SentenceChunker:
    """
    Sentence-aware document chunker.

    Groups sentences together until the target size is reached.
    Respects sentence boundaries for more coherent chunks.

    Uses regex-based sentence detection (no nltk dependency).

    Attributes:
        target_size: Target size of each chunk in characters
        overlap_sentences: Number of sentences to overlap between chunks
    """

    _target_size: int = DEFAULT_CHUNK_SIZE
    _overlap_sentences: int = 1

    # Sentence boundary pattern: matches . ! ? followed by space and capital
    # or end of string. Handles common abbreviations.
    SENTENCE_PATTERN = re.compile(
        r"(?<=[.!?])\s+(?=[A-Z])|"  # Standard sentence end
        r"(?<=[.!?])\s*$|"  # End of text
        r"(?<=\n)\s*(?=\S)"  # After newline
    )

    def __post_init__(self) -> None:
        """Validate chunker configuration."""
        if self._target_size < 1:
            raise ValueError("target_size must be positive")
        if self._overlap_sentences < 0:
            raise ValueError("overlap_sentences cannot be negative")

    @property
    def chunk_size(self) -> int:
        """Get the target chunk size."""
        return self._target_size

    @property
    def chunk_overlap(self) -> int:
        """Get the overlap (in sentences, not chars - returns 0 for compatibility)."""
        return 0  # Sentence overlap doesn't map to char overlap

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences using regex.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        if not text.strip():
            return []

        # Split on sentence boundaries
        sentences = self.SENTENCE_PATTERN.split(text)

        # Clean up and filter empty
        result = []
        for s in sentences:
            s = s.strip()
            if s:
                result.append(s)

        return result

    def chunk_document(self, document: Document) -> list[DocumentChunk]:
        """
        Split a document into sentence-aware chunks.

        Args:
            document: Document to chunk

        Returns:
            List of document chunks respecting sentence boundaries
        """
        content = document.content
        chunks: list[DocumentChunk] = []

        if not content or not content.strip():
            return chunks

        sentences = self._split_sentences(content)

        if not sentences:
            return chunks

        current_sentences: list[str] = []
        current_length = 0
        chunk_index = 0
        start_char = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # If adding this sentence exceeds target, create chunk
            if current_length + sentence_len > self._target_size and current_sentences:
                # Create chunk from current sentences
                chunk_content = " ".join(current_sentences)
                end_char = start_char + len(chunk_content)

                chunk = DocumentChunk(
                    id=f"{document.id}_chunk_{chunk_index}",
                    document_id=document.id,
                    content=chunk_content,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    metadata={
                        "source": document.metadata.source,
                        "chunk_size": len(chunk_content),
                        "sentence_count": len(current_sentences),
                    },
                )
                chunks.append(chunk)
                chunk_index += 1

                # Start new chunk with overlap
                if self._overlap_sentences > 0 and len(current_sentences) > self._overlap_sentences:
                    overlap = current_sentences[-self._overlap_sentences :]
                    start_char = end_char - sum(len(s) for s in overlap) - len(overlap) + 1
                    current_sentences = overlap.copy()
                    current_length = (
                        sum(len(s) for s in current_sentences) + len(current_sentences) - 1
                    )
                else:
                    start_char = end_char + 1
                    current_sentences = []
                    current_length = 0

            # Add sentence
            current_sentences.append(sentence)
            current_length += sentence_len + (1 if current_length > 0 else 0)  # +1 for space

        # Handle remaining sentences
        if current_sentences:
            chunk_content = " ".join(current_sentences)
            chunk = DocumentChunk(
                id=f"{document.id}_chunk_{chunk_index}",
                document_id=document.id,
                content=chunk_content,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=start_char + len(chunk_content),
                metadata={
                    "source": document.metadata.source,
                    "chunk_size": len(chunk_content),
                    "sentence_count": len(current_sentences),
                },
            )
            chunks.append(chunk)

        return chunks


@dataclass
class ParagraphChunker:
    """
    Paragraph-aware document chunker.

    Splits documents on paragraph boundaries (double newlines).
    Can merge small paragraphs to reach target size.

    Attributes:
        target_size: Target size of each chunk in characters
        merge_small: Whether to merge small paragraphs together
    """

    _target_size: int = DEFAULT_CHUNK_SIZE
    _merge_small: bool = True

    def __post_init__(self) -> None:
        """Validate chunker configuration."""
        if self._target_size < 1:
            raise ValueError("target_size must be positive")

    @property
    def chunk_size(self) -> int:
        """Get the target chunk size."""
        return self._target_size

    @property
    def chunk_overlap(self) -> int:
        """Get the overlap (paragraphs don't overlap)."""
        return 0

    def _split_paragraphs(self, text: str) -> list[str]:
        """
        Split text into paragraphs.

        Args:
            text: Text to split

        Returns:
            List of paragraphs
        """
        if not text.strip():
            return []

        # Split on double newlines (paragraph boundaries)
        paragraphs = re.split(r"\n\s*\n", text)

        # Clean up and filter empty
        result = []
        for p in paragraphs:
            p = p.strip()
            if p:
                result.append(p)

        return result

    def chunk_document(self, document: Document) -> list[DocumentChunk]:
        """
        Split a document into paragraph-aware chunks.

        Args:
            document: Document to chunk

        Returns:
            List of document chunks respecting paragraph boundaries
        """
        content = document.content
        chunks: list[DocumentChunk] = []

        if not content or not content.strip():
            return chunks

        paragraphs = self._split_paragraphs(content)

        if not paragraphs:
            return chunks

        if not self._merge_small:
            # Each paragraph is a chunk
            start_char = 0
            for i, para in enumerate(paragraphs):
                chunk = DocumentChunk(
                    id=f"{document.id}_chunk_{i}",
                    document_id=document.id,
                    content=para,
                    chunk_index=i,
                    start_char=start_char,
                    end_char=start_char + len(para),
                    metadata={
                        "source": document.metadata.source,
                        "chunk_size": len(para),
                        "paragraph_count": 1,
                    },
                )
                chunks.append(chunk)
                start_char += len(para) + 2  # +2 for \n\n
            return chunks

        # Merge small paragraphs
        current_paragraphs: list[str] = []
        current_length = 0
        chunk_index = 0
        start_char = 0

        for para in paragraphs:
            para_len = len(para)

            # If adding this paragraph exceeds target, create chunk
            if current_length + para_len > self._target_size and current_paragraphs:
                chunk_content = "\n\n".join(current_paragraphs)
                end_char = start_char + len(chunk_content)

                chunk = DocumentChunk(
                    id=f"{document.id}_chunk_{chunk_index}",
                    document_id=document.id,
                    content=chunk_content,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    metadata={
                        "source": document.metadata.source,
                        "chunk_size": len(chunk_content),
                        "paragraph_count": len(current_paragraphs),
                    },
                )
                chunks.append(chunk)
                chunk_index += 1

                start_char = end_char + 2  # +2 for \n\n separator
                current_paragraphs = []
                current_length = 0

            # Add paragraph
            current_paragraphs.append(para)
            current_length += para_len + (2 if current_length > 0 else 0)  # +2 for \n\n

        # Handle remaining paragraphs
        if current_paragraphs:
            chunk_content = "\n\n".join(current_paragraphs)
            chunk = DocumentChunk(
                id=f"{document.id}_chunk_{chunk_index}",
                document_id=document.id,
                content=chunk_content,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=start_char + len(chunk_content),
                metadata={
                    "source": document.metadata.source,
                    "chunk_size": len(chunk_content),
                    "paragraph_count": len(current_paragraphs),
                },
            )
            chunks.append(chunk)

        return chunks


# Note: ChunkerFactory is located in axiompy.agents.io.settings
# to avoid circular import issues.
