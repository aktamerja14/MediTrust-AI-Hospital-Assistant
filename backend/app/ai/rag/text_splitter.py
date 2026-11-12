"""Text Splitter for RAG Pipeline.

Splits documents into overlapping chunks for embedding and retrieval.
"""
import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from app.core.config import settings

logger = logging.getLogger("meditrust.ai.rag.text_splitter")


@dataclass
class TextChunk:
    """A chunk of text with metadata."""
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict


class TextSplitter:
    """Splits text into overlapping chunks."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.chunk_size = chunk_size or settings.RAG_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.RAG_CHUNK_OVERLAP

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

    def split_text(self, text: str, metadata: Optional[dict] = None) -> List[TextChunk]:
        """Split text into overlapping chunks.

        Args:
            text: Text to split
            metadata: Base metadata to attach to each chunk

        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []

        # Clean text
        text = self._clean_text(text)

        # Split by paragraphs first, then by sentences if needed
        paragraphs = self._split_paragraphs(text)

        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0

        for para in paragraphs:
            # If adding this paragraph exceeds chunk_size, finalize current chunk
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunks.append(self._create_chunk(
                    current_chunk, chunk_index, current_start,
                    current_start + len(current_chunk), metadata
                ))
                chunk_index += 1

                # Start new chunk with overlap
                overlap_text = self._get_overlap(current_chunk)
                current_chunk = overlap_text + para
                current_start = current_start + len(current_chunk) - len(overlap_text) - len(para)
            else:
                current_chunk += para

        # Add final chunk
        if current_chunk.strip():
            chunks.append(self._create_chunk(
                current_chunk, chunk_index, current_start,
                current_start + len(current_chunk), metadata
            ))

        logger.debug("Split text into %d chunks (size=%d, overlap=%d)",
                     len(chunks), self.chunk_size, self.chunk_overlap)
        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove zero-width characters
        text = text.replace('\u200b', '').replace('\ufeff', '')
        return text.strip()

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines or single newlines with indentation
        paragraphs = re.split(r'\n\s*\n|\n(?=\s{2,})', text)
        # Further split long paragraphs by sentences
        result = []
        for para in paragraphs:
            if len(para) <= self.chunk_size:
                result.append(para + "\n\n")
            else:
                # Split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) > self.chunk_size and current:
                        result.append(current + " ")
                        current = sent
                    else:
                        current += sent + " "
                if current:
                    result.append(current)
        return result

    def _get_overlap(self, text: str) -> str:
        """Get overlap text from end of chunk."""
        if self.chunk_overlap == 0:
            return ""
        # Try to get overlap at sentence boundary
        overlap_text = text[-self.chunk_overlap:]
        # Find last sentence boundary in overlap
        last_period = max(
            overlap_text.rfind('. '),
            overlap_text.rfind('! '),
            overlap_text.rfind('? '),
        )
        if last_period > self.chunk_overlap // 2:
            return overlap_text[last_period + 2:]
        return overlap_text

    def _create_chunk(
        self,
        content: str,
        index: int,
        start: int,
        end: int,
        base_metadata: Optional[dict],
    ) -> TextChunk:
        """Create a TextChunk with metadata."""
        metadata = dict(base_metadata) if base_metadata else {}
        metadata.update({
            "chunk_index": index,
            "chunk_start": start,
            "chunk_end": end,
        })
        return TextChunk(
            content=content.strip(),
            chunk_index=index,
            start_char=start,
            end_char=end,
            metadata=metadata,
        )