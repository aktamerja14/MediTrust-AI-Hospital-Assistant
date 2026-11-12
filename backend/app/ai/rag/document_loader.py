"""Document Loader for RAG Pipeline.

Loads and extracts text from PDF and TXT files.
"""
import logging
from pathlib import Path
from typing import List, Optional

import pypdf

from app.core.config import settings

logger = logging.getLogger("meditrust.ai.rag.document_loader")


class DocumentLoaderError(Exception):
    """Exception for document loading errors."""
    pass


class DocumentLoader:
    """Loads and extracts text from documents."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or settings.DOCUMENT_STORAGE_PATH).resolve()

    def load_document(self, storage_filename: str) -> List[dict]:
        """Load a document and return list of pages with text and metadata.

        Args:
            storage_filename: Server-side storage filename (e.g., doc_abc123.pdf)

        Returns:
            List of dicts with keys: content, page_number, metadata
        """
        file_path = self.storage_path / storage_filename

        if not file_path.exists():
            raise DocumentLoaderError(f"File not found: {file_path}")

        if not self._is_supported(file_path):
            raise DocumentLoaderError(f"Unsupported file type: {file_path.suffix}")

        try:
            if file_path.suffix.lower() == ".pdf":
                return self._load_pdf(file_path)
            else:
                return self._load_txt(file_path)
        except DocumentLoaderError:
            raise
        except Exception as e:
            logger.exception("Failed to load document %s: %s", storage_filename, e)
            raise DocumentLoaderError(f"Failed to load document: {str(e)}")

    def _is_supported(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def _load_pdf(self, file_path: Path) -> List[dict]:
        """Extract text from PDF with page numbers."""
        pages = []
        try:
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append({
                            "content": text,
                            "page_number": page_num,
                            "metadata": {"source": file_path.name, "page": page_num},
                        })
            logger.info("Loaded PDF: %s (%d pages)", file_path.name, len(pages))
            return pages
        except Exception as e:
            logger.exception("PDF extraction failed for %s: %s", file_path.name, e)
            raise DocumentLoaderError(f"PDF extraction failed: {str(e)}")

    def _load_txt(self, file_path: Path) -> List[dict]:
        """Load text file as single page."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            if not content.strip():
                return []
            logger.info("Loaded TXT: %s", file_path.name)
            return [{
                "content": content,
                "page_number": 1,
                "metadata": {"source": file_path.name, "page": 1},
            }]
        except UnicodeDecodeError:
            # Try latin-1 fallback
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()
            return [{
                "content": content,
                "page_number": 1,
                "metadata": {"source": file_path.name, "page": 1},
            }]
        except Exception as e:
            logger.exception("TXT loading failed for %s: %s", file_path.name, e)
            raise DocumentLoaderError(f"TXT loading failed: {str(e)}")