"""Document Loader for RAG Pipeline.

Loads and extracts text from PDF, TXT, and JSON files.
"""
import json
import logging
from pathlib import Path
from typing import Any, List, Optional

import pypdf

from app.core.config import settings

logger = logging.getLogger("meditrust.ai.rag.document_loader")


class DocumentLoaderError(Exception):
    """Exception for document loading errors."""
    pass


class DocumentLoader:
    """Loads and extracts text from documents."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".json"}

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
            elif file_path.suffix.lower() == ".json":
                return self._load_json(file_path)
            else:
                return self._load_txt(file_path)
        except DocumentLoaderError:
            raise
        except Exception as e:
            logger.exception("Failed to load document %s: %s", storage_filename, e)
            raise DocumentLoaderError(f"Failed to load document: {str(e)}")

    def _is_supported(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def _load_json(self, file_path: Path) -> List[dict]:
        """Load JSON file and convert to readable text suitable for RAG.

        Supports:
        - JSON arrays of objects (e.g., Q&A pairs)
        - JSON objects with nested structures
        - Preserves key-value relationships
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except UnicodeDecodeError:
            raise DocumentLoaderError("JSON file must be UTF-8 encoded")
        except json.JSONDecodeError as e:
            raise DocumentLoaderError(f"Invalid JSON syntax: {str(e)}")
        except Exception as e:
            logger.exception("JSON loading failed for %s: %s", file_path.name, e)
            raise DocumentLoaderError(f"Failed to load JSON: {str(e)}")

        if data is None:
            raise DocumentLoaderError("JSON content cannot be null")

        # Convert JSON to readable text
        text_content = self._json_to_text(data, file_path.name)

        if not text_content.strip():
            raise DocumentLoaderError("JSON file has no extractable content")

        logger.info("Loaded JSON: %s", file_path.name)
        return [{
            "content": text_content,
            "page_number": 1,
            "metadata": {"source": file_path.name, "page": 1, "file_type": "json"},
        }]

    def _json_to_text(self, data: Any, source_name: str, prefix: str = "") -> str:
        """Recursively convert JSON data to readable text preserving key-value relationships."""
        lines = []

        def _process(value: Any, current_prefix: str = "") -> None:
            if isinstance(value, dict):
                for key, val in value.items():
                    new_prefix = f"{current_prefix}{key}" if current_prefix else key
                    if isinstance(val, (dict, list)):
                        _process(val, new_prefix + ": ")
                    else:
                        lines.append(f"{new_prefix}: {val}")
            elif isinstance(value, list):
                for idx, item in enumerate(value):
                    if isinstance(item, (dict, list)):
                        _process(item, f"{current_prefix}[{idx}]: ")
                    else:
                        lines.append(f"{current_prefix}[{idx}]: {item}")
            else:
                if current_prefix:
                    lines.append(f"{current_prefix}{value}")
                else:
                    lines.append(str(value))

        _process(data, prefix)
        return "\n".join(lines)

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