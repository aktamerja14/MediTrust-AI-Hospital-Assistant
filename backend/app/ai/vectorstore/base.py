"""Base Vector Store Interface.

All vector stores must implement this interface.
Allows swapping FAISS for other stores (Chroma, Pinecone, etc.) later.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class VectorDocument:
    """Document with vector embedding and metadata."""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]


@dataclass
class SearchResult:
    """Result from vector similarity search."""
    document: VectorDocument
    score: float  # Higher = more similar (cosine similarity)


class VectorStoreError(Exception):
    """Base exception for vector store errors."""
    def __init__(self, message: str, store: str, original_error: Optional[Exception] = None):
        self.store = store
        self.original_error = original_error
        super().__init__(f"[{store}] {message}")


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""

    def __init__(self, dimension: int, path: str = ""):
        self.dimension = dimension
        self.path = path
        self._initialized = False

    @property
    @abstractmethod
    def store_name(self) -> str:
        """Return the store identifier (e.g., 'faiss')."""
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the vector store (create index, load from disk, etc.)."""
        pass

    @abstractmethod
    def add_documents(self, documents: List[VectorDocument]) -> None:
        """Add documents to the vector store."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar documents.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of SearchResult sorted by similarity (highest first)
        """
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> bool:
        """Delete a document by ID. Returns True if deleted."""
        pass

    @abstractmethod
    def persist(self) -> None:
        """Persist the index to disk."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return store statistics."""
        pass