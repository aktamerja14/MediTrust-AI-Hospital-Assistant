"""FAISS Vector Store Implementation.

Local file-based vector store using Facebook AI Similarity Search (FAISS).
Suitable for development and small-to-medium production workloads.
Persists index and metadata to disk.
"""
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any

import faiss
import numpy as np

from app.ai.vectorstore.base import (
    BaseVectorStore,
    VectorDocument,
    SearchResult,
    VectorStoreError,
)

logger = logging.getLogger("meditrust.ai.vectorstore.faiss")


class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store with metadata persistence."""

    def __init__(self, dimension: int, path: str = "data/vector_store"):
        super().__init__(dimension, path)
        self._index: Optional[faiss.Index] = None
        self._documents: Dict[str, VectorDocument] = {}  # id -> document
        self._id_to_index: Dict[str, int] = {}  # document id -> FAISS index position

    @property
    def store_name(self) -> str:
        return "faiss"

    def initialize(self) -> None:
        """Initialize or load FAISS index from disk."""
        index_path = Path(self.path) / "index.faiss"
        meta_path = Path(self.path) / "metadata.json"
        id_map_path = Path(self.path) / "id_map.pkl"

        if index_path.exists() and meta_path.exists():
            self._load_from_disk(index_path, meta_path, id_map_path)
        else:
            self._create_new_index()
            self._ensure_directory()

        self._initialized = True
        logger.info("FAISS vector store initialized: %d documents, dim=%d",
                    self._index.ntotal, self.dimension)

    def _ensure_directory(self) -> None:
        """Create storage directory if it doesn't exist."""
        Path(self.path).mkdir(parents=True, exist_ok=True)

    def _create_new_index(self) -> None:
        """Create a new FAISS index (IndexFlatIP for cosine similarity)."""
        # IndexFlatIP = inner product (cosine similarity when vectors are normalized)
        self._index = faiss.IndexFlatIP(self.dimension)
        self._documents = {}
        self._id_to_index = {}

    def _load_from_disk(
        self,
        index_path: Path,
        meta_path: Path,
        id_map_path: Path,
    ) -> None:
        """Load index and metadata from disk."""
        try:
            self._index = faiss.read_index(str(index_path))
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                self._documents = {
                    doc_id: VectorDocument(**doc_data)
                    for doc_id, doc_data in meta.items()
                }
            with open(id_map_path, "rb") as f:
                self._id_to_index = pickle.load(f)

            logger.info("Loaded FAISS index from disk: %d vectors", self._index.ntotal)
        except Exception as e:
            logger.warning("Failed to load FAISS index, creating new: %s", e)
            self._create_new_index()

    def _save_to_disk(self) -> None:
        """Persist index and metadata to disk."""
        if not self._initialized:
            return

        self._ensure_directory()
        index_path = Path(self.path) / "index.faiss"
        meta_path = Path(self.path) / "metadata.json"
        id_map_path = Path(self.path) / "id_map.pkl"

        try:
            faiss.write_index(self._index, str(index_path))
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {doc_id: doc.__dict__ for doc_id, doc in self._documents.items()},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            with open(id_map_path, "wb") as f:
                pickle.dump(self._id_to_index, f)
            logger.debug("FAISS index persisted to %s", self.path)
        except Exception as e:
            logger.error("Failed to persist FAISS index: %s", e)
            raise VectorStoreError(f"Persist failed: {str(e)}", self.store_name, e)

    def _normalize_embedding(self, embedding: List[float]) -> np.ndarray:
        """Normalize embedding for cosine similarity."""
        arr = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr

    def add_documents(self, documents: List[VectorDocument]) -> None:
        """Add documents to the vector store."""
        if not self._initialized:
            self.initialize()

        if not documents:
            return

        # Normalize embeddings for cosine similarity
        embeddings = np.array([
            self._normalize_embedding(doc.embedding)
            for doc in documents
        ], dtype=np.float32)

        # Validate dimensions
        if embeddings.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}",
                self.store_name,
            )

        # Add to FAISS index
        start_idx = self._index.ntotal
        self._index.add(embeddings)

        # Track documents
        for i, doc in enumerate(documents):
            faiss_idx = start_idx + i
            self._documents[doc.id] = doc
            self._id_to_index[doc.id] = faiss_idx

        logger.info("Added %d documents to FAISS index (total: %d)", len(documents), self._index.ntotal)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar documents using cosine similarity."""
        if not self._initialized:
            self.initialize()

        if self._index.ntotal == 0:
            return []

        # Normalize query embedding
        query_vec = self._normalize_embedding(query_embedding).reshape(1, -1)

        # Search (k = top_k * 2 to account for filtering, then filter)
        search_k = min(top_k * 3, self._index.ntotal)
        scores, indices = self._index.search(query_vec, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            # Find document by FAISS index
            doc_id = None
            for d_id, d_idx in self._id_to_index.items():
                if d_idx == idx:
                    doc_id = d_id
                    break

            if doc_id is None:
                continue

            doc = self._documents.get(doc_id)
            if doc is None:
                continue

            # Apply metadata filter if provided
            if filter_metadata:
                match = True
                for key, value in filter_metadata.items():
                    if doc.metadata.get(key) != value:
                        match = False
                        break
                if not match:
                    continue

            results.append(SearchResult(document=doc, score=float(score)))

            if len(results) >= top_k:
                break

        return results

    def delete_document(self, document_id: str) -> bool:
        """Delete a document by ID.

        Note: FAISS doesn't support true deletion. We mark as deleted
        and rebuild index on next persist if needed.
        For now, just remove from metadata tracking.
        """
        if document_id in self._documents:
            del self._documents[document_id]
            if document_id in self._id_to_index:
                del self._id_to_index[document_id]
            logger.info("Marked document %s for deletion", document_id)
            return True
        return False

    def persist(self) -> None:
        """Persist the index to disk."""
        self._save_to_disk()

    def get_stats(self) -> Dict[str, Any]:
        """Return store statistics."""
        return {
            "store": self.store_name,
            "total_vectors": self._index.ntotal if self._index else 0,
            "tracked_documents": len(self._documents),
            "dimension": self.dimension,
            "path": self.path,
        }