"""Local Embedding Provider using sentence-transformers.

Runs locally, no API key required, no rate limits.
Models: sentence-transformers/all-MiniLM-L6-v2 (384 dims, fast)
        sentence-transformers/all-mpnet-base-v2 (768 dims, better quality)
"""
import logging
from typing import List, Optional

import numpy as np

from app.ai.embeddings.base import (
    BaseEmbeddingProvider,
    EmbeddingModelNotFoundError,
    EmbeddingProviderError,
)

logger = logging.getLogger("meditrust.ai.embeddings.local")

# Model dimensions mapping (for validation without loading)
MODEL_DIMENSIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "sentence-transformers/all-MiniLM-L12-v2": 384,
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1": 384,
    "sentence-transformers/multi-qa-mpnet-base-dot-v1": 768,
}


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Local sentence-transformers embedding provider."""

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        api_key: str = "",  # Not used for local
        timeout: int = 60,  # Not used for local
        device: str = "cpu",
    ):
        self.device = device
        self._model = None
        super().__init__(model, api_key, timeout)

    def _validate_config(self) -> None:
        if self.model not in MODEL_DIMENSIONS:
            logger.warning(
                "Unknown model '%s', dimension will be determined at runtime",
                self.model
            )

    @property
    def provider_name(self) -> str:
        return "local"

    @property
    def dimension(self) -> int:
        if self._dimension is not None:
            return self._dimension
        # Return known dimension or default
        return MODEL_DIMENSIONS.get(self.model, 384)

    def _load_model(self):
        """Lazy load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading embedding model: %s on %s", self.model, self.device)
                self._model = SentenceTransformer(self.model, device=self.device)
                # Update dimension from actual model
                self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info("Embedding model loaded, dimension: %d", self._dimension)
            except ImportError as e:
                raise EmbeddingProviderError(
                    "sentence-transformers not installed. Run: pip install sentence-transformers",
                    self.provider_name,
                    e,
                )
            except Exception as e:
                logger.exception("Failed to load embedding model: %s", e)
                raise EmbeddingModelNotFoundError(
                    f"Could not load model '{self.model}'",
                    self.provider_name,
                    e,
                )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using local model."""
        if not texts:
            return []

        self._load_model()

        try:
            # sentence-transformers returns numpy array
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32,
            )
            return embeddings.tolist()
        except Exception as e:
            logger.exception("Embedding generation failed: %s", e)
            raise EmbeddingProviderError(
                f"Failed to generate embeddings: {str(e)}",
                self.provider_name,
                e,
            )