"""
Embedding Model Wrapper — src/rag/embeddings.py

Wraps the sentence-transformers all-MiniLM-L6-v2 model to provide a consistent
embedding interface for both documents and queries. Uses a singleton pattern
to avoid reloading the ~80MB model on every call.

Compatible with LangChain's Embeddings interface for seamless integration
with FAISS and other vector stores.

Connection to system:
    - Used by vector_store.py to embed document chunks and queries.
    - Model name configurable via settings.EMBEDDING_MODEL.
"""

from typing import ClassVar

from langchain.schema.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LocalEmbeddings(Embeddings):
    """
    LangChain-compatible embedding wrapper around sentence-transformers.

    Uses all-MiniLM-L6-v2 by default (384-dim embeddings, no API key needed).
    Implements singleton pattern: the model is loaded once and reused.
    """

    _instance: ClassVar["LocalEmbeddings | None"] = None
    _model: ClassVar[SentenceTransformer | None] = None

    def __init__(self, model_name: str | None = None):
        """
        Initialize the embedding model (loads on first call, reuses thereafter).

        Args:
            model_name: HuggingFace model name. Defaults to settings.EMBEDDING_MODEL.
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL

        if LocalEmbeddings._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            LocalEmbeddings._model = SentenceTransformer(self.model_name)
            logger.info(
                f"Embedding model loaded. Dimension: "
                f"{LocalEmbeddings._model.get_sentence_embedding_dimension()}"
            )

    @classmethod
    def get_instance(cls, model_name: str | None = None) -> "LocalEmbeddings":
        """
        Get or create the singleton embedding instance.

        Args:
            model_name: Optional model name override.

        Returns:
            Singleton LocalEmbeddings instance.
        """
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing purposes)."""
        cls._instance = None
        cls._model = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of document texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        if not texts:
            return []

        logger.info(f"Embedding {len(texts)} documents...")
        embeddings = LocalEmbeddings._model.encode(
            texts,
            show_progress_bar=len(texts) > 50,
            batch_size=64,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Generate embedding for a single query text.

        Args:
            text: Query text string.

        Returns:
            Embedding vector as a list of floats.
        """
        embedding = LocalEmbeddings._model.encode(
            [text],
            normalize_embeddings=True,
        )
        return embedding[0].tolist()

    @property
    def dimension(self) -> int:
        """Get the embedding dimension (384 for all-MiniLM-L6-v2)."""
        return LocalEmbeddings._model.get_sentence_embedding_dimension()
