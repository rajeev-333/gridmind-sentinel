"""
Cross-Encoder Reranker — src/rag/reranker.py

Reranks candidate documents from hybrid retrieval using a cross-encoder model
(cross-encoder/ms-marco-MiniLM-L-6-v2). Unlike bi-encoder embeddings, cross-
encoders process the query and document together as a single input, enabling
much more accurate relevance scoring at the cost of throughput.

Applied after hybrid retrieval to select the top-k most relevant chunks
before passing to the LLM for answer generation.

Connection to system:
    - Called by pipeline.py after hybrid_search() results are obtained.
    - Model name configurable via settings.RERANKER_MODEL.
"""

from typing import ClassVar

from langchain.schema import Document
from sentence_transformers import CrossEncoder

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentReranker:
    """
    Cross-encoder reranker for refining hybrid retrieval results.

    Uses cross-encoder/ms-marco-MiniLM-L-6-v2 to score query-document pairs
    and return the top-k most relevant documents.

    Implements singleton pattern to avoid reloading the model.
    """

    _instance: ClassVar["DocumentReranker | None"] = None
    _model: ClassVar[CrossEncoder | None] = None

    def __init__(self, model_name: str | None = None):
        """
        Initialize the cross-encoder model.

        Args:
            model_name: HuggingFace cross-encoder model name.
                       Defaults to settings.RERANKER_MODEL.
        """
        self.model_name = model_name or settings.RERANKER_MODEL

        if DocumentReranker._model is None:
            logger.info(f"Loading reranker model: {self.model_name}")
            DocumentReranker._model = CrossEncoder(self.model_name)
            logger.info("Reranker model loaded")

    @classmethod
    def get_instance(cls, model_name: str | None = None) -> "DocumentReranker":
        """Get or create the singleton reranker instance."""
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing purposes)."""
        cls._instance = None
        cls._model = None

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[Document]:
        """
        Rerank documents by cross-encoder relevance score.

        Args:
            query: The search query.
            documents: Candidate documents from hybrid retrieval.
            top_k: Number of results to return. Defaults to settings.RAG_TOP_K (5).

        Returns:
            List of top-k Document objects sorted by cross-encoder score (descending).
        """
        k = top_k or settings.RAG_TOP_K

        if not documents:
            return []

        if len(documents) <= k:
            # No need to rerank if we have fewer than top_k docs
            logger.info(
                f"Reranker: {len(documents)} docs ≤ top_k={k}, "
                f"scoring all without trimming"
            )

        # Create query-document pairs for cross-encoder
        pairs = [[query, doc.page_content] for doc in documents]

        # Score all pairs
        scores = DocumentReranker._model.predict(pairs)

        # Sort by score (descending) and return top-k
        scored_docs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        top_docs = [doc for doc, _score in scored_docs[:k]]

        logger.info(
            f"Reranker: {len(documents)} candidates → top {len(top_docs)} | "
            f"scores: [{scored_docs[0][1]:.4f} ... {scored_docs[-1][1]:.4f}]"
        )

        return top_docs

    def rerank_with_scores(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Rerank documents and return (Document, score) tuples.

        Useful for debugging and evaluation — shows the cross-encoder score
        for each returned document.

        Args:
            query: The search query.
            documents: Candidate documents from hybrid retrieval.
            top_k: Number of results to return.

        Returns:
            List of (Document, score) tuples sorted by descending score.
        """
        k = top_k or settings.RAG_TOP_K

        if not documents:
            return []

        pairs = [[query, doc.page_content] for doc in documents]
        scores = DocumentReranker._model.predict(pairs)

        scored_docs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [(doc, float(score)) for doc, score in scored_docs[:k]]
