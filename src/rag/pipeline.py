"""
RAG Pipeline Orchestrator — src/rag/pipeline.py

End-to-end RAG pipeline that:
    1. Loads standards documents from data/standards/
    2. Chunks them using the configured strategy (recursive or semantic)
    3. Builds FAISS + BM25 hybrid index (or loads from disk if available)
    4. Performs hybrid retrieval on a query
    5. Reranks results with cross-encoder
    6. Returns top-5 relevant chunks

This is the primary entry point for RAG queries in the GridMind Sentinel system.

Usage:
    from src.rag.pipeline import RAGPipeline

    pipeline = RAGPipeline()
    pipeline.initialize()  # Build or load index
    results = pipeline.query("What is the procedure for voltage sag recovery?")

Connection to system:
    - Called by agents/remediation.py (Phase 3) via the rag_search tool.
    - Uses all other modules in src/rag/ as building blocks.
    - Index persisted to data/faiss_index/ for fast startup.
"""

from pathlib import Path
from typing import Literal

from langchain.schema import Document

from src.rag.document_loader import load_and_chunk
from src.rag.embeddings import LocalEmbeddings
from src.rag.reranker import DocumentReranker
from src.rag.vector_store import HybridRetriever
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """
    End-to-end Hybrid RAG Pipeline for power grid standards retrieval.

    Flow: Load docs → Chunk → Embed → Hybrid Retrieve (FAISS + BM25 + RRF) → Rerank → Top-5

    Attributes:
        retriever: HybridRetriever instance (FAISS + BM25).
        reranker: DocumentReranker instance (cross-encoder).
        strategy: Chunking strategy ("recursive" or "semantic").
        is_initialized: Whether the pipeline has been initialized.
    """

    def __init__(
        self,
        strategy: Literal["recursive", "semantic"] = "recursive",
        standards_dir: Path | None = None,
        index_dir: Path | None = None,
    ):
        """
        Create a RAG pipeline instance.

        Args:
            strategy: Document chunking strategy. "recursive" (default) or "semantic".
            standards_dir: Path to standards documents. Defaults to settings.
            index_dir: Path to FAISS index directory. Defaults to settings.
        """
        self.strategy = strategy
        self.standards_dir = standards_dir
        self.index_dir = index_dir
        self.retriever = HybridRetriever()
        self.reranker = DocumentReranker.get_instance()
        self.is_initialized = False

    def initialize(self, force_rebuild: bool = False) -> None:
        """
        Initialize the pipeline: load existing index or build from scratch.

        Args:
            force_rebuild: If True, rebuild the index even if a saved one exists.
        """
        index_path = self.index_dir or settings.faiss_index_path

        # Try loading existing index
        if not force_rebuild and self.retriever.load_index(index_path):
            logger.info(
                f"Pipeline initialized from saved index "
                f"({self.retriever.num_documents} chunks)"
            )
            self.is_initialized = True
            return

        # Build from scratch
        logger.info("Building RAG index from scratch...")
        chunks = load_and_chunk(
            strategy=self.strategy,
            directory=self.standards_dir,
        )

        self.retriever.build_index(chunks)
        self.retriever.save_index(index_path)

        logger.info(
            f"Pipeline initialized: {self.retriever.num_documents} chunks indexed, "
            f"strategy={self.strategy}"
        )
        self.is_initialized = True

    def query(
        self,
        question: str,
        top_k: int | None = None,
        retrieval_k: int | None = None,
    ) -> list[Document]:
        """
        Run a full RAG query: hybrid retrieve → rerank → return top-k.

        Args:
            question: Natural language query about power grid standards.
            top_k: Number of final reranked results. Defaults to settings.RAG_TOP_K (5).
            retrieval_k: Number of candidates from hybrid retrieval before reranking.
                        Defaults to settings.RETRIEVAL_TOP_K (20).

        Returns:
            List of top-k Document objects sorted by relevance.

        Raises:
            RuntimeError: If pipeline has not been initialized.
        """
        if not self.is_initialized:
            raise RuntimeError(
                "Pipeline not initialized — call initialize() first"
            )

        final_k = top_k or settings.RAG_TOP_K
        retrieve_k = retrieval_k or settings.RETRIEVAL_TOP_K

        logger.info(f"RAG query: '{question[:80]}...'")

        # Step 1: Hybrid retrieval (FAISS + BM25 + RRF)
        candidates = self.retriever.hybrid_search(question, top_k=retrieve_k)

        # Step 2: Cross-encoder reranking
        results = self.reranker.rerank(question, candidates, top_k=final_k)

        logger.info(
            f"RAG query complete: {len(candidates)} candidates → "
            f"{len(results)} final results"
        )

        return results

    def query_with_scores(
        self,
        question: str,
        top_k: int | None = None,
        retrieval_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Run a RAG query and return results with cross-encoder scores.

        Same as query() but includes relevance scores for evaluation/debugging.

        Args:
            question: Natural language query.
            top_k: Number of final results.
            retrieval_k: Number of hybrid retrieval candidates.

        Returns:
            List of (Document, score) tuples sorted by descending relevance.
        """
        if not self.is_initialized:
            raise RuntimeError(
                "Pipeline not initialized — call initialize() first"
            )

        final_k = top_k or settings.RAG_TOP_K
        retrieve_k = retrieval_k or settings.RETRIEVAL_TOP_K

        candidates = self.retriever.hybrid_search(question, top_k=retrieve_k)
        return self.reranker.rerank_with_scores(question, candidates, top_k=final_k)

    def get_stats(self) -> dict:
        """
        Get pipeline statistics for monitoring/evaluation.

        Returns:
            Dict with indexed document count, index status, and configuration.
        """
        return {
            "is_initialized": self.is_initialized,
            "num_chunks": self.retriever.num_documents,
            "chunking_strategy": self.strategy,
            "embedding_model": settings.EMBEDDING_MODEL,
            "reranker_model": settings.RERANKER_MODEL,
            "top_k": settings.RAG_TOP_K,
            "retrieval_k": settings.RETRIEVAL_TOP_K,
        }
