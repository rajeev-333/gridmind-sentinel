"""
Hybrid Vector Store — src/rag/vector_store.py

Implements a hybrid retrieval system combining:
    1. FAISS (dense vector search) — semantic similarity via embeddings
    2. BM25 (sparse keyword search) — lexical matching via tf-idf
    3. Reciprocal Rank Fusion (RRF) — combines both ranked lists into one

The FAISS index is persisted to disk (data/faiss_index/) on first build
and loaded from disk on subsequent runs, avoiding redundant recomputation.

Connection to system:
    - Called by pipeline.py for hybrid retrieval.
    - Uses embeddings.py for vector encoding.
    - Document chunks come from document_loader.py.
"""

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from langchain.schema import Document
from rank_bm25 import BM25Okapi

from src.rag.embeddings import LocalEmbeddings
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """
    Hybrid retriever combining FAISS (dense) and BM25 (sparse) with RRF fusion.

    Attributes:
        documents: List of chunked Document objects.
        faiss_index: FAISS inner-product index for dense retrieval.
        bm25: BM25Okapi index for sparse keyword retrieval.
        embeddings: LocalEmbeddings instance for encoding.
    """

    def __init__(self):
        self.documents: list[Document] = []
        self.faiss_index: faiss.IndexFlatIP | None = None
        self.bm25: BM25Okapi | None = None
        self.embeddings: LocalEmbeddings = LocalEmbeddings.get_instance()
        self._is_built = False

    def build_index(self, documents: list[Document]) -> None:
        """
        Build both FAISS and BM25 indices from document chunks.

        Args:
            documents: List of chunked Document objects to index.
        """
        if not documents:
            raise ValueError("Cannot build index from empty document list")

        self.documents = documents
        texts = [doc.page_content for doc in documents]

        # --- Build FAISS index ---
        logger.info(f"Building FAISS index from {len(texts)} chunks...")
        embeddings_matrix = np.array(
            self.embeddings.embed_documents(texts), dtype=np.float32
        )

        dimension = embeddings_matrix.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product (cosine sim for normalized vecs)
        self.faiss_index.add(embeddings_matrix)
        logger.info(
            f"FAISS index built: {self.faiss_index.ntotal} vectors, dim={dimension}"
        )

        # --- Build BM25 index ---
        logger.info("Building BM25 index...")
        tokenized_texts = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        logger.info(f"BM25 index built: {len(tokenized_texts)} documents")

        self._is_built = True

    def save_index(self, index_dir: Path | None = None) -> None:
        """
        Persist the FAISS index and document metadata to disk.

        Saves:
            - faiss_index.bin: The FAISS index binary
            - documents.pkl: Pickled document list
            - metadata.json: Index metadata (doc count, dimension)

        Args:
            index_dir: Directory to save to. Defaults to settings.faiss_index_path.
        """
        if not self._is_built:
            raise RuntimeError("Cannot save — index has not been built yet")

        save_dir = index_dir or settings.faiss_index_path
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss_path = save_dir / "faiss_index.bin"
        faiss.write_index(self.faiss_index, str(faiss_path))
        logger.info(f"FAISS index saved to {faiss_path}")

        # Save documents (for BM25 rebuild and result lookup)
        docs_path = save_dir / "documents.pkl"
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)
        logger.info(f"Documents saved to {docs_path}")

        # Save metadata
        meta_path = save_dir / "metadata.json"
        metadata = {
            "num_documents": len(self.documents),
            "dimension": self.faiss_index.d,
            "num_vectors": self.faiss_index.ntotal,
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {meta_path}")

    def load_index(self, index_dir: Path | None = None) -> bool:
        """
        Load a previously saved FAISS index and documents from disk.

        Args:
            index_dir: Directory to load from. Defaults to settings.faiss_index_path.

        Returns:
            True if loaded successfully, False if no saved index found.
        """
        load_dir = index_dir or settings.faiss_index_path
        faiss_path = load_dir / "faiss_index.bin"
        docs_path = load_dir / "documents.pkl"

        if not faiss_path.exists() or not docs_path.exists():
            logger.info("No saved index found — will need to build from scratch")
            return False

        # Load FAISS index
        self.faiss_index = faiss.read_index(str(faiss_path))
        logger.info(
            f"FAISS index loaded: {self.faiss_index.ntotal} vectors, "
            f"dim={self.faiss_index.d}"
        )

        # Load documents
        with open(docs_path, "rb") as f:
            self.documents = pickle.load(f)
        logger.info(f"Loaded {len(self.documents)} documents")

        # Rebuild BM25 from loaded documents
        tokenized_texts = [
            doc.page_content.lower().split() for doc in self.documents
        ]
        self.bm25 = BM25Okapi(tokenized_texts)
        logger.info("BM25 index rebuilt from loaded documents")

        self._is_built = True
        return True

    def search_faiss(self, query: str, top_k: int = 20) -> list[tuple[Document, float]]:
        """
        Search using FAISS (dense vector similarity).

        Args:
            query: Query string.
            top_k: Number of results to return.

        Returns:
            List of (Document, score) tuples sorted by descending similarity.
        """
        if not self._is_built:
            raise RuntimeError("Index not built — call build_index() or load_index() first")

        query_embedding = np.array(
            [self.embeddings.embed_query(query)], dtype=np.float32
        )
        scores, indices = self.faiss_index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing results
                continue
            results.append((self.documents[idx], float(score)))

        return results

    def search_bm25(self, query: str, top_k: int = 20) -> list[tuple[Document, float]]:
        """
        Search using BM25 (sparse keyword matching).

        Args:
            query: Query string.
            top_k: Number of results to return.

        Returns:
            List of (Document, score) tuples sorted by descending BM25 score.
        """
        if not self._is_built:
            raise RuntimeError("Index not built — call build_index() or load_index() first")

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices by score
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include results with non-zero BM25 score
                results.append((self.documents[idx], float(scores[idx])))

        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int | None = None,
        rrf_k: int = 60,
    ) -> list[Document]:
        """
        Hybrid search using Reciprocal Rank Fusion (RRF) of FAISS + BM25.

        RRF formula: RRF_score(d) = Σ 1 / (k + rank(d))
        where k is a constant (typically 60) and the sum is over all ranking lists.

        Args:
            query: Query string.
            top_k: Number of final results to return. Defaults to settings.RETRIEVAL_TOP_K.
            rrf_k: RRF smoothing constant (default: 60).

        Returns:
            List of Document objects sorted by fused RRF score (descending).
        """
        retrieval_k = top_k or settings.RETRIEVAL_TOP_K

        # Get results from both retrievers
        faiss_results = self.search_faiss(query, top_k=retrieval_k)
        bm25_results = self.search_bm25(query, top_k=retrieval_k)

        # Build RRF score map (keyed by document index in self.documents)
        rrf_scores: dict[int, float] = {}

        for rank, (doc, _score) in enumerate(faiss_results):
            doc_idx = self.documents.index(doc)
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (rrf_k + rank + 1)

        for rank, (doc, _score) in enumerate(bm25_results):
            doc_idx = self.documents.index(doc)
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (rrf_k + rank + 1)

        # Sort by RRF score (descending) and return top documents
        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
        top_docs = [self.documents[i] for i in sorted_indices[:retrieval_k]]

        logger.info(
            f"Hybrid search: query='{query[:50]}...' — "
            f"FAISS={len(faiss_results)}, BM25={len(bm25_results)}, "
            f"fused={len(top_docs)} results"
        )
        return top_docs

    @property
    def is_built(self) -> bool:
        """Whether the index has been built or loaded."""
        return self._is_built

    @property
    def num_documents(self) -> int:
        """Number of indexed documents."""
        return len(self.documents)
