"""
RAG Pipeline Package

Implements a hybrid Retrieval-Augmented Generation pipeline for power grid
standards documents (IEC 61968, IEC 61850, IEEE C37, IEEE P2030).

Components:
    - document_loader: Load and chunk standards documents
    - embeddings: Sentence-transformer embedding wrapper
    - vector_store: FAISS + BM25 hybrid retriever with RRF fusion
    - reranker: Cross-encoder reranking
    - pipeline: End-to-end RAG orchestrator
"""
