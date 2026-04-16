"""
Memory Package — src/memory/

Implements the memory architecture from PRD Section 5 (Feature 4):
    - Short-term memory: LangGraph state dict (handled by state.py)
    - Long-term memory: ChromaDB collection for incident history
    - Document memory: FAISS index (handled by src/rag/)

Components:
    - long_term: ChromaDB-based incident memory for similar case retrieval
"""
