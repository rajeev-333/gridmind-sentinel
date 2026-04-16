"""
Document Loader Module — src/rag/document_loader.py

Loads IEC/IEEE standards documents from the data/standards/ directory and
splits them into chunks using two strategies:
    - Strategy A: Recursive character splitter (chunk_size=512, overlap=64)
    - Strategy B: Semantic sentence splitter via sentence-transformers

Each chunk is returned as a LangChain Document object with metadata
(source filename, chunk_id, chunking strategy).

Connection to system:
    - Called by pipeline.py to prepare documents for embedding and indexing.
    - Chunk size and overlap configurable via settings (src/utils/config.py).
"""

from pathlib import Path
from typing import Literal

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_standards_documents(directory: Path | None = None) -> list[Document]:
    """
    Load all .txt files from the standards directory into LangChain Documents.

    Args:
        directory: Path to standards directory. Defaults to settings.standards_path.

    Returns:
        List of Document objects with page_content and metadata (source, filename).
    """
    standards_dir = directory or settings.standards_path

    if not standards_dir.exists():
        raise FileNotFoundError(f"Standards directory not found: {standards_dir}")

    documents = []
    txt_files = sorted(standards_dir.glob("*.txt"))

    if not txt_files:
        raise ValueError(f"No .txt files found in {standards_dir}")

    for filepath in txt_files:
        content = filepath.read_text(encoding="utf-8")
        doc = Document(
            page_content=content,
            metadata={
                "source": str(filepath),
                "filename": filepath.name,
            },
        )
        documents.append(doc)
        logger.info(f"Loaded document: {filepath.name} ({len(content)} chars)")

    logger.info(f"Loaded {len(documents)} documents from {standards_dir}")
    return documents


def chunk_documents_recursive(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Strategy A: Split documents using recursive character splitter.

    Splits on paragraph boundaries, then sentences, then words, then characters.
    Maintains overlap between adjacent chunks for context continuity.

    Args:
        documents: List of full-text Document objects.
        chunk_size: Maximum chunk size in characters (default: settings.CHUNK_SIZE = 512).
        chunk_overlap: Overlap between adjacent chunks (default: settings.CHUNK_OVERLAP = 64).

    Returns:
        List of chunked Document objects with added metadata (chunk_id, strategy).
    """
    size = chunk_size or settings.CHUNK_SIZE
    overlap = chunk_overlap or settings.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc.page_content)
        for i, split_text in enumerate(splits):
            chunk_doc = Document(
                page_content=split_text,
                metadata={
                    **doc.metadata,
                    "chunk_id": i,
                    "strategy": "recursive",
                    "chunk_size": size,
                    "chunk_overlap": overlap,
                },
            )
            chunks.append(chunk_doc)

    logger.info(
        f"Recursive chunking: {len(documents)} docs → {len(chunks)} chunks "
        f"(size={size}, overlap={overlap})"
    )
    return chunks


def chunk_documents_semantic(documents: list[Document]) -> list[Document]:
    """
    Strategy B: Split documents by semantic sentence boundaries.

    Uses spaCy-style sentence splitting (via simple regex heuristics) to
    keep semantically coherent sentences together, then groups sentences
    into chunks that fit within the target chunk size.

    Args:
        documents: List of full-text Document objects.

    Returns:
        List of chunked Document objects with added metadata (chunk_id, strategy).
    """
    import re

    target_size = settings.CHUNK_SIZE
    chunks = []

    for doc in documents:
        # Split into sentences using regex (handles abbreviations reasonably)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', doc.page_content)

        current_chunk = []
        current_length = 0
        chunk_id = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If adding this sentence exceeds target, save current chunk
            if current_length + len(sentence) > target_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_id": chunk_id,
                        "strategy": "semantic",
                        "chunk_size": len(chunk_text),
                    },
                )
                chunks.append(chunk_doc)
                chunk_id += 1
                current_chunk = []
                current_length = 0

            current_chunk.append(sentence)
            current_length += len(sentence) + 1  # +1 for space

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={
                    **doc.metadata,
                    "chunk_id": chunk_id,
                    "strategy": "semantic",
                    "chunk_size": len(chunk_text),
                },
            )
            chunks.append(chunk_doc)

    logger.info(
        f"Semantic chunking: {len(documents)} docs → {len(chunks)} chunks"
    )
    return chunks


def load_and_chunk(
    strategy: Literal["recursive", "semantic"] = "recursive",
    directory: Path | None = None,
) -> list[Document]:
    """
    Convenience function: load documents and chunk with the specified strategy.

    Args:
        strategy: "recursive" (Strategy A) or "semantic" (Strategy B).
        directory: Optional path to standards directory.

    Returns:
        List of chunked Document objects ready for embedding.
    """
    documents = load_standards_documents(directory)

    if strategy == "recursive":
        return chunk_documents_recursive(documents)
    elif strategy == "semantic":
        return chunk_documents_semantic(documents)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
