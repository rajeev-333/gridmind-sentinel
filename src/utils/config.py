"""
Configuration Module — src/utils/config.py

Loads all application settings from environment variables (via .env file).
Uses pydantic-settings for typed, validated configuration with sensible defaults.

Usage:
    from src.utils.config import settings
    print(settings.EMBEDDING_MODEL)  # "all-MiniLM-L6-v2"

Connection to system:
    - Used by every module to access paths, model names, and runtime parameters.
    - Reads from .env file in project root (copy .env.example → .env to configure).
"""

from pathlib import Path
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


# Project root directory (grandparent of this file: src/utils/config.py → src/ → project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Typed application settings loaded from environment variables."""

    model_config = ConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM Configuration ---
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"

    # --- Embedding Model ---
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # --- Reranker Model ---
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- RAG Pipeline ---
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    RAG_TOP_K: int = 5
    RETRIEVAL_TOP_K: int = 20

    # --- Data Paths (relative to project root) ---
    STANDARDS_DIR: str = "data/standards"
    FAISS_INDEX_DIR: str = "data/faiss_index"
    SIMULATED_DATA_DIR: str = "data/simulated"

    # --- FastAPI ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # --- Database ---
    DATABASE_URL: str = "sqlite:///data/gridmind.db"

    # --- Logging ---
    LOG_LEVEL: str = "INFO"

    @property
    def standards_path(self) -> Path:
        """Absolute path to standards documents directory."""
        return PROJECT_ROOT / self.STANDARDS_DIR

    @property
    def faiss_index_path(self) -> Path:
        """Absolute path to FAISS index directory."""
        return PROJECT_ROOT / self.FAISS_INDEX_DIR

    @property
    def simulated_data_path(self) -> Path:
        """Absolute path to simulated telemetry data directory."""
        return PROJECT_ROOT / self.SIMULATED_DATA_DIR


# Singleton settings instance — import this everywhere
settings = Settings()
