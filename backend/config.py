import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(BASE_DIR / ".env")


class Settings:
    """Application settings and constants."""

    # -------------------------
    # Groq
    # -------------------------
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_PRIMARY_MODEL: str = os.getenv(
        "GROQ_PRIMARY_MODEL",
        "llama-3.1-70b-versatile",
    )
    GROQ_FALLBACK_MODEL: str = os.getenv(
        "GROQ_FALLBACK_MODEL",
        "llama-3.1-8b-instant",
    )

    # -------------------------
    # Embeddings
    # -------------------------
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    # -------------------------
    # Paths
    # -------------------------
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_PDF_DIR: Path = DATA_DIR / "raw_pdfs"
    CHUNKED_TEXT_DIR: Path = DATA_DIR / "chunked_text"
    VECTOR_DB_DIR: Path = BASE_DIR / "vectordb" / "chroma"

    # -------------------------
    # Chunking
    # -------------------------
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150

    # -------------------------
    # Retrieval
    # -------------------------
    TOP_K: int = 5
    SCORE_THRESHOLD: float = 0.3


settings = Settings()