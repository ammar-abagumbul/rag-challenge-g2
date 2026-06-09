"""Central configuration for the RAG retrieval pipeline.

Everything that might differ between machines (paths, Azure endpoint, model
names, retrieval depths) lives here so the rest of the code stays declarative.
Values can be overridden with environment variables / a .env file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths -----------------------------------------------------------------
# The ChromaDB was populated by populate_chroma.py at <cwd>/chroma_db/chroma_db.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", PROJECT_ROOT / "chroma_db" / "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "hku_innowings_scraper")

# --- Azure OpenAI ----------------------------------------------------------
# Mirrors the known-good gateway config from the reference chatbot. Only the
# API key is required at runtime; the rest have working defaults.
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT",
    "https://api-iw.azure-api.net/sig-shared-jpeast/deployments/gpt-4o-mini/"
    "chat/completions?api-version=2025-01-01-preview",
)
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# --- Retrieval knobs -------------------------------------------------------
# How many candidates each stage pulls before the reranker trims the pool.
SEMANTIC_TOP_K = int(os.getenv("SEMANTIC_TOP_K", "8"))      # dense vector hits per sub-query
KEYWORD_TOP_K = int(os.getenv("KEYWORD_TOP_K", "5"))        # FTS hits per generated keyword
MAX_SUBQUERIES = int(os.getenv("MAX_SUBQUERIES", "4"))      # cap on query decomposition
MAX_KEYWORDS = int(os.getenv("MAX_KEYWORDS", "6"))          # cap on generated keyword terms
CANDIDATE_POOL = int(os.getenv("CANDIDATE_POOL", "30"))     # max merged candidates sent to reranker
FINAL_TOP_N = int(os.getenv("FINAL_TOP_N", "6"))           # chunks kept for answer generation


def require_api_key() -> str:
    """Return the Azure key or raise a clear error (only needed for LLM calls)."""
    if not AZURE_OPENAI_API_KEY:
        raise RuntimeError(
            "Missing Azure OpenAI credentials. Set AZURE_OPENAI_API_KEY in your "
            "environment or a .env file at the project root."
        )
    return AZURE_OPENAI_API_KEY
