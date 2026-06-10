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

# --- Azure OpenAI (via the InnoWings APIM gateway) -------------------------
# These are Azure API Management gateways, NOT stock Azure OpenAI resources:
# the path is  {BASE}/deployments/{model}/chat/completions?api-version=...
# with NO "/openai/" segment, authenticated via the `api-key` header. The stock
# AzureOpenAI client injects "/openai/" (-> 404), so we drive a plain OpenAI
# client against this base instead (see rag/llm.py).
#
# NOTE: chat and embeddings live on DIFFERENT gateways. `.env` typically sets
# AZURE_OPENAI_ENDPOINT to the *embedding* gateway (sig-embedding), so the chat
# base has its own var (AZURE_CHAT_ENDPOINT) to avoid collision. Only the key
# is required at runtime; the rest have working defaults.
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
CHAT_ENDPOINT = os.getenv(
    "AZURE_CHAT_ENDPOINT",
    "https://api-iw.azure-api.net/sig-shared-jpeast",   # chat gateway base, no path
)
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# --- Latency controls ------------------------------------------------------
# Target: a full answer in <= 25s. The 600s SDK default with 2 silent retries
# can let one hung call blow any budget, so we disable retries and cap each
# call. The critical path is three sequential stages:
#     (decompose || keywords)  ->  rerank  ->  answer
# Worst case = LLM_TIMEOUT + LLM_TIMEOUT + ANSWER_TIMEOUT + retrieval(~0.6s).
# With the defaults below: 7 + 7 + 9 + 0.6 ~= 23.6s, inside the 25s budget.
# On a timeout a stage degrades gracefully (e.g. keeps retrieval order) rather
# than crashing. Raise these if your endpoint is fast and you want fuller answers.
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "7"))         # decompose / keywords / rerank
ANSWER_TIMEOUT = float(os.getenv("ANSWER_TIMEOUT", "9"))   # final answer generation
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "0"))   # no retries -> predictable bound

# --- Retrieval knobs -------------------------------------------------------
# How many candidates each stage pulls before the reranker trims the pool.
SEMANTIC_TOP_K = int(os.getenv("SEMANTIC_TOP_K", "8"))      # dense vector hits per sub-query
KEYWORD_TOP_K = int(os.getenv("KEYWORD_TOP_K", "5"))        # FTS hits per generated keyword
MAX_SUBQUERIES = int(os.getenv("MAX_SUBQUERIES", "4"))      # cap on query decomposition
MAX_KEYWORDS = int(os.getenv("MAX_KEYWORDS", "6"))          # cap on generated keyword terms
CANDIDATE_POOL = int(os.getenv("CANDIDATE_POOL", "30"))     # max merged candidates sent to reranker
# Kept high enough that "list all X" questions (e.g. all SIGs, which are split
# across many small chunks) retain enough coverage to enumerate fully. Still
# far inside the latency budget. Lower it for tighter, more precise contexts.
FINAL_TOP_N = int(os.getenv("FINAL_TOP_N", "10"))          # chunks kept for answer generation


def require_api_key() -> str:
    """Return the Azure key or raise a clear error (only needed for LLM calls)."""
    if not AZURE_OPENAI_API_KEY:
        raise RuntimeError(
            "Missing Azure OpenAI credentials. Set AZURE_OPENAI_API_KEY in your "
            "environment or a .env file at the project root."
        )
    return AZURE_OPENAI_API_KEY
