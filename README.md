# rag-challenge-g2

Text-based **advanced RAG retrieval** over the HKU InnoWings / InnoAcademy
knowledge base (4,723 chunks in a persisted ChromaDB collection).

On top of plain vector search it implements the three required enhancements:

| # | Requirement | Where | How |
|---|-------------|-------|-----|
| 1 | **Reranking** | `rag/llm.py::rerank` | The merged candidate pool is reordered by an LLM (gpt-4o-mini) judging true relevance to the original question, not raw vector distance. |
| 2 | **Query decomposition / rewriting** | `rag/llm.py::decompose_query` | Complex questions are split into focused, self-contained sub-queries (pronouns resolved); simple ones are lightly rewritten. |
| 3 | **Keyword retrieval** | `rag/llm.py::generate_keywords` + `rag/retrieval.py::keyword_search` | The agent generates distinctive keyword terms, retrieved via ChromaDB's built-in full-text `$contains` filter and fused with the semantic hits. |

## Pipeline

```
question
  ├─ decompose_query (LLM)        → [sub-query, ...]          # requirement 2
  ├─ generate_keywords (LLM)      → [term, ...]               # requirement 3
  ├─ semantic_search per sub-query → dense candidates  ┐
  ├─ keyword_search over terms     → FTS candidates    ┘ merge + dedup
  ├─ rerank (LLM) vs original question                       # requirement 1
  └─ generate_answer from top-N reranked chunks
```

## Layout

```
rag/
  config.py     # paths, Azure config, retrieval knobs (env-overridable)
  llm.py        # decompose / keywords / rerank / answer  (Azure gpt-4o-mini)
  retrieval.py  # semantic + keyword search, orchestration
  answer.py     # public rag_answer / generate_rag_answers
main.py         # CLI runner
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install "chromadb>=1.5.9" "openai>=2.41.0" "python-dotenv>=1.2.2"
copy .env.example .env   # then add your AZURE_OPENAI_API_KEY
```

> The ChromaDB collection was built with `chromadb>=1.5.9`; older versions
> (e.g. an Anaconda base env on 0.5.x) cannot open it. Use the venv above.
> The 384-dim `all-MiniLM-L6-v2` embedding model downloads once on first query.

## Usage

```powershell
# Demo questions:
.\.venv\Scripts\python.exe main.py

# A single question (full answer):
.\.venv\Scripts\python.exe main.py "What are the current SIGs in InnoWings?"

# Inspect retrieval only (no answer generation, no key needed):
.\.venv\Scripts\python.exe main.py --retrieval-only "robotics tech talks"
```

Programmatic:

```python
from rag import generate_rag_answers
for q, a in generate_rag_answers(["What are the current SIGs in InnoWings?"]):
    print(q, "->", a)
```

**Graceful degradation:** if no API key is set, the LLM stages (decomposition,
keyword generation, reranking) are skipped and the system falls back to pure
semantic retrieval rather than crashing.
