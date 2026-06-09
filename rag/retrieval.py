"""Retrieval layer over the persisted ChromaDB collection.

Implements the two retrieval modes that feed the reranker:
  * semantic_search  -> dense vector similarity (the collection's default
                        all-MiniLM-L6-v2 embeddings, resolved automatically).
  * keyword_search   -> ChromaDB full-text `$contains` matching, driven by the
                        LLM-generated keyword terms (requirement 3).

`retrieve` runs query decomposition, fans out both modes over every sub-query,
merges/deduplicates the candidates, and hands the pool to the LLM reranker.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List

import chromadb

from . import config, llm


@dataclass
class Chunk:
    """A single retrieved passage plus where it came from."""
    id: str
    text: str
    metadata: dict
    sources: set = field(default_factory=set)  # e.g. {"semantic", "keyword"}


@lru_cache(maxsize=1)
def get_collection():
    """Open the persisted collection (cached for the process).

    No embedding function is passed: the collection stores `embedding_function:
    default`, so ChromaDB reconstructs the same all-MiniLM-L6-v2 model used at
    population time. This keeps query and index embeddings consistent.
    """
    client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
    return client.get_collection(name=config.COLLECTION_NAME)


# --------------------------------------------------------------------------
# Dense / semantic retrieval
# --------------------------------------------------------------------------
def semantic_search(query: str, top_k: int = config.SEMANTIC_TOP_K) -> List[Chunk]:
    """Top-k nearest neighbours by embedding similarity."""
    res = get_collection().query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas"],
    )
    return _rows_to_chunks(res, source="semantic")


# --------------------------------------------------------------------------
# Keyword / full-text retrieval (requirement 3)
# --------------------------------------------------------------------------
def keyword_search(terms: List[str], per_term: int = config.KEYWORD_TOP_K) -> List[Chunk]:
    """Documents literally containing the generated keyword terms.

    Uses `collection.get(where_document={"$contains": term})`, ChromaDB's
    full-text filter — pure lexical matching with no embedding involved. Each
    term is queried independently and the hits are merged.
    """
    collection = get_collection()
    chunks: Dict[str, Chunk] = {}
    for term in terms:
        term = term.strip()
        if not term:
            continue
        try:
            res = collection.get(
                where_document={"$contains": term},
                limit=per_term,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            print(f"  [keyword] '{term}' failed: {exc}")
            continue
        _merge_get_rows(res, chunks, source="keyword")
    return list(chunks.values())


# --------------------------------------------------------------------------
# Orchestration: decomposition -> hybrid fan-out -> merge -> rerank
# --------------------------------------------------------------------------
def retrieve(question: str, *, verbose: bool = True) -> List[Chunk]:
    """Full advanced-retrieval pipeline for one question.

    Returns the top FINAL_TOP_N chunks after LLM reranking.
    """
    # 1. Query decomposition / rewriting.
    subqueries = llm.decompose_query(question)
    if verbose:
        print(f"  Sub-queries ({len(subqueries)}): {subqueries}")

    # 2. Keyword generation (once for the whole question).
    keywords = llm.generate_keywords(question)
    if verbose:
        print(f"  Keywords: {keywords}")

    # 3. Fan out both retrieval modes and merge into a single candidate pool.
    pool: Dict[str, Chunk] = {}
    for sq in subqueries:
        for chunk in semantic_search(sq):
            _add(pool, chunk)
    for chunk in keyword_search(keywords):
        _add(pool, chunk)

    candidates = list(pool.values())[: config.CANDIDATE_POOL]
    if verbose:
        print(f"  Candidate pool: {len(candidates)} unique chunks")
    if not candidates:
        return []

    # 4. LLM rerank against the ORIGINAL question, then keep the top N.
    order = llm.rerank(question, [c.text for c in candidates], config.FINAL_TOP_N)
    reranked = [candidates[i] for i in order]
    if verbose:
        print(f"  Reranked -> kept {len(reranked)} chunks")
    return reranked


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------
def _add(pool: Dict[str, Chunk], chunk: Chunk) -> None:
    """Insert a chunk into the pool, unioning source tags on duplicates."""
    existing = pool.get(chunk.id)
    if existing:
        existing.sources |= chunk.sources
    else:
        pool[chunk.id] = chunk


def _rows_to_chunks(res: dict, source: str) -> List[Chunk]:
    """Convert a `collection.query` result (nested lists) into Chunks."""
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    chunks = []
    for cid, doc, meta in zip(ids, docs, metas):
        chunks.append(Chunk(id=cid, text=doc or "", metadata=meta or {}, sources={source}))
    return chunks


def _merge_get_rows(res: dict, pool: Dict[str, Chunk], source: str) -> None:
    """Merge a `collection.get` result (flat lists) into a pool dict."""
    ids = res.get("ids") or []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    for cid, doc, meta in zip(ids, docs, metas):
        _add(pool, Chunk(id=cid, text=doc or "", metadata=meta or {}, sources={source}))
