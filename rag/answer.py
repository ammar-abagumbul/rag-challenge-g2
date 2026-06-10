"""Public RAG entry points: retrieve -> rerank -> generate.

Matches the signature expected by the challenge harness:
    from rag import generate_rag_answers
    answers = generate_rag_answers(["question 1", "question 2"])
"""

import time
from typing import List, Tuple

from . import llm, retrieval


def _as_question(item) -> str:
    """Coerce one loaded item into a question string.

    The competition file is expected to be a JSON array of strings, but be
    tolerant of array-of-objects (e.g. {"question": "..."}) so an unexpected
    shape at exam time can't crash the run.
    """
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("question", "query", "q", "text", "prompt"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val
    return str(item)


def rag_answer(question: str, *, verbose: bool = True) -> str:
    """Answer a single question with the full advanced-retrieval pipeline."""
    start = time.perf_counter()
    chunks = retrieval.retrieve(question, verbose=verbose)
    contexts = [c.text for c in chunks]

    t0 = time.perf_counter()
    answer = llm.generate_answer(question, contexts)
    if verbose:
        print(f"  [t] answer: {time.perf_counter() - t0:.2f}s "
              f"| total: {time.perf_counter() - start:.2f}s")
    return answer


def generate_rag_answers(questions: List[str]) -> List[Tuple[str, str]]:
    """Answer a batch of questions.

    Returns a list of (question, answer) pairs.
    """
    # Tolerate a top-level wrapper object like {"questions": [...]} too.
    if isinstance(questions, dict):
        for key in ("questions", "data", "items"):
            if isinstance(questions.get(key), list):
                questions = questions[key]
                break

    pairs: List[Tuple[str, str]] = []
    for raw in questions:
        question = _as_question(raw)
        preview = question[:80] + ("..." if len(question) > 80 else "")
        print(f"\n>> Answering: {preview}")
        pairs.append((question, rag_answer(question)))
    return pairs
