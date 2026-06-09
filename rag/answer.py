"""Public RAG entry points: retrieve -> rerank -> generate.

Matches the signature expected by the challenge harness:
    from rag import generate_rag_answers
    answers = generate_rag_answers(["question 1", "question 2"])
"""

from typing import List, Tuple

from . import llm, retrieval


def rag_answer(question: str, *, verbose: bool = True) -> str:
    """Answer a single question with the full advanced-retrieval pipeline."""
    chunks = retrieval.retrieve(question, verbose=verbose)
    contexts = [c.text for c in chunks]
    return llm.generate_answer(question, contexts)


def generate_rag_answers(questions: List[str]) -> List[Tuple[str, str]]:
    """Answer a batch of questions.

    Returns a list of (question, answer) pairs.
    """
    pairs: List[Tuple[str, str]] = []
    for question in questions:
        preview = question[:80] + ("..." if len(question) > 80 else "")
        print(f"\n🤖 Answering: {preview}")
        pairs.append((question, rag_answer(question)))
    return pairs
