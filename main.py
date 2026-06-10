"""CLI runner for the text-based RAG retrieval pipeline.

Usage:
    python main.py                       # run a couple of demo questions
    python main.py "your question here"  # answer a single question
    python main.py --retrieval-only "q"  # show retrieved chunks, no LLM answer
"""

import sys

from rag import generate_rag_answers
from rag import retrieval


def _retrieval_only(question: str) -> None:
    """Print the reranked chunks without calling the answer generator."""
    chunks = retrieval.retrieve(question)
    print(f"\n=== Top {len(chunks)} chunks for: {question!r} ===")
    for i, c in enumerate(chunks, 1):
        tags = ",".join(sorted(c.sources))
        origin = c.metadata.get("image_name") or c.metadata.get("parent_url", "")
        kind = "image" if c.metadata.get("source_type") == "local_image" else "text"
        print(f"\n[{i}] ({tags}|{kind}) {origin}")
        print(c.text[:300].strip())


def main() -> None:
    args = sys.argv[1:]

    if args and args[0] == "--retrieval-only":
        question = " ".join(args[1:]) or "What are the current SIGs in InnoWings?"
        _retrieval_only(question)
        return

    if args:
        questions = [" ".join(args)]
    else:
        questions = [
            "What are the current SIGs in InnoWings?",
            "Tell me about recent Tech Talks in InnoAcademy.",
        ]

    for question, answer in generate_rag_answers(questions):
        print(f"\nQ: {question}\nA: {answer}\n" + "-" * 60)


if __name__ == "__main__":
    main()
