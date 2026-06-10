"""CLI runner for the text-based RAG retrieval pipeline.

Usage:
    python main.py                       # run a couple of demo questions
    python main.py "your question here"  # answer a single question
    python main.py --retrieval-only "q"  # show retrieved chunks, no LLM answer
"""

import builtins as _builtins
import sys

# The competition harness runs this file verbatim, including a bare
#   json.load(open('competition_questions.json'))
# and a final print(list(results)). On a Windows cp1252 locale that read can
# fail to decode a UTF-8 questions file, and the print can crash on non-ASCII
# model output (em-dashes / curly quotes from gpt-4o-mini). We have no control
# over that command, so we make the process UTF-8 safe at import time -- this
# runs even via `from main import generate_rag_answers`.

# 1) stdout/stderr -> UTF-8 so printing answers can't raise UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# 2) Default text-mode open() to UTF-8 so the harness's encoding-less
#    open('competition_questions.json') decodes a UTF-8 file correctly under a
#    cp1252 default locale. Binary mode (model weights, the sqlite DB) is left
#    untouched, and an explicit encoding= is always respected.
_real_open = _builtins.open


def _utf8_open(file, mode="r", buffering=-1, encoding=None, *args, **kwargs):
    if "b" not in mode and encoding is None:
        encoding = "utf-8"
    return _real_open(file, mode, buffering, encoding, *args, **kwargs)


_builtins.open = _utf8_open

from rag import generate_rag_answers, rag_answer
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
