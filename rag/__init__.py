"""Text-based RAG retrieval package for the HKU InnoWings challenge.

Public entry points:
    from rag import generate_rag_answers, rag_answer
"""

from .answer import generate_rag_answers, rag_answer

__all__ = ["generate_rag_answers", "rag_answer"]
