"""Azure OpenAI helpers used by the retrieval pipeline.

Three of the four functions implement the "advanced RAG" requirements:
  * decompose_query   -> query decomposition / rewriting  (requirement 2)
  * generate_keywords -> agent-generated keyword queries   (requirement 3)
  * rerank            -> LLM relevance reranking            (requirement 1)
And generate_answer turns the retrieved chunks into the final response.

Every LLM call is defensive: if the model returns malformed JSON or the API
fails, we fall back to a sensible default so retrieval never hard-crashes.
"""

import json
import re
from functools import lru_cache
from typing import List, Sequence

from openai import OpenAI

from . import config


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    """Lazily build a single shared client (created on first LLM use).

    Targets the InnoWings APIM gateway directly: base_url already includes the
    deployment, `api-version` is sent as a query param, and auth is the
    `api-key` header (the gateway rejects the standard Bearer / Ocp-Apim forms).
    A bounded per-call timeout + retry budget keeps the critical path
    predictable so the pipeline stays well within its latency target.
    """
    key = config.require_api_key()
    return OpenAI(
        base_url=f"{config.CHAT_ENDPOINT}/deployments/{config.CHAT_MODEL}",
        api_key=key,
        default_query={"api-version": config.AZURE_OPENAI_API_VERSION},
        default_headers={"api-key": key},
        timeout=config.ANSWER_TIMEOUT,        # client-level safety default
        max_retries=config.LLM_MAX_RETRIES,
    )


def _chat(
    messages: list,
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: float = config.LLM_TIMEOUT,
) -> str:
    """Single chat completion call, returning the assistant text.

    `timeout` is enforced per request so each pipeline stage has its own bound;
    on expiry the SDK raises and the caller degrades gracefully.
    """
    resp = _client().chat.completions.create(
        model=config.CHAT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return (resp.choices[0].message.content or "").strip()


def _extract_json(text: str):
    """Pull the first JSON array/object out of a model response.

    Models sometimes wrap JSON in prose or ```json fences; this is forgiving.
    Returns the parsed value, or None if nothing parseable is found.
    """
    if not text:
        return None
    # Strip code fences if present.
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Last resort: grab the outermost [...] or {...} span.
    match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return None


# --------------------------------------------------------------------------
# Requirement 2: query decomposition / rewriting
# --------------------------------------------------------------------------
def decompose_query(question: str) -> List[str]:
    """Break a complex question into focused sub-queries.

    Simple questions come back as a single (lightly rewritten) query, so the
    caller can always treat the result as a list. The original question is
    always retained as well to avoid losing the user's exact phrasing.
    """
    system = (
        "You rewrite a user's question into search queries for a vector "
        "database about HKU InnoWings / InnoAcademy (student innovation programs, "
        "SIGs, tech talks, workshops, events). "
        "If the question is simple, return one cleaned-up query. "
        "If it is complex or covers multiple things, split it into 2-"
        f"{config.MAX_SUBQUERIES} specific, self-contained sub-queries. "
        "Resolve pronouns and vague references into explicit terms. "
        'Respond ONLY with a JSON array of strings, e.g. ["...", "..."].'
    )
    try:
        raw = _chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": question}],
            max_tokens=256,
        )
        parsed = _extract_json(raw)
    except Exception as exc:  # network / auth / quota -> degrade gracefully
        print(f"  [decompose] LLM call failed ({exc}); using original query.")
        parsed = None

    subqueries: List[str] = []
    if isinstance(parsed, list):
        subqueries = [str(q).strip() for q in parsed if str(q).strip()]

    # Always include the verbatim question and cap the count.
    if question not in subqueries:
        subqueries.insert(0, question)
    return subqueries[: config.MAX_SUBQUERIES] or [question]


# --------------------------------------------------------------------------
# Requirement 3: agent-generated keyword queries
# --------------------------------------------------------------------------
def generate_keywords(question: str) -> List[str]:
    """Generate compact keyword terms for full-text (FTS) retrieval.

    These target ChromaDB's `$contains` document filter, so they should be the
    distinctive words/phrases a relevant chunk would literally contain
    (program names, acronyms, event titles) rather than full sentences.
    """
    system = (
        "Extract the most distinctive keyword search terms for the question "
        "below, to be matched literally against documents about HKU InnoWings / "
        "InnoAcademy. Prefer proper nouns, acronyms, program/event names and "
        "domain terms. Each term should be 1-3 words. Avoid stop words and "
        f"generic verbs. Return at most {config.MAX_KEYWORDS} terms as a JSON "
        'array of strings, e.g. ["SIG", "tech talk", "robotics"].'
    )
    try:
        raw = _chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": question}],
            max_tokens=128,
        )
        parsed = _extract_json(raw)
    except Exception as exc:
        print(f"  [keywords] LLM call failed ({exc}); skipping keyword search.")
        parsed = None

    if not isinstance(parsed, list):
        return []
    seen, terms = set(), []
    for term in parsed:
        t = str(term).strip()
        key = t.lower()
        if t and key not in seen:
            seen.add(key)
            terms.append(t)
    return terms[: config.MAX_KEYWORDS]


# --------------------------------------------------------------------------
# Requirement 1: LLM relevance reranking
# --------------------------------------------------------------------------
def rerank(question: str, candidates: Sequence[str], top_n: int) -> List[int]:
    """Reorder candidate chunks by true relevance to `question`.

    `candidates` is a list of chunk texts. Returns a list of indices into that
    list, most-relevant first, length <= top_n. Falls back to the original
    order if the model response can't be parsed.
    """
    if not candidates:
        return []

    # Number each candidate; truncate long chunks to keep the prompt compact.
    listing = "\n".join(
        f"[{i}] {text[:500].strip()}" for i, text in enumerate(candidates)
    )
    system = (
        "You are a search result reranker. Given a question and numbered "
        "candidate passages, score how well each passage helps answer the "
        "question and return the best ones. Judge on actual relevance, not "
        "keyword overlap. Respond ONLY with a JSON array of the candidate "
        "numbers ordered from most to least relevant, e.g. [3, 0, 5]. "
        f"Include at most {top_n} numbers; drop clearly irrelevant passages."
    )
    user = f"Question: {question}\n\nCandidates:\n{listing}"

    try:
        raw = _chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            max_tokens=128,
        )
        parsed = _extract_json(raw)
    except Exception as exc:
        print(f"  [rerank] LLM call failed ({exc}); keeping retrieval order.")
        parsed = None

    order: List[int] = []
    if isinstance(parsed, list):
        for idx in parsed:
            try:
                i = int(idx)
            except (ValueError, TypeError):
                continue
            if 0 <= i < len(candidates) and i not in order:
                order.append(i)

    if not order:  # parse failure -> preserve incoming order
        order = list(range(len(candidates)))
    return order[:top_n]


# --------------------------------------------------------------------------
# Final answer generation
# --------------------------------------------------------------------------
def generate_answer(question: str, contexts: Sequence[str]) -> str:
    """Compose the final grounded answer from the reranked context chunks."""
    if not contexts:
        return "I could not find relevant information to answer that question."

    context_block = "\n\n---\n\n".join(
        f"[Source {i + 1}]\n{c}" for i, c in enumerate(contexts)
    )
    system = (
        "You are a helpful assistant answering questions about HKU InnoWings / "
        "InnoAcademy using only the provided sources. Be accurate. When the "
        "question asks you to list or enumerate items (e.g. 'what are the X', "
        "'list all Y'), include EVERY distinct item that appears across the "
        "sources, not just the first few -- scan all sources before answering. "
        "Otherwise be concise. If the sources do not contain the answer, say so "
        "plainly instead of guessing."
    )
    user = f"Question: {question}\n\nSources:\n{context_block}\n\nAnswer:"
    try:
        return _chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=600,
            timeout=config.ANSWER_TIMEOUT,
        )
    except Exception as exc:
        return f"[answer generation failed: {exc}]"
