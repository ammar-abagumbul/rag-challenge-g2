"""Ingest local image captions into the retrievable ChromaDB collection.

The colleague's `process_image.py` captions the photos in `images/` and writes
`processed_images.json`, but never indexes them — they ended up in a separate,
empty `image_embeddings` collection in the wrong DB path. This script loads
that JSON and upserts each caption as a text-typed chunk into the SAME
collection the RAG pipeline already searches (`hku_innowings_scraper` at
chroma_db/chroma_db). Once indexed, the existing semantic + keyword + rerank
flow surfaces them with no special-casing.

Idempotent: re-running upserts by a deterministic id, so it won't duplicate.

Usage:
    python ingest_images.py
"""

import json
import re
from pathlib import Path

import chromadb

from rag import config

IMAGES_JSON = Path(config.PROJECT_ROOT) / "processed_images.json"


def _slug(name: str) -> str:
    """Stable id fragment from an image filename."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def _build_document(record: dict) -> str:
    """Self-describing text so the chunk is meaningful in retrieval AND answers."""
    name = record.get("image_name", "")
    description = (record.get("description") or "").strip()
    caption = (record.get("caption") or "").strip()
    parts = [f"[Local image: {name}]", description]
    if caption:
        parts.append(f"Caption: {caption}")
    return "\n\n".join(p for p in parts if p)


def main() -> None:
    if not IMAGES_JSON.exists():
        raise FileNotFoundError(
            f"{IMAGES_JSON} not found. Run process_image.py first to generate it."
        )

    records = json.loads(IMAGES_JSON.read_text(encoding="utf-8"))
    print(f"Loaded {len(records)} image captions from {IMAGES_JSON.name}")

    client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
    collection = client.get_or_create_collection(name=config.COLLECTION_NAME)
    before = collection.count()

    ids, documents, metadatas = [], [], []
    seen_ids: dict = {}
    skipped = 0
    for rec in records:
        name = rec.get("image_name", "")
        doc = _build_document(rec)
        if not name or not doc.strip():
            skipped += 1
            continue
        base_id = f"local_image_{_slug(name)}"
        # Distinct filenames can slug to the same id (e.g. "Open event"/"Open Event ");
        # disambiguate so the upsert stays unique and stable.
        n = seen_ids.get(base_id, 0)
        seen_ids[base_id] = n + 1
        ids.append(base_id if n == 0 else f"{base_id}_{n}")
        documents.append(doc)
        metadatas.append({
            "source_type": "local_image",
            "image_name": name,
            "parent_url": "",          # local photo, no source URL
            "images_json": "[]",
            "urls_json": "{}",
        })

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    after = collection.count()
    print(f"Upserted {len(ids)} image chunks (skipped {skipped} empty).")
    print(f"Collection '{config.COLLECTION_NAME}' count: {before} -> {after}")


if __name__ == "__main__":
    main()
