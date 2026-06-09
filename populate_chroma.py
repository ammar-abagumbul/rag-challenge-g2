import json
import uuid
from pathlib import Path
import chromadb

CHROMA_PATH = Path.cwd() / "chroma_db/chroma_db"
JSONL_FILES = ["hku_innowings_chunks.jsonl", "hku_innowings_chunks2.jsonl"]
COLLECTION_NAME = "hku_innowings_scraper"

BLACKLIST_PATTERNS = [
    "ast-container", "#colophon", "#content", ".entry-header",
    "Skip to content", "Recent Posts", "Recent Comments",
    "Archives", "Categories", "Previous image Next image",
]

def is_blacklisted(text):
    return any(p in text for p in BLACKLIST_PATTERNS)

def split_text(text, max_len=800, chunk_size=600, overlap=100):
    if len(text) <= max_len:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + overlap, end)
            if space > start:
                end = space
        chunks.append(text[start:end].strip())
        next_start = end - overlap
        if next_start <= start:
            next_start = start + chunk_size
        start = next_start
    return chunks

def load_jsonl(path):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    decoder = json.JSONDecoder()
    index = 0
    while index < len(content):
        while index < len(content) and content[index].isspace():
            index += 1
        if index >= len(content):
            break
        try:
            arr, idx = decoder.raw_decode(content, index)
            for c in arr:
                chunks.append(c)
            index = idx
        except:
            break
    return chunks


if __name__ == "__main__":
    print("[=== ChromaDB Population Script ===]")

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' current count: {collection.count()}")

    # Load all chunks from both JSONL files
    all_raw = []
    for fname in JSONL_FILES:
        loaded = load_jsonl(fname)
        print(f"Loaded {len(loaded)} chunks from {fname}")
        all_raw.extend(loaded)
    print(f"Total raw chunks: {len(all_raw)}")

    # Pre-load existing texts to avoid re-inserting
    existing = collection.get()
    seen_texts = set(existing["documents"]) if existing["documents"] else set()

    blacklisted = 0
    dupes = 0
    added = 0
    split_count = 0

    for chunk in all_raw:
        text = chunk.get("text_content", "").strip()
        if not text:
            continue

        # Blacklist filter
        if is_blacklisted(text):
            blacklisted += 1
            continue

        # Dedup filter
        if text in seen_texts:
            dupes += 1
            continue
        seen_texts.add(text)

        # Split large chunks
        sub_texts = split_text(text)
        if len(sub_texts) > 1:
            split_count += 1

        for sub_text in sub_texts:
            chunk_id = f"hku_innowings_chunk_{uuid.uuid4()}"
            collection.add(
                documents=[sub_text],
                metadatas=[{
                    "parent_url": chunk.get("parent_url", ""),
                    "images_json": str(chunk.get("associated_images", [])),
                    "urls_json": str(chunk.get("url_mappings", {})),
                }],
                ids=[chunk_id],
            )
            added += 1

        if added % 100 == 0 and added > 0:
            print(f"  ... {added} chunks added so far")

    print()
    print(f"=== Done ===")
    print(f"Blacklisted dropped:  {blacklisted}")
    print(f"Duplicates dropped:   {dupes}")
    print(f"Chunks split:         {split_count}")
    print(f"Total added:          {added}")
    print(f"Final ChromaDB count: {collection.count()}")
