import json, random

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

chunks = []
with open("hku_innowings_chunks.jsonl", "r", encoding="utf-8") as f:
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

print(f"Total raw chunks: {len(chunks)}")

blacklisted = [c for c in chunks if is_blacklisted(c["text_content"])]
kept = [c for c in chunks if not is_blacklisted(c["text_content"])]
print(f"Blacklisted dropped: {len(blacklisted)}")
print(f"After blacklist: {len(kept)}")

seen = set(); dupes = []; unique = []
for c in kept:
    t = c["text_content"]
    if t in seen:
        dupes.append(c)
    else:
        seen.add(t)
        unique.append(c)
print(f"Duplicates dropped: {len(dupes)}")
print(f"After dedup: {len(unique)}")

final_count = sum(len(split_text(c["text_content"])) for c in unique)
split_count = sum(1 for c in unique if len(split_text(c["text_content"])) > 1)
print(f"Chunks that get split: {split_count}")
print(f"Final chunk count: {final_count}")

random.seed(42)
print("\n--- BLACKLIST SAMPLE (5 being dropped) ---")
for c in random.sample(blacklisted, min(5, len(blacklisted))):
    print(" ", repr(c["text_content"][:120].encode("ascii", "replace").decode()))

print("\n--- DEDUP SAMPLE (5 being dropped as duplicates) ---")
for c in random.sample(dupes, min(5, len(dupes))):
    print(" ", repr(c["text_content"][:120].encode("ascii", "replace").decode()))

print("\n--- SPLIT SAMPLE (2 chunks being split) ---")
split_examples = [c for c in unique if len(split_text(c["text_content"])) > 1]
for c in random.sample(split_examples, min(2, len(split_examples))):
    subs = split_text(c["text_content"])
    print(f"  Original ({len(c['text_content'])} chars) -> {len(subs)} pieces")
    for i, s in enumerate(subs):
        print(f"    [{i+1}] {s[:120].encode('ascii','replace').decode()}")
    print()
