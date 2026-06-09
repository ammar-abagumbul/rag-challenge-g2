import json
from typing import Tuple, Set, Dict


def process_multiple_json_arrays(file_path) -> Tuple[Set[str], Dict[str, Dict[str, str]]]:
    parent_urls = set()
    image_metadata = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    decoder = json.JSONDecoder()
    index = 0
    content_length = len(content)

    # Fast-forward past any initial whitespace
    while index < content_length and content[index].isspace():
        index += 1

    # Loop through the file and decode each JSON array structure found
    while index < content_length:
        try:
            chunks, idx = decoder.raw_decode(content, index)

            if isinstance(chunks, list):
                for chunk in chunks:
                    if not isinstance(chunk, dict):
                        continue

                    # 1. Extract unique parent URLs
                    parent_url = chunk.get("parent_url")
                    if parent_url:
                        parent_urls.add(parent_url)

                    # 2. Extract image metadata (captions)
                    images = chunk.get("associated_images") or []
                    for img in images:
                        img_url = img.get("image_url")
                        if img_url:
                            if img_url not in image_metadata:
                                image_metadata[img_url] = {
                                    "caption": img.get("vlm_caption", ""),
                                }

            # Move our pointer to where the last JSON array ended
            index = idx

            # Skip any whitespace/newlines between arrays
            while index < content_length and content[index].isspace():
                index += 1

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON stream at position {index}: {e}")
            break


    return parent_urls, image_metadata


# print the number of unique parent URLs and image metadata
parent_urls, image_metadata = process_multiple_json_arrays("hku_innowings_chunks.jsonl")
print(f"Unique parent URLs: {len(parent_urls)}")
print(f"Unique image metadata: {len(image_metadata)}")
