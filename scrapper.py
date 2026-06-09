import json
import os
import re
import time
import urllib.parse
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Set

import chromadb
import requests
import torch
from bs4 import BeautifulSoup
from bs4.element import NavigableString
from dotenv import load_dotenv
from openai import AzureOpenAI
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from collections import deque

from preprocess_jsonl import process_multiple_json_arrays

load_dotenv()

SEED_URLS = ["https://innowings.engg.hku.hk/", "https://innoacademy.engg.hku.hk/"]
# SEED_URLS = ["https://innoacademy.engg.hku.hk/pitching/"]
ALLOWED_DOMAINS = {"innowings.engg.hku.hk", "innoacademy.engg.hku.hk"}
MAX_DEPTH = 5
DELAY_SECONDS = 0.5
CLIP_THRESHOLD = 0.90

BASE_DIR = Path.cwd()
CHROMA_PATH = BASE_DIR / "chroma_db/chroma_db"

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")

assert AZURE_OPENAI_API_KEY is not None, "Key should not be none"
assert AZURE_OPENAI_ENDPOINT is not None, "Endpoint should not be none"

azure_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2025-01-01-preview",
)
AZURE_VISION_MODEL = "gpt-5-mini"

PREPROCESSED_JSONL = "hku_innowings_chunks.jsonl"
OUTPUT_JSONL = "hku_innowings_chunks2.jsonl"

BLACKLIST_PATTERNS = [
    "ast-container", "#colophon", "#content", ".entry-header",
    "Skip to content", "Recent Posts", "Recent Comments",
    "Archives", "Categories", "Previous image Next image",
]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="hku_innowings_scraper")

# Pre-load all existing document texts to skip exact duplicates on re-runs
_existing = collection.get()
seen_texts: Set[str] = set(_existing["documents"]) if _existing["documents"] else set()


def normalize_url(url: str) -> str:
    """Strip trailing slash from path so a/b and a/b/ are treated as the same page."""
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return parsed._replace(path=path, fragment="").geturl()


_raw_visited, transcribed_posters = process_multiple_json_arrays(PREPROCESSED_JSONL)
# Normalize loaded parent URLs so trailing-slash variants count as already visited
visited_urls: Set[str] = {normalize_url(u) for u in _raw_visited}


clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

POSTER_PROMPTS = [
    "an event poster",
    "a university notice flyer",
    "an informational infographic",
    "an advertisement flyer with text",
]
NON_POSTER_PROMPTS = [
    "a scenic photograph",
    "a close-up photo of a person or object",
    "a stock photo",
    "a random web image",
]
ALL_PROMPTS = POSTER_PROMPTS + NON_POSTER_PROMPTS

IMAGE_EXTENSIONS = {
    'jpg', 'jpeg', 'png', 'gif', 'webp', 'svg',
    'bmp', 'ico', 'tiff', 'heic', 'avif'
}


def is_blacklisted(text: str) -> bool:
    return any(pattern in text for pattern in BLACKLIST_PATTERNS)


def split_text(text: str, max_len: int = 800, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    """Return text as-is if short enough; otherwise split at word boundaries with overlap."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            space = text.rfind(' ', start + overlap, end)
            if space > start:
                end = space
        chunks.append(text[start:end].strip())
        next_start = end - overlap
        if next_start <= start:
            next_start = start + chunk_size
        start = next_start
    return chunks


def analyze_image_with_clip(img_url: str) -> bool:
    try:
        if img_url in transcribed_posters:
            return True

        response = requests.get(
            img_url, timeout=5, headers={"User-Agent": "Mozilla/5.0"}
        )
        if response.status_code != 200:
            return False

        image = Image.open(BytesIO(response.content)).convert("RGB")

        inputs = clip_processor(
            text=ALL_PROMPTS, images=image, return_tensors="pt", padding=True
        ).to(device)

        with torch.no_grad():
            dev_type = device.type if isinstance(device, torch.device) else str(device)
            with torch.amp.autocast(
                device_type=dev_type, enabled=(dev_type == "cuda"), dtype=torch.float16
            ):
                outputs = clip_model(**inputs)

                logits_per_image = outputs.logits_per_image

                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        poster_prob_sum = float(sum(probs[: len(POSTER_PROMPTS)]))
        return poster_prob_sum >= CLIP_THRESHOLD

    except Exception as e:
        print(f"[!] CLIP analysis failed for {img_url}: {e}")
        return False


def get_azure_vision_caption(img_url: str) -> str:
    try:
        if img_url in transcribed_posters:
            return transcribed_posters[img_url]["caption"]

        response = azure_client.chat.completions.create(
            model=AZURE_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in just the necessary detail. Extract critical information like dates, locations, costs, sponsors, organizers, and event names. Format your response cleanly.",
                        },
                        {"type": "image_url", "image_url": {"url": img_url}},
                    ],
                }
            ],
        )

        caption = response.choices[0].message.content.strip()
        transcribed_posters[img_url] = caption

        return caption
    except Exception as e:
        print(f"[!] Azure OpenAI API call failed: {e}")
        return "[Error: Poster transcription unavailable]"


def clean_and_flatten_dom(soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
    """Parses the HTML DOM as a linear sequence of elements to maintain reading

    order. Extracts text sections, handles anchor tags dynamically, and isolates
    high-value images.
    """
    for noise in soup(["script", "style", "nav", "footer"]):
        noise.decompose()

    elements_stream = []
    body = soup.body if soup.body else soup

    current_section = {
        "type": "section",
        "content_pieces": [],
        "images": [],
        "url_mappings": {},
    }

    def walk_tree(element):
        if isinstance(element, NavigableString):
            text = element.strip()
            if text:
                current_section["content_pieces"].append(text)
            return

        if element.name == "a":
            href = element.get("href")
            if element.find("img"):
                pass
            elif href:
                visible_text = element.get_text(strip=True)
                if visible_text:
                    absolute_url = urllib.parse.urljoin(base_url, href)
                    placeholder = f"[Link: {visible_text}]"
                    current_section["content_pieces"].append(placeholder)
                    current_section["url_mappings"][placeholder] = absolute_url

        elif element.name == "img":
            src = element.get("src")
            if src:
                abs_img_url = urllib.parse.urljoin(base_url, src)
                alt_text = element.get("alt", "No descriptive alt text provided")

                if analyze_image_with_clip(abs_img_url):
                    print(f"[+] Image passed Poster Guard: {abs_img_url}")
                    vlm_caption = get_azure_vision_caption(abs_img_url)

                    image_obj = {
                        "image_url": abs_img_url,
                        "alt_text": alt_text,
                        "vlm_caption": vlm_caption,
                    }

                    # Determine 'n' based on the upcoming array index position
                    img_index = len(current_section["images"])
                    current_section["images"].append(image_obj)

                    # Check if this image is wrapped by a link
                    parent_a = element.find_parent("a")
                    if parent_a and parent_a.get("href"):
                        link_url = urllib.parse.urljoin(base_url, parent_a.get("href"))
                        mapping_key = f"img_{img_index}"
                        current_section["url_mappings"][mapping_key] = link_url

                    caption_placeholder = f" [Poster Transcription: {vlm_caption}]"
                    current_section["content_pieces"].append(caption_placeholder)
            return

        if element.name in ["section", "article", "main"] or (
            element.name and re.match(r"^h[1-6]$", element.name)
        ):
            if current_section["content_pieces"] or current_section["images"]:
                elements_stream.append(dict(current_section))

            current_section["content_pieces"] = []
            current_section["images"] = []
            current_section["url_mappings"] = {}

        for child in element.children:
            walk_tree(child)

    walk_tree(body)

    if current_section["content_pieces"] or current_section["images"]:
        elements_stream.append(dict(current_section))

    return elements_stream


def is_image_url(url: str) -> bool:
    _, ext = os.path.splitext(url)
    clean_ext = ext.lstrip('.').lower()
    return clean_ext in IMAGE_EXTENSIONS


def crawl_and_process(url: str, depth: int = 1):
    """Recursive core crawler restricted to bounds and target depth limits."""
    if depth > MAX_DEPTH or url in visited_urls:
        return

    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.netloc not in ALLOWED_DOMAINS:
        return

    if is_image_url(url):
        return

    print(f"[*] Crawling Depth {depth}: {url}")
    visited_urls.add(url)

    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            return

        soup = BeautifulSoup(response.content, "html.parser")

        sections = clean_and_flatten_dom(soup, url)

        all_chunks_data = []

        for sec in sections:
            combined_text = " ".join(sec["content_pieces"]).strip()
            if not combined_text and not sec["images"]:
                continue

            chunk_id = f"hku_innowings_chunk_{uuid.uuid4()}"

            # Map elements into target storage schema
            chunk_data = {
                "chunk_id": chunk_id,
                "parent_url": url,
                "text_content": combined_text,
                "associated_images": sec["images"],
                "url_mappings": sec["url_mappings"],
            }

            all_chunks_data.append(chunk_data)

            # Persist directly to Vector Store
            collection.add(
                documents=[chunk_data["text_content"]],
                metadatas=[
                    {
                        "parent_url": chunk_data["parent_url"],
                        "images_json": str(chunk_data["associated_images"]),
                        "urls_json": str(chunk_data["url_mappings"]),
                    }
                ],
                ids=[chunk_data["chunk_id"]],
            )

        # Extract child links for Next Depth Exploration
        for anchor in soup.find_all("a", href=True):
            next_url = urllib.parse.urljoin(url, anchor["href"])
            next_url = urllib.parse.urlsplit(next_url)._replace(fragment="").geturl()
            time.sleep(DELAY_SECONDS)
            crawl_and_process(next_url, depth + 1)

        with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
            json.dump(all_chunks_data, f, indent=4, ensure_ascii=False)
            f.write("\n")

    except Exception as e:
        print(f"[!] System processing failure at URL {url}: {e}")


def crawl_and_process_bfs(start_url: str):
    """Iterative BFS core crawler using a queue to process levels sequentially."""

    queue = deque([(normalize_url(start_url), 1)])

    while queue:
        url, depth = queue.popleft()
        url = normalize_url(url)

        if depth > MAX_DEPTH or url in visited_urls:
            continue

        parsed_url = urllib.parse.urlparse(url)
        if parsed_url.netloc not in ALLOWED_DOMAINS:
            continue

        if is_image_url(url):
            continue

        print(f"[*] Crawling Depth {depth}: {url}")
        visited_urls.add(url)

        try:
            response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if response.status_code != 200:
                continue

            soup = BeautifulSoup(response.content, "html.parser")
            sections = clean_and_flatten_dom(soup, url)
            all_chunks_data = []

            for sec in sections:
                combined_text = " ".join(sec["content_pieces"]).strip()
                if not combined_text and not sec["images"]:
                    continue

                # Blacklist filter — drop WordPress scaffolding and nav noise
                if is_blacklisted(combined_text):
                    continue

                # Dedup filter — skip if identical text already in ChromaDB
                if combined_text in seen_texts:
                    continue
                seen_texts.add(combined_text)

                # Split large chunks into overlapping sub-chunks
                sub_texts = split_text(combined_text)

                for sub_text in sub_texts:
                    chunk_id = f"hku_innowings_chunk_{uuid.uuid4()}"
                    chunk_data = {
                        "chunk_id": chunk_id,
                        "parent_url": url,
                        "text_content": sub_text,
                        "associated_images": sec["images"],
                        "url_mappings": sec["url_mappings"],
                    }
                    all_chunks_data.append(chunk_data)

                    collection.add(
                        documents=[sub_text],
                        metadatas=[{
                            "parent_url": url,
                            "images_json": str(sec["images"]),
                            "urls_json": str(sec["url_mappings"]),
                        }],
                        ids=[chunk_id],
                    )

            with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
                json.dump(all_chunks_data, f, indent=4, ensure_ascii=False)
                f.write("\n")

            # Extract child links — normalize before queuing
            for anchor in soup.find_all("a", href=True):
                next_url = urllib.parse.urljoin(url, anchor["href"])
                next_url = normalize_url(next_url)

                if next_url not in visited_urls:
                    queue.append((next_url, depth + 1))

            # Politeness delay between requests
            time.sleep(DELAY_SECONDS)

        except Exception as e:
            print(f"[!] System processing failure at URL {url}: {e}")


if __name__ == "__main__":
    print("[=== Executing Tam Innovation Wing Scraper Init Pipeline ===]")
    for seed in SEED_URLS:
        crawl_and_process_bfs(seed)
    print(f"\n[=== Scrape Complete! Total Chunks Cached: {collection.count()} ===]")
