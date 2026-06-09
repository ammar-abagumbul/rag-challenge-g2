import requests
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

# -------------------------------------------------------------------------
# Setup & Configuration
# -------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

POSTER_PROMPTS = [
    "an event poster",
    "a university notice flyer",
    "an informational infographic",
    "an advertisement flyer with text"
]
NON_POSTER_PROMPTS = [
    "a scenic photograph",
    "a close-up photo of a person or object",
    "a stock photo",
    "a random web image"
]
ALL_PROMPTS = POSTER_PROMPTS + NON_POSTER_PROMPTS

# -------------------------------------------------------------------------
# Core Functions
# -------------------------------------------------------------------------

def load_image_from_url(url: str) -> Image.Image:
    """Downloads an image from a URL and returns a PIL Image."""
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        return Image.open(response.raw).convert("RGB")
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return None

def initialize_clip(model_name: str = "openai/clip-vit-base-patch32"):
    """Initializes and returns the CLIP model and processor."""
    print(f"Loading {model_name} onto {device}...")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor

def predict_image_distributions(image_urls: list, prompts: list, model, processor):
    """
    Takes a list of image URLs and a list of text prompts,
    and outputs the probability distribution over those prompts for each image.
    """
    results = {}

    for url in image_urls:
        image = load_image_from_url(url)
        if image is None:
            continue  # Skip if the image failed to download

        # Process inputs for the model
        inputs = processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

            # logits_per_image is the similarity score between the image and all text prompts
            logits_per_image = outputs.logits_per_image

            # Convert logits to a software probability distribution via Softmax
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        # Map prompts to their respective probabilities
        prompt_probs = {prompt: float(prob) for prompt, prob in zip(prompts, probs)}
        results[url] = prompt_probs

    return results

# -------------------------------------------------------------------------
# Execution Block
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize model and processor
    model, processor = initialize_clip()

    # Test URLs (Replace these with your actual image links)
    sample_urls = [
    # "https://innoacademy.engg.hku.hk/wp-content/uploads/2026/01/Pitch-2026-1024x724.png",
    "https://innoacademy.engg.hku.hk/wp-content/uploads/2021/11/2022-Pitching-a3-landscape-1-1024x724.jpeg",
    # "https://innoacademy.engg.hku.hk/wp-content/uploads/2026/04/24e4e318-1c77-48ad-9c37-affb39cc1c74-768x432.jpg",
    # "https://innoacademy.engg.hku.hk/wp-content/uploads/2025/10/2025_RAICOM-768x576.jpg",
    # "https://innoacademy.engg.hku.hk/wp-content/uploads/2025/06/InnoGrow_HKAE-768x511.jpg",
    # "https://innoacademy.engg.hku.hk/wp-content/uploads/2025/02/Student-Development-Projects-1024x579.png",
    # "https://innoacademy.engg.hku.hk/wp-content/uploads/2026/02/MechanicalFlowerPosterWithQRCode-2-768x1138.jpg",
    # "https://innoacademy.engg.hku.hk/wp-content/uploads/2026/02/Happy-Chinese-New-Year-with-Robots-Arms-1024x724.png",
    # "https://innoacademy.engg.hku.hk/wp-content/uploads/2025/10/Final-poster-inw-workshop-2025-oct-1024x576.png",
    ]

    print("\nRunning inference...")
    distributions = predict_image_distributions(sample_urls, ALL_PROMPTS, model, processor)

    # Print the results beautifully
    for url, probs in distributions.items():
            print(f"\n{'='*60}\nURL: {url}\n{'='*60}")

            print("Poster Classes (Positive):")
            for prompt in POSTER_PROMPTS:
                print(f"  - {prompt}: {probs[prompt]:.4f}")

            print("\nNon-Poster Classes (Negative):")
            for prompt in NON_POSTER_PROMPTS:
                print(f"  - {prompt}: {probs[prompt]:.4f}")

            # Optional: Print total aggregate probability for Poster vs Non-Poster
            poster_score = sum(probs[p] for p in POSTER_PROMPTS)
            non_poster_score = sum(probs[p] for p in NON_POSTER_PROMPTS)
            print(f"\nAggregate Classification:")
            print(f"  --> Total Poster Probability: {poster_score:.4f}")
            print(f"  --> Total Non-Poster Probability: {non_poster_score:.4f}")
