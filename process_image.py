import os 
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path
from typing import List,Dict
import chromadb
from dotenv import load_dotenv
from openai import AzureOpenAI
import json
from tqdm import tqdm
load_dotenv()

BASE_DIR=Path.cwd()
CHROMA_PATH=BASE_DIR/'chroma_db'

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_VISION_MODEL=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")
azure_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2025-01-01-preview",
)
chroma_client=chromadb.PersistentClient(path=CHROMA_PATH)
collection=chroma_client.get_or_create_collection(name="image_embeddings") # TODO: change name to the shared name
processed_images=[]
# clip_model=CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# clip_processor=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
def get_azure_caption(image_path: str) -> str:
    try:
        import base64
        
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        response = azure_client.chat.completions.create(
            model=AZURE_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe the content of this image, by identifying the objects, location and the overall scene. return every detail you can find in the image, like number of objects, people. When the image contains a caption, return it. format your response cleanly in json with description and caption separated."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
        )
        caption = response.choices[0].message.content
        return caption
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return ""
if __name__=="__main__":
    image_path=BASE_DIR/'images'
    image_paths = list(image_path.iterdir())
    # Use tqdm to wrap the iterator
    processed_images = []
    for img in tqdm(image_paths, desc="Processing images", unit="image"):
        caption = json.loads(get_azure_caption(img))
        processed_images.append({
            "image_name": str(img.name),
            "description": str(caption["description"]),
            "caption": str(caption["caption"])
        })
    with open(BASE_DIR/'processed_images.json', 'w') as f:
        json.dump(processed_images, f, indent=4)