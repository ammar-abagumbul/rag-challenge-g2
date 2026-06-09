import os

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")

azure_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2025-01-01-preview",
)

AZURE_VISION_MODEL = "gpt-5-mini"


def get_azure_vision_caption(img_url: str) -> str:
    try:
        response = azure_client.chat.completions.create(
            model=AZURE_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in detail. Extract critical information like dates, locations, costs, sponsors, organizers, and event names. Format your response cleanly.",
                        },
                        {"type": "image_url", "image_url": {"url": img_url}},
                    ],
                }
            ],
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[!] Azure OpenAI API call failed: {e}")
        return "[Error: Poster transcription unavailable]"


if __name__ == "__main__":
    url = "https://innoacademy.engg.hku.hk/wp-content/uploads/2025/01/Pitching-Poster-V1-1024x724.jpg"
    print(get_azure_vision_caption(url))