import os

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# 1. SETUP: Replace with your actual Azure parameters
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "YOUR_ACTUAL_KEY_HERE")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://YOUR_://azure.com")
# Set this to your actual deployment name from Azure Studio
REAL_DEPLOYMENT_NAME = "some jibberish"

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2025-01-01-preview",
)

# Public image for vision test
TEST_IMAGE_URL = "https://innowings.engg.hku.hk/wp-content/uploads/2021/08/WhatsApp-Image-2022-11-21-at-11.19.32-AM.jpg"


# 2. TEST FUNCTION
def execute_test(deployment_name: str, label: str):
    print(f"\n=== Running Test: {label} (Deployment: '{deployment_name}') ===")
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in 3 words."},
                        {"type": "image_url", "image_url": {"url": TEST_IMAGE_URL}},
                    ],
                }
            ],
            max_tokens=20,
        )
        print(" [SUCCESS] API Call completed!")
        print(f" [RESPONSE] {response.choices[0].message.content.strip()}")
    except Exception as e:
        print(f" [FAILED] Error: {e}")


# 3. EXECUTION
if __name__ == "__main__":
    # Test 1: Triggers the routing bug with invalid deployment string
    execute_test(deployment_name="gpt-5-mini", label="PROVING THE BUG")

    # Test 2: Verifies fix with valid deployment
    if REAL_DEPLOYMENT_NAME != "YOUR_REAL_DEPLOYMENT_NAME_HERE":
        execute_test(deployment_name=REAL_DEPLOYMENT_NAME, label="VERIFYING FIX")
