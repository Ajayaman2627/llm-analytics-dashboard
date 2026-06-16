from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

token = os.getenv("GITHUB_TOKEN")
if not token or token == "your_token_here":
    print("ERROR: Add your GitHub token to .env first")
else:
    client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=token,
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello in 5 words."}],
        max_tokens=50,
    )
    print("GitHub Models works!")
    print("Response:", response.choices[0].message.content)
    print("Tokens used:", response.usage.prompt_tokens, "in /", response.usage.completion_tokens, "out")
