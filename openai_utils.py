# openai_utils.py

import os
import requests
from dotenv import load_dotenv

# Load your OpenAI API key from .env
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# Chat completions endpoint and headers
ENDPOINT = "https://api.openai.com/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def chat_completion(messages, model="gpt-4", temperature=0.3, max_tokens=512):
    """
    Send a chat-completion request and return the assistant’s reply.
    - messages: list of {"role": "...", "content": "..."} dicts
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    resp = requests.post(ENDPOINT, headers=HEADERS, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]
