# src/generation.py
from typing import List
from src.ollama_client import OllamaClient


def generate_greedy(client: OllamaClient, prompt: str) -> str:
    orig_temp = client.temperature
    client.temperature = 0.0
    out = client.chat_once(prompt)
    client.temperature = orig_temp
    return out


def generate_k_paths(client: OllamaClient, prompt: str, K: int) -> List[str]:
    return [client.chat_once(prompt) for _ in range(K)]