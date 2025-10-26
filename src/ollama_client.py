# src/ollama_client.py
import ollama
from tenacity import retry, stop_after_attempt, wait_exponential


class OllamaClient:
    def __init__(self, model: str, temperature: float, top_p: float, max_tokens: int, stop=None):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop = stop or []

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=8))
    def chat_once(self, prompt: str) -> str:
        resp = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful, step-by-step reasoner."},
                {"role": "user", "content": prompt}
            ],
            options={
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.max_tokens,
                "stop": self.stop,
            }
        )
        return resp["message"]["content"]