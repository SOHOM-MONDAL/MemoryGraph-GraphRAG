"""Groq Cloud LLM provider implementation."""

import logging
from typing import Optional
import httpx

from services.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class GroqProvider(LLMProvider):
    """High-speed cloud LLM provider using Groq API (OpenAI-compatible)."""

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        base_url: str = "https://api.groq.com/openai/v1",
        timeout: int = 60,
    ) -> None:
        self.api_key = api_key.strip() if api_key else ""
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.5,
    ) -> str:
        """Call Groq chat completions API with system and prompt context."""
        if not self.api_key:
            return (
                "[Groq Error: GROQ_API_KEY is not set. "
                "Please configure GROQ_API_KEY in your environment variables or .env file.]"
            )

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2048,
        }

        timeout_config = httpx.Timeout(float(self.timeout), connect=5.0)
        try:
            with httpx.Client(timeout=timeout_config) as client:
                response = client.post(url, headers=headers, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "")
                    return ""
                else:
                    err_msg = response.text
                    logger.error("Groq API error (%d): %s", response.status_code, err_msg)
                    return f"[Groq API Error {response.status_code}: {err_msg}]"
        except httpx.TimeoutException:
            logger.warning("Groq API request timed out after %ds", self.timeout)
            return "[Groq Timeout: Request exceeded timeout limit.]"
        except Exception as e:
            logger.error("Unexpected error in Groq generation: %s", str(e))
            return f"[Groq Connection Error: {str(e)}]"

    def health(self) -> bool:
        """Verify API key validity against Groq models endpoint."""
        if not self.api_key:
            return False
        url = f"{self.base_url}/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            with httpx.Client(timeout=httpx.Timeout(3.0, connect=2.0)) as client:
                res = client.get(url, headers=headers)
                return res.status_code == 200
        except Exception:
            return False
