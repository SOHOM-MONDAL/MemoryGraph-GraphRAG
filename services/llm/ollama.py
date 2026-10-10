"""Ollama LLM provider implementation."""

import logging
from typing import Optional
import httpx

from services.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """LLM provider connecting to local Ollama instance."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:1.5b",
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if system:
            payload["system"] = system

        timeout_config = httpx.Timeout(float(self.timeout), connect=2.0)
        try:
            with httpx.Client(timeout=timeout_config) as client:
                response = client.post(url, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "")
                else:
                    logger.warning(
                        "Ollama request failed with status %d: %s",
                        response.status_code,
                        response.text,
                    )
                    return f"[Ollama Error: Status {response.status_code}. Using fallback response]"
        except (httpx.ConnectError, httpx.ConnectTimeout):
            logger.warning("Could not connect to Ollama at %s", self.base_url)
            return (
                "[Ollama Offline: Ollama service is not reachable. "
                "Retrieved context is available in sources.]"
            )
        except httpx.TimeoutException:
            logger.warning("Ollama request timed out after %ds", self.timeout)
            return "[Ollama Timeout: LLM generation exceeded configured timeout.]"
        except Exception as e:
            logger.error("Unexpected error in Ollama generation: %s", str(e))
            return f"[LLM Error: {str(e)}]"

    def health(self) -> bool:
        url = f"{self.base_url}/api/tags"
        try:
            with httpx.Client(timeout=httpx.Timeout(2.0, connect=1.0)) as client:
                res = client.get(url)
                return res.status_code == 200
        except Exception:
            return False
