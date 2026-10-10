"""LLM module factory and exports."""

from config import Settings
from services.llm.base import LLMProvider
from services.llm.groq import GroqProvider
from services.llm.ollama import OllamaProvider


def get_llm_provider(settings: Settings) -> LLMProvider:
    """Return configured LLM provider instance."""
    provider_type = settings.LLM_PROVIDER.lower()
    if provider_type == "groq":
        return GroqProvider(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            base_url=settings.GROQ_BASE_URL,
            timeout=settings.LLM_TIMEOUT,
        )
    elif provider_type == "ollama":
        return OllamaProvider(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            timeout=settings.LLM_TIMEOUT,
        )
    raise ValueError(f"Unsupported LLM provider: {provider_type}")


__all__ = ["LLMProvider", "OllamaProvider", "GroqProvider", "get_llm_provider"]
