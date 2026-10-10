"""Tests for Groq LLM provider."""

import pytest
from config import Settings
from services.llm import get_llm_provider
from services.llm.groq import GroqProvider


def test_groq_provider_missing_api_key():
    """Verify GroqProvider gracefully informs user if GROQ_API_KEY is not set."""
    provider = GroqProvider(api_key="", model="llama-3.3-70b-versatile")
    assert provider.health() is False

    response = provider.generate("Hello world")
    assert "GROQ_API_KEY is not set" in response


def test_groq_provider_factory_resolution():
    """Verify get_llm_provider instantiates GroqProvider when configured."""
    settings = Settings(
        LLM_PROVIDER="groq",
        GROQ_API_KEY="gsk_sample_test_key_12345",
        GROQ_MODEL="llama-3.3-70b-versatile",
    )
    provider = get_llm_provider(settings)
    assert isinstance(provider, GroqProvider)
    assert provider.model == "llama-3.3-70b-versatile"
    assert provider.api_key == "gsk_sample_test_key_12345"
