"""Base LLM provider abstract interface."""

from abc import ABC, abstractmethod
from typing import Optional


class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: User/context prompt.
            system: Optional system instruction.
            temperature: Sampling temperature.

        Returns:
            Generated text string.
        """
        pass

    @abstractmethod
    def health(self) -> bool:
        """Check whether the LLM service is reachable and healthy.

        Returns:
            True if healthy, False otherwise.
        """
        pass
