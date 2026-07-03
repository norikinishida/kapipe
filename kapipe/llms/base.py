from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Base class for large language model clients."""

    provider: str

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0
    ) -> str:
        """Generate text for the given prompt."""