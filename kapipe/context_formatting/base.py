from abc import ABC, abstractmethod
from typing import Any


class BaseContextFormatter(ABC):
    """Base class for Context Formatting components."""

    @abstractmethod
    def convert(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Convert input data into LLM-readable context."""