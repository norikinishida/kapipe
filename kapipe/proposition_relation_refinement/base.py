from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BasePropositionRelationRefiner(ABC):
    """Base class for Proposition Relation Refinement components."""

    @abstractmethod
    def refine(
        self,
        triple: dict[str, Any],
    ) -> dict[str, Any]:
        """Verify and refine a proposition relation triple."""
