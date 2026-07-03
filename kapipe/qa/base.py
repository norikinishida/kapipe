from __future__ import annotations

from abc import ABC, abstractmethod

from ..datatypes import ContextsForOneExample, Question


class BaseQA(ABC):
    """Base class for question answering components."""

    @abstractmethod
    def answer(
        self,
        question: Question,
        contexts_for_question: ContextsForOneExample | None = None
    ) -> Question:
        """Answer a single question."""