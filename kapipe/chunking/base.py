# kapipe/chunking/base.py

from __future__ import annotations

from abc import ABC, abstractmethod

from ..datatypes import Document, Passage


class BaseChunker(ABC):
    """Base class for chunking components."""

    @abstractmethod
    def split_passage_to_chunked_passages(
        self,
        passage: Passage,
        window_size: int,
    ) -> list[Passage]:
        """Split a passage into chunked passages."""

    @abstractmethod
    def convert_text_to_document(
        self,
        doc_key: str,
        text: str,
        title: str | None = None,
    ) -> Document:
        """Convert raw text into a document."""