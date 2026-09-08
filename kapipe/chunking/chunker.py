from __future__ import annotations

import spacy
from spacy.lang.en import English

from ..datatypes import Passage, Document
from .base import BaseChunker


class Chunker(BaseChunker):

    def __init__(self, model_name: str | None = None) -> None:

        if model_name is None:
            self.nlp = English()
            self.nlp.add_pipe("sentencizer")
        else:
            self.nlp = spacy.load(model_name, disable=["ner", "textcat"])

    ###########
    # Mapping from text to tokens, sentences, or chunks
    ###########

    def split_text_to_tokens(self, text: str) -> list[str]:
        """Split the text into tokens."""

        doc = self.nlp(text)
        return [tok.text for tok in doc]

    def split_text_to_sentences(self, text: str) -> list[str]:
        """Split the text into sentences."""

        if len(text) < self.nlp.max_length:
            doc = self.nlp(text)
            return [s.text for s in doc.sents if s.text.strip()]
        return self._split_text_long(text=text)

    def _split_text_long(self, text: str) -> list[str]:
        """Split a long text into sentences."""

        # Split the long text into paragraphs
        lines = text.split("\n")
        lines = [l + "\n" for l in lines[:-1]] + lines[-1:]
        paragraphs: list[str] = []
        buffer = ""
        for line in lines:
            if line.strip() == "":
                buffer += line
            else:
                if buffer:
                    paragraphs.append(buffer)
                buffer = line
        if buffer:
            paragraphs.append(buffer)

        # Split each paragraph independently
        sentences: list[str] = []
        for line in paragraphs:
            doc = self.nlp(line)
            sentences.extend([s.text for s in doc.sents if s.text.strip()])
        return sentences

    def split_text_to_tokenized_sentences(self, text: str) -> list[list[str]]:
        """Split the text into tokenized sentences."""

        doc = self.nlp(text)
        return [
            [tok.text for tok in sent] for sent in doc.sents if len(sent) > 0
        ]

    def split_text_to_chunks(self, text: str, window_size: int) -> list[str]:
        """Split the text into chunks of sentences, each chunk having a maximum of `window_size` words."""

        chunks: list[str] = []

        # Initialize the buffer
        buffer: list[str] = []
        buffer_len = 0

        # Split the text into sentences
        sentences = self.split_text_to_sentences(text=text)

        for sent in sentences:
            # Append a sentence to the current buffer
            buffer.append(sent)
            buffer_len += len(sent.split(" "))

            # Add the current buffer as a new chunk when it reaches the window size
            if buffer_len >= window_size:
                # Textualize the current buffer
                chunk = " ".join(buffer)
                chunks.append(chunk)

                # Initialize the buffer
                buffer = []
                buffer_len = 0

        if len(buffer) > 0:
            chunk = " ".join(buffer)
            chunks.append(chunk)
       
        return chunks

    ###########
    # Mapping from Passage to list[Passage] or Document
    ###########

    def split_passage_to_chunked_passages(
        self,
        passage: Passage,
        window_size: int,
    ) -> list[Passage]:
        """Split a Passage into chunked Passages, each chunk having a maximum of `window_size` words."""

        # Split the Passage content
        chunks = self.split_text_to_chunks(
            text=passage["text"],
            window_size=window_size
        )

        # Extract metadata except fields reconstructed for each chunk
        metadata = {
            key: value
            for key, value in passage.items()
            if key not in {"passage_key", "title", "text", "source_passage_key"}
        }

        # Create chunks with the optional title in the canonical field order
        if "title" in passage:
            return [
                {
                    "passage_key": f"{passage['passage_key']}/chunk#{chunk_i:04d}",
                    "title": passage["title"],
                    "text": chunk,
                    "source_passage_key": passage["passage_key"],
                    **metadata,
                }
                for chunk_i, chunk in enumerate(chunks)
            ]

        # Create chunks without adding an absent optional title
        return [
            {
                "passage_key": f"{passage['passage_key']}/chunk#{chunk_i:04d}",
                "text": chunk,
                "source_passage_key": passage["passage_key"],
                **metadata,
            }
            for chunk_i, chunk in enumerate(chunks)
        ]

    def convert_passage_to_document(
        self,
        doc_key: str,
        passage: Passage,
        do_tokenize: bool
    ) -> Document:
        """Convert a Passage to a Document, optionally tokenizing the sentences."""

        if do_tokenize:
            # Split the text to (tokenized) sentences
            sentences = self.split_text_to_tokenized_sentences(text=passage["text"])
            sentences = [" ".join(s) for s in sentences]

            # Prepend the title as the first (tokenized) sentence
            if "title" in passage:
                title = self.split_text_to_tokens(text=passage["title"])
                title = " ".join(title)
                sentences = [title] + sentences

            # Clean up the sentences
            sentences = self.remove_line_breaks(sentences=sentences)
        else:
            # Split the text to (raw) sentences
            sentences = self.split_text_to_sentences(text=passage["text"])

            # Prepend the title as the first (raw) sentence
            if "title" in passage:
                title = passage["title"]
                sentences = [title] + sentences

            # Clean up the sentences
            sentences = self.remove_line_breaks(sentences=sentences)

        # Create a Document object
        document = {
            "doc_key": doc_key,
            "source_passage": passage,
            "sentences": sentences
        }
        return document

    def convert_text_to_document(
        self,
        doc_key: str,
        text: str,
        title: str | None = None,
    ) -> Document:
        """Convert a text to a Document, optionally prepending the title as the first sentence."""

        # Split the text to (tokenized) sentences
        sentences = self.split_text_to_tokenized_sentences(text=text)
        sentences = [" ".join(s) for s in sentences]

        # Prepend the title as the first (tokenized) sentence
        if title:
            title = self.split_text_to_tokens(text=title)
            title = " ".join(title)
            sentences = [title] + sentences

        # Clean up the sentences
        sentences = self.remove_line_breaks(sentences=sentences)

        # Create a Document object
        document = {
            "doc_key": doc_key,
            "source_text": text,
            "sentences": sentences
        }

        return document

    def remove_line_breaks(self, sentences: list[str]) -> list[str]:
        """Remove line breaks from a list of sentences."""

        return [" ".join(s.split()) for s in sentences]
