from __future__ import annotations

import spacy
from spacy.lang.en import English

from ..datatypes import Passage, Document


class Chunker:

    def __init__(self, model_name: str | None = None):
        if model_name is None:
            self.nlp = English()
            self.nlp.add_pipe("sentencizer")
        else:
            self.nlp = spacy.load(model_name, disable=["ner", "textcat"])

    ###########
    # Mapping from text to tokens, sentences, or chunks
    ###########

    def split_text_to_tokens(self, text: str) -> list[str]:
        doc = self.nlp(text)
        return [tok.text for tok in doc]

    def split_text_to_sentences(self, text: str) -> list[str]:
        if len(text) < self.nlp.max_length:
            doc = self.nlp(text)
            return [s.text for s in doc.sents if s.text.strip()]
        return self._split_text_long(text=text)

    def _split_text_long(self, text: str) -> list[str]:
        # Split the long text into paragraphs
        lines = text.split("\n")
        lines = [l + "\n" for l in lines[:-1]] + lines[-1:]
        paragraphs = []
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
        sentences = []
        for line in paragraphs:
            doc = self.nlp(line)
            sentences.extend([s.text for s in doc.sents if s.text.strip()])
        return sentences

    def split_text_to_tokenized_sentences(self, text: str) -> list[list[str]]:
        doc = self.nlp(text)
        return [
            [tok.text for tok in sent] for sent in doc.sents if len(sent) > 0
        ]

    def split_text_to_chunks(self, text: str, window_size: int) -> list[str]:
        chunks = [] # list[str]

        # Initialize the buffer
        buffer = []
        buffer_len = 0

        # First split the text into sentences
        sentences = self.split_text_to_sentences(text=text)

        for sent in sentences:
            # Append a sentence to the current buffer
            buffer.append(sent)
            buffer_len += len(sent.split(" "))
            # If the buffer length exceeds the specified length, we add the current buffer as a new chunk
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
        window_size: int
    ) -> list[Passage]:
        # Split the Passage content
        chunks = self.split_text_to_chunks(
            text=passage["text"],
            window_size=window_size
        )
        # Extract meta data (neigher "title" nor "text")
        meta = {k: v for k, v in passage.items() if k not in {"title", "text"}}
        # Prepend "title"
        if "title" in passage:
            return [{"title": passage["title"], "text": chunk, **meta} for chunk in chunks]
        return [{"text": chunk, **meta} for chunk in chunks]

    def convert_passage_to_document(
        self,
        doc_key: str,
        passage: Passage,
        do_tokenize: bool
    ) -> Document:
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
        # Create a Document
        document = {
            "doc_key": doc_key,
            "source_passage": passage,
            "sentences": sentences
        }
        return document

    def remove_line_breaks(self, sentences: list[str]) -> list[str]:
        return [" ".join(s.split()) for s in sentences]

        

