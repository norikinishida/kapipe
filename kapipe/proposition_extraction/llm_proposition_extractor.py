from __future__ import annotations

import torch

from .. import utils
from ..datatypes import Passage
from ..llms import BaseLLM, OpenAILLM
from .base import BasePropositionExtractor


class LLMPropositionExtractor(BasePropositionExtractor):
    """Proposition extractor based on a large language model."""

    def __init__(
        self,
        # External
        model: BaseLLM,
        # Internal
        prompt_template_name_or_path: str = "proposition_extraction_01",
    ) -> None:

        self.model = model
        self.prompt_template_name_or_path = prompt_template_name_or_path

        # Load the prompt template for proposition extraction
        self.prompt_template = utils.read_prompt_template(
            prompt_template_name_or_path=self.prompt_template_name_or_path,
            prompt_template_package_name=(
                "kapipe.proposition_extraction.prompt_templates"
            ),
        )

        # Validate the prompt template
        if "{input_text}" not in self.prompt_template:
            raise ValueError(
                "The prompt template must contain {input_text}."
            )

    def extract(
        self,
        passage: Passage,
    ) -> list[Passage]:
        """Extract propositions from a passage."""

        with torch.no_grad():
            # Switch a Hugging Face model to inference mode
            if self.model.provider == "hf":
                self.model.llm.eval()

            # Generate the prompt
            prompt = self.generate_prompt(passage=passage)

            # Generate the response
            generated_text = self.model.generate(prompt)

            # Parse the generated response into proposition statements
            statements = self.parse(generated_text=generated_text)

            # Treat the title as the first proposition 
            # when it is available and non-empty.
            if "title" in passage and passage["title"].strip():
                statements = [passage["title"].strip()] + statements

            # Preserve metadata except fields reconstructed for each proposition
            metadata = {
                key: value
                for key, value in passage.items()
                if key not in {"passage_key", "title", "text", "source_passage_key"}
            }

            # Convert each statement into a proposition in the canonical field order
            propositions: list[Passage] = [
                {
                    "passage_key": (
                        f"{passage['passage_key']}/proposition#{proposition_i:04d}"
                    ),
                    "text": statement,
                    "source_passage_key": passage["passage_key"],
                    **metadata,
                }
                for proposition_i, statement in enumerate(statements)
            ]

        return propositions

    def generate_prompt(
        self,
        passage: Passage,
    ) -> str:
        """Generate the proposition-extraction prompt."""

        return self.prompt_template.format(input_text=passage["text"])

    def parse(
        self,
        generated_text: str,
    ) -> list[str]:
        """Parse the generated response into proposition statements."""

        statements = [
            line.strip()
            for line in generated_text.split("\n")
            if line.strip() != ""
        ]

        return statements

    def submit_batch(
        self,
        passages: list[Passage],
    ) -> list[str]:
        """Submit passage prompts and return the OpenAI Batch IDs.

        Pass the same passages in the same order to fetch_and_process_batch().
        Keep the model settings and prompt template unchanged between calls.
        """

        # Validate that the model is an OpenAILLM instance for Batch API usage
        if not isinstance(self.model, OpenAILLM):
            raise TypeError("Batch API requires OpenAILLM")

        # Generate prompts using the same method as extract()
        prompts: list[str] = []
        for passage in passages:
            prompt: str = self.generate_prompt(passage=passage)
            prompts.append(prompt)

        # Submit the prompts and get the Batch IDs
        batch_ids: list[str] = self.model.submit_batch(prompts=prompts)
        return batch_ids

    def fetch_and_process_batch(
        self,
        passages: list[Passage],
        batch_ids: list[str],
    ) -> list[Passage]:
        """Fetch responses and extract propositions from the original passages.

        Require the same passages, order, model settings, and prompt template
        used at submission. Raise an error if the batch is not complete.
        """

        # Validate that the model is an OpenAILLM instance for Batch API usage
        if not isinstance(self.model, OpenAILLM):
            raise TypeError("Batch API requires OpenAILLM")

        # Fetch generated texts in the original request order
        generated_texts: list[str] = self.model.fetch_batch(batch_ids=batch_ids)

        # Validate that the number of generated texts matches the number of passages
        if len(generated_texts) != len(passages):
            raise ValueError("The response count does not match the passage count")

        # Process each Batch response using the same procedure as extract()
        propositions: list[Passage] = []
        for passage, generated_text in zip(passages, generated_texts, strict=True):
            # Parse the generated response into proposition statements
            statements = self.parse(generated_text=generated_text)

            # Treat the title as the first proposition when it is available
            if "title" in passage:
                statements = [passage["title"].strip()] + statements

            # Preserve metadata except fields reconstructed for each proposition
            metadata = {
                key: value
                for key, value in passage.items()
                if key not in {"passage_key", "title", "text", "source_passage_key"}
            }

            # Convert each statement into a proposition in the canonical field order
            propositions_for_passage: list[Passage] = [
                {
                    "passage_key": (
                        f"{passage['passage_key']}/proposition#{proposition_i:04d}"
                    ),
                    "text": statement,
                    "source_passage_key": passage["passage_key"],
                    **metadata,
                }
                for proposition_i, statement in enumerate(statements)
            ]
            propositions.extend(propositions_for_passage)

        return propositions
