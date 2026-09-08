from __future__ import annotations

import torch

from .. import utils
from ..datatypes import Passage
from ..llms import BaseLLM
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
            prompt = self.prompt_template.format(
                input_text=passage["text"],
            )

            # Generate the response
            generated_text = self.model.generate(prompt)

            # Parse the generated response into proposition statements
            statements = [
                line.strip()
                for line in generated_text.split("\n")
                if line.strip() != ""
            ]

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
