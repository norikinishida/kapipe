from __future__ import annotations

import logging
from typing import Any

import torch

from .. import utils
from ..datatypes import Passage
from ..llms import BaseLLM, OpenAILLM
from .base import BasePropositionRelationExtractor


logger = logging.getLogger(__name__)


class LLMPropositionRelationExtractor(BasePropositionRelationExtractor):
    """Proposition relation extractor based on a large language model."""

    def __init__(
        self,
        # External
        model: BaseLLM,
        # Internal
        prompt_template_name_or_path: str = "proposition_relation_extraction_01",
        # Optional
        use_timestamp: bool = False,
    ) -> None:

        self.model = model
        self.prompt_template_name_or_path = prompt_template_name_or_path
        self.use_timestamp = use_timestamp

        # Load the prompt template for proposition relation extraction
        self.prompt_template = utils.read_prompt_template(
            prompt_template_name_or_path=self.prompt_template_name_or_path,
            prompt_template_package_name=(
                "kapipe.proposition_relation_extraction.prompt_templates"
            ),
        )

        # Validate the prompt template
        for placeholder in ["{subject}", "{object_list}"]:
            if placeholder not in self.prompt_template:
                raise ValueError(
                    f"The prompt template must contain {placeholder}."
                )

    def extract(
        self,
        head_proposition: Passage,
        tail_propositions: list[Passage],
    ) -> list[dict[str, Any]]:
        """Extract relations from a head proposition to tail propositions."""

        # Avoid an unnecessary LLM call when no candidate tail exists
        if len(tail_propositions) == 0:
            return []

        with torch.no_grad():
            # Switch a Hugging Face model to inference mode
            if self.model.provider == "hf":
                self.model.llm.eval()

            # Generate the prompt
            prompt = self.generate_prompt(
                head_proposition=head_proposition,
                tail_propositions=tail_propositions,
            )

            # Generate the response
            generated_text = self.model.generate(prompt)

            # Parse the generated response into proposition relation records
            triples = self.parse(
                head_proposition=head_proposition,
                tail_propositions=tail_propositions,
                generated_text=generated_text,
            )

        return triples

    def generate_prompt(
        self,
        head_proposition: Passage,
        tail_propositions: list[Passage],
    ) -> str:
        """Generate a prompt from a head proposition and tail propositions."""

        # Verbalize the head proposition
        head_text = self.textualize_proposition(
            proposition=head_proposition,
        )

        # Verbalize candidate tails while preserving their input order
        tail_lines: list[str] = []
        for tail_i, tail_proposition in enumerate(tail_propositions):
            tail_text = self.textualize_proposition(
                proposition=tail_proposition,
            )
            tail_lines.append(f"- {tail_i}: {tail_text}")
        object_list = "\n".join(tail_lines)

        # Fill the prompt template with the verbalized propositions
        prompt = self.prompt_template.format(
            subject=head_text,
            object_list=object_list,
        )

        return prompt

    def textualize_proposition(
        self,
        proposition: Passage,
    ) -> str:
        """Verbalize a proposition into text."""

        # Include the timestamp only when temporal information is requested
        if self.use_timestamp:
            timestamp = proposition["timestamp"].strip()
            text = proposition["text"].strip()
            return f"[As of {timestamp}] {text}"

        # Use only the required Passage text by default
        return proposition["text"].strip()

    def parse(
        self,
        head_proposition: Passage,
        tail_propositions: list[Passage],
        generated_text: str,
    ) -> list[dict[str, Any]]:
        """Parse the generated text into proposition relation records."""

        # Parse the generated response as a JSON array
        output = utils.safe_json_loads(
            generated_text=generated_text,
            fallback=[],
            list_type=True,
        )

        # Convert valid generated entries into proposition relation records
        triples: list[dict[str, Any]] = []
        for entry in output:
            # Skip malformed array elements
            if not isinstance(entry, dict):
                logger.warning(
                    "Skipped a proposition relation that is not a JSON object: %s",
                    entry,
                )
                continue

            # Skip entries that do not contain all required keys
            required_keys = {"object_index", "relation", "explanation"}
            if not required_keys.issubset(entry.keys()):
                logger.warning(
                    "Skipped a proposition relation with missing fields: %s",
                    entry,
                )
                continue

            # Skip entries with an invalid object_index
            tail_index = entry["object_index"]
            if isinstance(tail_index, bool) or not isinstance(tail_index, int):
                logger.warning(
                    "Skipped a proposition relation with a non-integer "
                    "object_index: %s",
                    entry,
                )
                continue

            # Skip entries with an out-of-range object_index
            tail_index = int(tail_index)
            if not 0 <= tail_index < len(tail_propositions):
                logger.warning(
                    "Skipped a proposition relation with an out-of-range "
                    "object_index: %s",
                    entry,
                )
                continue

            # Retrieve the tail proposition corresponding to the validated index
            tail_proposition = tail_propositions[tail_index]

            # Skip entries with a non-string relation label
            relation_label = entry["relation"]
            if not isinstance(relation_label, str):
                logger.warning(
                    "Skipped a proposition relation with a non-string label: %s",
                    entry,
                )
                continue
            relation_label = relation_label.strip()

            # Skip entries with a non-string explanation
            explanation = entry["explanation"]
            if not isinstance(explanation, str):
                logger.warning(
                    "Skipped a proposition relation with a non-string "
                    "explanation: %s",
                    entry,
                )
                continue

            # Normalize the explanation
            explanation = explanation.strip()

            # Skip entries with a "NOREL" relation label
            if relation_label == "NOREL":
                continue

            # Record the directed relation from the head to the selected tail
            triples.append({
                "head": head_proposition,
                "relation": relation_label,
                "tail": tail_proposition,
                "explanation": explanation,
            })

        return triples

    def submit_batch(
        self,
        head_propositions: list[Passage],
        batch_tail_propositions: list[list[Passage]],
    ) -> list[str]:
        """Submit relation-extraction prompts and return OpenAI Batch IDs.

        Pass the same propositions in the same order to
        fetch_and_process_batch(). Keep the model settings, prompt template,
        and use_timestamp unchanged between calls.
        """

        # Validate that the model is an OpenAILLM instance for Batch API usage
        if not isinstance(self.model, OpenAILLM):
            raise TypeError("Batch API requires OpenAILLM")

        # Validate that there is one candidate tail list for every head proposition
        if len(head_propositions) != len(batch_tail_propositions):
            raise ValueError(
                "The number of head propositions does not match "
                "the number of tail proposition lists"
            )

        # Generate prompts while skipping inputs handled without an LLM call
        prompts: list[str] = []
        for head_proposition, tail_propositions in zip(
            head_propositions,
            batch_tail_propositions,
            strict=True,
        ):
            # Avoid an unnecessary LLM call when no candidate tail exists
            if len(tail_propositions) == 0:
                continue

            # Generate the prompt using the same method as extract()
            prompt: str = self.generate_prompt(
                head_proposition=head_proposition,
                tail_propositions=tail_propositions,
            )
            prompts.append(prompt)

        # Validate that there is at least one prompt to submit
        if len(prompts) == 0:
            raise ValueError(
                "No prompts to submit because all tail proposition lists are empty"
            )

        # Submit the prompts and get the Batch IDs
        batch_ids: list[str] = self.model.submit_batch(prompts=prompts)
        return batch_ids

    def fetch_and_process_batch(
        self,
        head_propositions: list[Passage],
        batch_tail_propositions: list[list[Passage]],
        batch_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Fetch responses and extract proposition relation records.

        Require the same propositions, order, model settings, prompt template,
        and use_timestamp used at submission. Raise an error if the batch is
        not complete.
        """

        # Validate that the model is an OpenAILLM instance for Batch API usage
        if not isinstance(self.model, OpenAILLM):
            raise TypeError("Batch API requires OpenAILLM")

        # Validate that there is one candidate tail list for every head proposition
        if len(head_propositions) != len(batch_tail_propositions):
            raise ValueError(
                "The number of head propositions does not match "
                "the number of tail proposition lists"
            )

        # Preserve only inputs submitted to the Batch API
        batch_inputs: list[tuple[Passage, list[Passage]]] = []
        for head_proposition, tail_propositions in zip(
            head_propositions,
            batch_tail_propositions,
            strict=True,
        ):
            # Avoid an unnecessary LLM call when no candidate tail exists.
            # Match submit_batch(), which skips empty tail lists.
            if len(tail_propositions) == 0:
                continue

            batch_inputs.append((head_proposition, tail_propositions))

        # Fetch generated texts in the original request order
        generated_texts: list[str] = self.model.fetch_batch(batch_ids=batch_ids)

        # Validate that the number of generated texts matches the number of 
        # submitted inputs.
        if len(generated_texts) != len(batch_inputs):
            raise ValueError(
                "The response count does not match the submitted input count"
            )

        # Parse each Batch response using the same procedure as extract()
        triples: list[dict[str, Any]] = []
        for (head_proposition, tail_propositions,), generated_text in zip(
            batch_inputs,
            generated_texts,
            strict=True,
        ):
            triples_for_head: list[dict[str, Any]] = self.parse(
                head_proposition=head_proposition,
                tail_propositions=tail_propositions,
                generated_text=generated_text,
            )
            triples.extend(triples_for_head)

        return triples
