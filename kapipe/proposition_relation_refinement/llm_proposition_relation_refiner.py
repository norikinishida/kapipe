from __future__ import annotations

import logging
from typing import Any

import torch

from .. import utils
from ..datatypes import Passage
from ..llms import BaseLLM, OpenAILLM
from .base import BasePropositionRelationRefiner


logger = logging.getLogger(__name__)


class LLMPropositionRelationRefiner(BasePropositionRelationRefiner):
    """Proposition relation refiner based on a large language model."""

    def __init__(
        self,
        # External
        model: BaseLLM,
        # Internal
        prompt_template_name_or_path: str = (
            "proposition_relation_refinement_01_without_timestamp"
        ),
        # Optional
        use_timestamp: bool = False,
    ) -> None:

        self.model = model
        self.prompt_template_name_or_path = prompt_template_name_or_path
        self.use_timestamp = use_timestamp

        # Load the prompt template
        self.prompt_template = utils.read_prompt_template(
            prompt_template_name_or_path=self.prompt_template_name_or_path,
            prompt_template_package_name=(
                "kapipe.proposition_relation_refinement.prompt_templates"
            ),
        )

        # Validate every placeholder required by prompt generation
        required_placeholders = [
            "{subject}",
            "{object}",
            "{predicted_relation}",
            "{predicted_explanation}",
        ]
        for placeholder in required_placeholders:
            if placeholder not in self.prompt_template:
                raise ValueError(
                    f"The prompt template must contain {placeholder}."
                )

    def refine(
        self,
        triple: dict[str, Any],
    ) -> dict[str, Any]:
        """Verify and refine a proposition relation triple."""

        with torch.no_grad():
            # Switch a Hugging Face model to inference mode
            if self.model.provider == "hf":
                self.model.llm.eval()

            # Generate the prompt
            prompt = self.generate_prompt(triple=triple)

            # Generate the response
            generated_text = self.model.generate(prompt)

            # Parse the generated response into a refined triple
            refined_triple = self.parse(
                triple=triple,
                generated_text=generated_text,
            )

        return refined_triple

    def generate_prompt(
        self,
        triple: dict[str, Any],
    ) -> str:
        """Generate a prompt from a proposition relation triple."""

        # Verbalize both propositions under the same timestamp policy
        subject_text = self.textualize_proposition(
            proposition=triple["head"],
        )
        object_text = self.textualize_proposition(
            proposition=triple["tail"],
        )

        # Fill the prompt with the propositions and current prediction
        prompt = self.prompt_template.format(
            subject=subject_text,
            object=object_text,
            predicted_relation=triple["relation"],
            predicted_explanation=triple["explanation"],
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

        # Avoid accessing optional timestamp metadata by default
        return proposition["text"].strip()

    def parse(
        self,
        triple: dict[str, Any],
        generated_text: str,
    ) -> dict[str, Any]:
        """Parse the generated text into a refined relation triple."""

        # Parse the generated response as a JSON object
        output = utils.safe_json_loads(
            generated_text=generated_text,
            fallback={},
        )

        # Extract the refined relation and explanation from the parsed output
        refined_relation = output.get("verified_relation", "NOREL")
        refined_explanation = output.get("explanation", "")
        if not isinstance(refined_relation, str):
            logger.warning(
                f"Invalid refined relation: {refined_relation}"
            )
            refined_relation = "NOREL"
        if not isinstance(refined_explanation, str):
            logger.warning(
                f"Invalid refined explanation: {refined_explanation}"
            )
            refined_explanation = ""

        # If the refined relation is "NOREL", it indicates no valid relation was found.
        # Otherwise, return the refined relation and explanation.
        if refined_relation == "NOREL":
            return {
                "head": triple["head"],
                "relation": "NOREL",
                "tail": triple["tail"],
                "explanation": refined_explanation,
                #
                "pre_refinement_relation": triple["relation"],
                "pre_refinement_explanation": triple["explanation"],
            }
        else:
            return {
                "head": triple["head"],
                "relation": refined_relation,
                "tail": triple["tail"],
                "explanation": refined_explanation,
                #
                "pre_refinement_relation": triple["relation"],
                "pre_refinement_explanation": triple["explanation"],
            }

    def submit_batch(
        self,
        triples: list[dict[str, Any]],
    ) -> list[str]:
        """Submit relation-refinement prompts and return OpenAI Batch IDs.

        Pass the same triples in the same order to fetch_and_process_batch().
        Keep the model settings, prompt template, and use_timestamp unchanged
        between calls.
        """

        # Validate that the model is an OpenAILLM instance for Batch API usage
        if not isinstance(self.model, OpenAILLM):
            raise TypeError("Batch API requires OpenAILLM")

        # Generate prompts using the same method as refine()
        prompts: list[str] = []
        for triple in triples:
            prompt: str = self.generate_prompt(triple=triple)
            prompts.append(prompt)

        # Submit the prompts and get the Batch IDs
        batch_ids: list[str] = self.model.submit_batch(prompts=prompts)
        return batch_ids

    def fetch_and_process_batch(
        self,
        triples: list[dict[str, Any]],
        batch_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Fetch responses and refine the original proposition relations.

        Require the same triples, order, model settings, prompt template, and
        use_timestamp used at submission. Raise an error if the batch is not
        complete.
        """

        # Validate that the model is an OpenAILLM instance for Batch API usage
        if not isinstance(self.model, OpenAILLM):
            raise TypeError("Batch API requires OpenAILLM")

        # Fetch generated texts in the original request order
        generated_texts: list[str] = self.model.fetch_batch(batch_ids=batch_ids)

        # Validate that the number of generated texts matches the number of 
        # submitted triples.
        if len(generated_texts) != len(triples):
            raise ValueError(
                "The response count does not match the triple count"
            )

        # Parse each Batch response using the same procedure as refine()
        refined_triples: list[dict[str, Any]] = []
        for triple, generated_text in zip(
            triples,
            generated_texts,
            strict=True,
        ):
            # Parse the generated response into a refined triple
            refined_triple: dict[str, Any] = self.parse(
                triple=triple,
                generated_text=generated_text,
            )
            refined_triples.append(refined_triple)

        return refined_triples
