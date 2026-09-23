from __future__ import annotations
 
import copy
import logging
# import re

import torch
from tqdm import tqdm

from .. import utils
from ..datatypes import (
    Question,
    ContextsForOneExample
)
from ..llms import HuggingFaceLLM, OpenAILLM
from .base import BaseQA


logger = logging.getLogger(__name__)


class LLMQA(BaseQA):

    def __init__(
        self,
        # External
        model: HuggingFaceLLM | OpenAILLM,
        # Internal
        prompt_template_name_or_path: str = "qa_04_without_context",
        # Optional
        n_contexts: int = -1,
    ):
        self.model = model
        self.prompt_template_name_or_path = prompt_template_name_or_path
        self.n_contexts = n_contexts

        self.prompt_template = utils.read_prompt_template(
            prompt_template_name_or_path=self.prompt_template_name_or_path,
            prompt_template_package_name="kapipe.qa.prompt_templates"
        )
        # Check requirements
        assert "{test_case_prompt}" in self.prompt_template

 
    def answer(
        self,
        question: Question,
        # Optional: context augmentation
        contexts_for_question: ContextsForOneExample | None = None
    ) -> Question:
        """Answer a single question."""

        with torch.no_grad():
            # Switch to inference mode for Hugging Face models
            if self.model.provider == "hf":
                self.model.llm.eval()

            # Generate the prompt
            prompt = self.generate_prompt(
                question=question,
                contexts_for_question=contexts_for_question
            )

            # Generate the response
            generated_text = self.model.generate(prompt)

            # Parse the generated response into structured fields
            answer, rationale, helpfulness_score = self.parse(
                question=question,
                generated_text=generated_text
            )

            # Integrate the structured fields into the document
            result = copy.deepcopy(question)
            result["output_answer"] = answer
            result["rationale"] = rationale
            result["helpfulness_score"] = helpfulness_score
            result["qa_prompt"] = prompt
            result["qa_generated_text"] = generated_text

            return result

    def generate_prompt(
        self,
        question: Question,
        # Optional: context augmentation
        contexts_for_question: ContextsForOneExample | None = None
    ) -> str:
        """Generate a prompt for the LLM."""

        if contexts_for_question is not None:
            # Extract the context texts from the `contexts_for_question`
            # Note that the number of contexts (context_texts) is
            # truncated to `n_contexts` here.
            if self.n_contexts >= 0:
                context_texts = [
                    utils.create_text_from_passage(passage=p, sep=" : ")
                    for p in contexts_for_question["contexts"][:self.n_contexts]
                ]
            else:
                context_texts = [
                    utils.create_text_from_passage(passage=p, sep=" : ")
                    for p in contexts_for_question["contexts"]
                ]

            actual_n_contexts = len(context_texts)

            # Generate the prompt section for the contexts
            if actual_n_contexts == 0:
                contexts_prompt = ""
            elif actual_n_contexts == 1:
                contexts_prompt = context_texts[0].strip()
            else:
                contexts_prompt = ""
                for c_i, c in enumerate(context_texts):
                    contexts_prompt += f"[{c_i+1}] {c.strip()}\n"
                    if c_i < actual_n_contexts - 1:
                        contexts_prompt += "\n"
                contexts_prompt = contexts_prompt.rstrip()
        else:
            contexts_prompt = ""

        # Get the prompt section for the test case (question)
        test_case_prompt = f"Question: {question['question']}".rstrip()

        # Add candidate answers (options) if available
        candidate_answers = question.get("candidate_answers")
        if candidate_answers and isinstance(candidate_answers, list):
            test_case_prompt += "\n"
            test_case_prompt += "Options:\n"

            # Join all candidates with a newline and a bullet point
            test_case_prompt += "\n".join([f"- {ans}" for ans in candidate_answers])

        test_case_prompt = test_case_prompt.rstrip()

        # Combine all the prompt sections
        prompt = self.prompt_template.format(
            contexts_prompt=contexts_prompt,
            test_case_prompt=test_case_prompt
        )

        return prompt

    def parse(
        self,
        question: Question,
        generated_text: str
    ) -> tuple[str, str, float]:
        """Parse the generated text into structured fields."""

        question_key = question["question_key"]

        ######
        # Pattern 1: Parse the generated text as string lines
        ######

        # # Parse each generated line
        # answer = generated_text
        # rationale = ""
        # score = 0.0
        # for generated_line in generated_text.split("\n"):
        #     generated_line = generated_line.strip()

        #     # Skip the empty line
        #     if generated_line == "":
        #         continue
            
        #     # Parse the generated_line
        #     if generated_line.startswith("Answer:"):
        #         answer = generated_line[len("Answer:"):].strip()
        #     elif generated_line.startswith("Rationale:"):
        #         rationale = generated_line[len("Rationale:"):].strip()
        #     elif generated_line.startswith("Score:"):
        #         # Parse a numeric score and an optional percent sign
        #         # match = re.search(r"Score:\s*([\d.]+)%?", generated_line)
        #         # if match:
        #         #     score_str = match.group(1)
        #         #     try:
        #         #         score = float(score_str)
        #         #         if f"{score_str}%" in generated_line:
        #         #             score /= 100.0
        #         #     except ValueError:
        #         #         logger.warning(f"Failed to parse score: {score_str}")
        #         #         score = 0.0
        #         match = re.search(r"Score:\s*([\d.]+)\s*(%)?", generated_line)
        #         if match:
        #             score_str = match.group(1)
        #             percent_mark = match.group(2)

        #             # Convert the parsed score into a float
        #             score = float(score_str)

        #             # Normalize percent scores into the 0.0-1.0 range
        #             if percent_mark == "%":
        #                 score /= 100.0
        #         else:
        #             score = 0.0
        #     else:
        #         logger.info(f"[{question_key}] Skipped a generated line of invalid formatting: '{generated_line}'")

        ######
        # Pattern 2: Parse the generated text as JSON
        ######

        # Parse the preferred JSON output format
        output = utils.safe_json_loads(
            generated_text=generated_text,
            fallback=None,
        )

        # If the JSON parsing failed, return a fallback response
        if output is None:
            logger.warning(
                f"[{question_key}] Failed to parse generated JSON: "
                f"{generated_text}"
            )
            return generated_text.strip(), "", 0.0

        # If any required field is missing, return a fallback response
        required_keys = {"rationale", "answer", "score"}
        if not required_keys.issubset(output.keys()):
            logger.warning(
                f"[{question_key}] Missing fields in generated JSON: "
                f"{output}"
            )
            return generated_text.strip(), "", 0.0

        # Extract every required field directly
        rationale = output["rationale"]
        answer = output["answer"]
        score = output["score"]

        # If any field has an invalid type, return a fallback response
        if not isinstance(rationale, str):
            logger.warning(
                f"[{question_key}] Invalid rationale: {rationale}"
            )
            return generated_text.strip(), "", 0.0
        if not isinstance(answer, str):
            logger.warning(
                f"[{question_key}] Invalid answer: {answer}"
            )
            return generated_text.strip(), "", 0.0
        if type(score) not in (int, float):
            logger.warning(
                f"[{question_key}] Invalid score: {score}"
            )
            return generated_text.strip(), "", 0.0

        # If the score is outside the valid range, return a fallback response
        score = float(score)
        if not 0.0 <= score <= 1.0:
            logger.warning(
                f"[{question_key}] Score outside [0.0, 1.0]: {score}"
            )
            return generated_text.strip(), "", 0.0

        return answer.strip(), rationale.strip(), score

    def submit_batch(
        self,
        questions: list[Question],
        contexts: list[ContextsForOneExample] | list[None] | None = None,
    ) -> list[str]:
        """Submit question-answering prompts and return OpenAI Batch IDs.

        Pass the same questions and contexts in the same order to
        fetch_and_process_batch(). Keep the model settings, prompt template,
        and n_contexts unchanged between calls.
        """

        # Validate that the model is an OpenAILLM instance for Batch API usage
        if not isinstance(self.model, OpenAILLM):
            raise TypeError("Batch API requires OpenAILLM")

        # Validate that contexts are provided for every question, or use None
        if contexts is None:
            contexts = [None] * len(questions)

        # Validate that there is one context entry for every question
        if len(contexts) != len(questions):
            raise ValueError(
                f"Expected {len(questions)} contexts, "
                f"but got {len(contexts)}"
            )

        # Generate prompts using the same method as answer()
        prompts: list[str] = []
        for question, contexts_for_question in zip(
            questions,
            contexts,
            strict=True,
        ):
            prompt: str = self.generate_prompt(
                question=question,
                contexts_for_question=contexts_for_question,
            )
            prompts.append(prompt)

        # Submit the prompts and get the Batch IDs
        batch_ids: list[str] = self.model.submit_batch(prompts=prompts)
        return batch_ids

    def fetch_and_process_batch(
        self,
        questions: list[Question],
        batch_ids: list[str],
        contexts: list[ContextsForOneExample] | list[None] | None = None,
    ) -> list[Question]:
        """Fetch responses and answer the original questions.

        Require the same questions, contexts, order, model settings, prompt
        template, and n_contexts used at submission. Raise an error if the
        batch is not complete.
        """

        # Validate that the model is an OpenAILLM instance for Batch API usage
        if not isinstance(self.model, OpenAILLM):
            raise TypeError("Batch API requires OpenAILLM")

        # Use an empty context for every question when contexts are omitted
        if contexts is None:
            contexts = [None] * len(questions)

        # Validate that there is one context entry for every question
        if len(contexts) != len(questions):
            raise ValueError(
                f"Expected {len(questions)} contexts, "
                f"but got {len(contexts)}"
            )

        # Fetch generated texts in the original request order
        generated_texts: list[str] = self.model.fetch_batch(
            batch_ids=batch_ids
        )

        # Validate that the number of generated texts matches the number of questions
        if len(generated_texts) != len(questions):
            raise ValueError(
                "The response count does not match the question count"
            )

        # Process each Batch response using the same procedure as answer()
        results: list[Question] = []
        for question, contexts_for_question, generated_text in zip(
            questions,
            contexts,
            generated_texts,
            strict=True,
        ):
            # Regenerate the original prompt stored in the result
            prompt: str = self.generate_prompt(
                question=question,
                contexts_for_question=contexts_for_question,
            )

            # Parse the generated response into structured fields
            answer, rationale, helpfulness_score = self.parse(
                question=question,
                generated_text=generated_text,
            )

            # Integrate the structured fields into a copied question
            result = copy.deepcopy(question)
            result["output_answer"] = answer
            result["rationale"] = rationale
            result["helpfulness_score"] = helpfulness_score
            result["qa_prompt"] = prompt
            result["qa_generated_text"] = generated_text
            results.append(result)

        return results
