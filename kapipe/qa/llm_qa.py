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
        prompt_template_name_or_path: str,
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

        # Require every field defined by the prompt contract
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

        # Validate the generated field types
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

        # Reject scores outside the required range
        score = float(score)
        if not 0.0 <= score <= 1.0:
            logger.warning(
                f"[{question_key}] Score outside [0.0, 1.0]: {score}"
            )
            return generated_text.strip(), "", 0.0

        return answer.strip(), rationale.strip(), score
 
    def batch_answer(
        self,
        questions: list[Question],
        # optional: context augmentation
        contexts: list[ContextsForOneExample] | list[None] | None = None
    ) -> list[Question]:
        """Answer a batch of questions."""

        results: list[Question] = []

        # Use a list of None for contexts if not provided
        if contexts is None:
            contexts = [None] * len(questions)

        # Validate that the number of contexts matches the number of questions
        if len(contexts) != len(questions):
            raise ValueError(
                f"Expected {len(questions)} contexts, but got {len(contexts)}"
            )

        for question, contexts_for_q in tqdm(
            zip(questions, contexts),
            total=len(questions),
            desc="answering steps"
        ):
            result = self.answer(
                question=question,
                contexts_for_question=contexts_for_q
            )
            results.append(result)

        return results

    def submit_batch(
        self,
        questions: list[Question],
        contexts: list[ContextsForOneExample] | list[None] | None = None,
    ) -> str:
        """Submit question-answering prompts and return the OpenAI Batch ID.

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

        # Submit the batch of prompts and get the batch ID
        batch_id: str = self.model.submit_batch(prompts=prompts)
        return batch_id

    def fetch_and_process_batch(
        self,
        questions: list[Question],
        batch_id: str,
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
            batch_id=batch_id
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


#####################
# Trainer (Evaluator)
#####################


# class LLMQATrainer:

#     def __init__(self, base_output_path: str):
#         self.base_output_path = base_output_path
#         self.paths = self.get_paths()

#     def get_paths(self) -> dict[str,str]:
#         paths = {}

#         # configurations
#         paths["snapshot_path"] = self.base_output_path

#         # evaluation outputs
#         paths["dev_gold_path"] = os.path.join(self.base_output_path, "dev.gold.json")
#         paths["dev_pred_path"] = os.path.join(self.base_output_path, "dev.pred.json")
#         paths["dev_eval_path"] = os.path.join(self.base_output_path, "dev.eval.json")
#         paths["test_gold_path"] = os.path.join(self.base_output_path, "test.gold.json")
#         paths["test_pred_path"] = os.path.join(self.base_output_path, "test.pred.json")
#         paths["test_eval_path"] = os.path.join(self.base_output_path, "test.eval.json")

#         return paths

#     def setup_dataset(
#         self,
#         answerer: LLMQA,
#         questions: list[Question],
#         split: str
#     ) -> None:
#         # Cache the gold annotations for evaluation
#         gold_path = self.paths[f"{split}_gold_path"]
#         if not os.path.exists(gold_path):
#             gold_questions = []
#             for question in tqdm(questions, desc="dataset setup"):
#                 gold_question = copy.deepcopy(question)
#                 gold_questions.append(gold_question)
#             utils.write_json(gold_path, gold_questions)
#             logger.info(f"Saved the gold annotations for evaluation in {gold_path}")

#     def save_answerer(self, answerer: LLMQA) -> None:
#         answerer.save(snapshot_path=self.paths["snapshot_path"])

#     def evaluate(
#         self,
#         answerer: LLMQA,
#         questions: list[Question],
#         contexts: list[ContextsForOneExample] | None,
#         split: str,
#         #
#         metric: str = "accuracy",
#         prediction_only: bool = False,
#         get_scores_only: bool = False
#     ) -> dict[str, Any] | None:
#         # Apply the answerer to the given questions,
#         # optionally based on the contexts
#         results = answerer.batch_answer(
#             questions=questions,
#             contexts=contexts
#         )

#         # Save the prediction results
#         utils.write_json(self.paths[f"{split}_pred_path"], results)

#         # Save the prompt-response pairs in plain text
#         with open(self.paths[f"{split}_pred_path"].replace(".json", ".txt"), "w") as f:
#             for result in results:
#                 question_key = result["question_key"]
#                 prompt = result["qa_prompt"]
#                 generated_text = result["qa_generated_text"]
#                 f.write("-------------------------------------\n\n")
#                 f.write(f"QUESTION_KEY: {question_key}\n\n")
#                 f.write("PROMPT:\n")
#                 f.write(prompt + "\n\n")
#                 f.write("GENERATED TEXT:\n")
#                 f.write(generated_text + "\n\n")
#                 f.flush()

#         if prediction_only:
#             return

#         # Evaluate the predicted answers
#         if metric == "recall":
#              scores = evaluation.qa.recall(
#                 pred_path=self.paths[f"{split}_pred_path"],
#                 gold_path=self.paths[f"{split}_gold_path"],
#                 exact_match=False
#             )
#         elif metric == "llm4eval":
#              scores = evaluation.qa.llm4eval(
#                 pred_path=self.paths[f"{split}_pred_path"],
#                 gold_path=self.paths[f"{split}_gold_path"],
#             )
#         else:
#             scores = evaluation.qa.accuracy(
#                 pred_path=self.paths[f"{split}_pred_path"],
#                 gold_path=self.paths[f"{split}_gold_path"],
#                 exact_match=False
#             )

#         if get_scores_only:
#             return scores

#         # Save the evaluation results
#         utils.write_json(self.paths[f"{split}_eval_path"], scores)
#         logger.info(utils.pretty_format_dict(scores))
#         return scores
