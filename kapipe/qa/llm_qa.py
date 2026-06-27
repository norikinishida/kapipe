from __future__ import annotations
 
import copy
import logging
import os
import re
from typing import Any

import torch
from tqdm import tqdm

from .. import evaluation
from .. import utils
from ..datatypes import (
    Config,
    Question,
    ContextsForOneExample
)
from ..llms import HuggingFaceLLM, OpenAILLM
from ..resources import resolve_snapshot_path


logger = logging.getLogger(__name__)


class LLMQA:

    @classmethod
    def from_identifier(
        cls,
        model: HuggingFaceLLM | OpenAILLM,
        identifier: str,
    ) -> "LLMQA":

        # Resolve the public identifier to the corresponding local snapshot
        snapshot_path = resolve_snapshot_path(
            component_name="qa",
            method_name="llm_qa",
            identifier=identifier,
        )

        # Load the answerer from the resolved snapshot
        answerer = cls.from_snapshot(
            model=model,
            snapshot_path=snapshot_path,
        )

        # Store the public identifier for later inspection
        answerer.identifier = identifier

        return answerer

    @classmethod
    def from_snapshot(
        cls,
        model: HuggingFaceLLM | OpenAILLM,
        snapshot_path: str,
    ) -> "LLMQA":

        # Define the default paths for the resources in the snapshot
        config_path = snapshot_path + "/config"

        # Initialize the answerer from explicit snapshot resources
        answerer = cls(
            model=model,
            config=config_path,
        )

        # Store the snapshot path for later inspection
        answerer.snapshot_path = snapshot_path

        return answerer

    def __init__(
        self,
        model: HuggingFaceLLM | OpenAILLM,
        # Initialization
        config: Config | str | None = None,
    ):
        logger.info("########## LLMQA Initialization Starts ##########")

        self.model = model

        # Load the configuration
        if isinstance(config, str):
            config_path = config
            config = utils.get_hocon_config(config_path=config_path)
            logger.info(f"Loaded configuration from {config_path}")
        self.config = config
        logger.info(utils.pretty_format_dict(self.config))

        # Initialize the prompt processor, which generates prompts for the LLM
        self.prompt_processor = PromptProcessor(
            prompt_template_name_or_path=(
                self.config["prompt_template_name_or_path"]
            ),
            n_contexts=self.config["n_contexts"],
        )

        # Check the LLM provider
        self.provider = self.config["provider"]
        if self.provider not in ["hf", "openai"]:
            raise ValueError(f"Invalid provider: {self.provider}")
        logger.info("LLM is provided by an argument")

        logger.info("########## LLMQA Initialization Ends ##########")

    def save(self, snapshot_path: str) -> None:
        """Save the configuration."""

        config_path = snapshot_path + "/config"
        utils.write_json(config_path, self.config)

    def answer(
        self,
        question: Question,
        # Optional: context augmentation
        contexts_for_question: ContextsForOneExample | None = None
    ) -> Question:
        """Answer a single question."""

        with torch.no_grad():
            # Switch to inference mode for Hugging Face models
            if self.provider == "hf":
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
        contexts_for_question: ContextsForOneExample
    ) -> str:
        """Generate a prompt for the LLM."""

        return self.prompt_processor.generate(
           question=question,
           contexts_for_question=contexts_for_question
        )

    def parse(
        self,
        question: Question,
        generated_text: str
    ) -> tuple[str, float, str]:
        """Parse the generated text into structured fields."""

        question_key = question["question_key"]

        # Parse each generated line
        answer = generated_text
        rationale = ""
        score = 0.0
        for generated_line in generated_text.split("\n"):
            generated_text = generated_line.strip()

            # Skip the empty line
            if generated_line == "":
                continue
            
            # Parse the generated_line
            if generated_line.startswith("Answer:"):
                answer = generated_line[len("Answer:"):].strip()
            elif generated_line.startswith("Rationale:"):
                rationale = generated_line[len("Rationale:"):].strip()
            elif generated_line.startswith("Score:"):
                # score = float(generated_line[len("Score:"):].strip())
                match = re.search(r"Score:\s*([\d.]+)%?", generated_line)
                if match:
                    score_str = match.group(1)
                    try:
                        score = float(score_str)
                        if f"{score_str}%" in generated_line:
                            score /= 100.0
                    except ValueError:
                        logger.warning(f"Failed to parse score: {score_str}")
                        score = 0.0                        
                else:
                    # score = generated_line[len("Score:"):].strip()
                    # score = float(score.split(" ")[0])
                    score = 0.0
            else:
                logger.info(f"[{question_key}] Skipped a generated line of invalid formatting: '{generated_line}'")

        return answer, rationale, score
 
    def batch_answer(
        self,
        questions: list[Question],
        # optional: context augmentation
        contexts: list[ContextsForOneExample] | None = None
    ) -> list[Question]:
        """Answer a batch of questions."""

        results: list[Question] = []

        # Use a list of None for contexts if not provided
        if contexts is None:
            contexts = [None] * len(questions)

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


class PromptProcessor:
    """A class to generate prompts for the LLM based on a template and provided contexts."""
    
    def __init__(
        self,
        prompt_template_name_or_path: str,
        # optional: context
        n_contexts: int = -1,
    ): 

        self.prompt_template_name_or_path = prompt_template_name_or_path
        self.n_contexts = n_contexts

        #####
        # Prompt template
        #####

        self.prompt_template = utils.read_prompt_template(
            prompt_template_name_or_path=self.prompt_template_name_or_path,
            prompt_template_package_name="kapipe.qa.prompt_templates",
        )

        # Check requirements
        assert "{test_case_prompt}" in self.prompt_template

    def generate(
        self,
        question: Question,
        # Optional: context augmentation
        contexts_for_question: ContextsForOneExample | None = None
    ) -> str:
        """Generate a prompt for the LLM."""

        if contexts_for_question is not None:
            # Prepare contexts
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

            # Get prompt part for contexts
            # Note that the number of contexts (context_texts) is limited to n_contexts before calling this function
            contexts_prompt = self.generate_contexts_prompt(context_texts=context_texts)

        else:
            contexts_prompt = ""

        # Get the prompt part for test case
        test_case_prompt = self.generate_test_case_prompt(question=question)

        # Combine the prompt parts
        prompt = self.prompt_template.format(
            contexts_prompt=contexts_prompt,
            test_case_prompt=test_case_prompt
        )

        return prompt

    def generate_contexts_prompt(self, context_texts: list[str]) -> str:
        """Generate a prompt for the contexts."""

        n_contexts = len(context_texts)

        if n_contexts == 0:
            return ""

        if n_contexts == 1:
            return context_texts[0].strip()

        prompt = ""
        for c_i, content_text in enumerate(context_texts):
            prompt += f"[{c_i+1}] {content_text.strip()} \n"
            if c_i < n_contexts - 1:
                prompt += "\n"
    
        return prompt.rstrip()

    def generate_test_case_prompt(self, question: Question) -> str:
        """Generate a prompt for the test case."""

        prompt = f"Question: {self.generate_input_question_prompt(question)}".rstrip()

        # Check if candidate_answers exists and is a list
        candidate_answers = question.get("candidate_answers")
        if candidate_answers and isinstance(candidate_answers, list):
            # Append options to the prompt
            prompt += "\nOptions:\n"

            # Join all candidates with a newline and a bullet point
            prompt += "\n".join([f"- {ans}" for ans in candidate_answers])
            
        return prompt.rstrip()
 
    def generate_input_question_prompt(self, question: Question) -> str:
        """Generate a prompt for the input question."""

        return question["question"]

    def generate_output_prompt(self, question: Question) -> str:
        """Generate a prompt for the output answer."""

        answer = ", ".join([a["answer"] for a in question["answers"]])
        score = 0.69 # FIXME
        prompt = ""
        prompt += f"Answer: {answer}\n"
        prompt += f"Score: {score}\n"

        return prompt.rstrip()


#####################
# Trainer (Evaluator)
#####################


class LLMQATrainer:

    def __init__(self, base_output_path: str):
        self.base_output_path = base_output_path
        self.paths = self.get_paths()

    def get_paths(self) -> dict[str,str]:
        paths = {}

        # configurations
        paths["snapshot_path"] = self.base_output_path

        # evaluation outputs
        paths["dev_gold_path"] = os.path.join(self.base_output_path, "dev.gold.json")
        paths["dev_pred_path"] = os.path.join(self.base_output_path, "dev.pred.json")
        paths["dev_eval_path"] = os.path.join(self.base_output_path, "dev.eval.json")
        paths["test_gold_path"] = os.path.join(self.base_output_path, "test.gold.json")
        paths["test_pred_path"] = os.path.join(self.base_output_path, "test.pred.json")
        paths["test_eval_path"] = os.path.join(self.base_output_path, "test.eval.json")

        return paths

    def setup_dataset(
        self,
        answerer: LLMQA,
        questions: list[Question],
        split: str
    ) -> None:
        # Cache the gold annotations for evaluation
        gold_path = self.paths[f"{split}_gold_path"]
        if not os.path.exists(gold_path):
            gold_questions = []
            for question in tqdm(questions, desc="dataset setup"):
                gold_question = copy.deepcopy(question)
                gold_questions.append(gold_question)
            utils.write_json(gold_path, gold_questions)
            logger.info(f"Saved the gold annotations for evaluation in {gold_path}")

    def save_answerer(self, answerer: LLMQA) -> None:
        answerer.save(snapshot_path=self.paths["snapshot_path"])

    def evaluate(
        self,
        answerer: LLMQA,
        questions: list[Question],
        contexts: list[ContextsForOneExample] | None,
        split: str,
        #
        metric: str = "accuracy",
        prediction_only: bool = False,
        get_scores_only: bool = False
    ) -> dict[str, Any] | None:
        # Apply the answerer to the given questions,
        # optionally based on the contexts
        results = answerer.batch_answer(
            questions=questions,
            contexts=contexts
        )

        # Save the prediction results
        utils.write_json(self.paths[f"{split}_pred_path"], results)

        # Save the prompt-response pairs in plain text
        with open(self.paths[f"{split}_pred_path"].replace(".json", ".txt"), "w") as f:
            for result in results:
                question_key = result["question_key"]
                prompt = result["qa_prompt"]
                generated_text = result["qa_generated_text"]
                f.write("-------------------------------------\n\n")
                f.write(f"QUESTION_KEY: {question_key}\n\n")
                f.write("PROMPT:\n")
                f.write(prompt + "\n\n")
                f.write("GENERATED TEXT:\n")
                f.write(generated_text + "\n\n")
                f.flush()

        if prediction_only:
            return

        # Evaluate the predicted answers
        if metric == "recall":
             scores = evaluation.qa.recall(
                pred_path=self.paths[f"{split}_pred_path"],
                gold_path=self.paths[f"{split}_gold_path"],
                exact_match=False
            )
        elif metric == "llm4eval":
             scores = evaluation.qa.llm4eval(
                pred_path=self.paths[f"{split}_pred_path"],
                gold_path=self.paths[f"{split}_gold_path"],
            )
        else:
            scores = evaluation.qa.accuracy(
                pred_path=self.paths[f"{split}_pred_path"],
                gold_path=self.paths[f"{split}_gold_path"],
                exact_match=False
            )

        if get_scores_only:
            return scores

        # Save the evaluation results
        utils.write_json(self.paths[f"{split}_eval_path"], scores)
        logger.info(utils.pretty_format_dict(scores))
        return scores
