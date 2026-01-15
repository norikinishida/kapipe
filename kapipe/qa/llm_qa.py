from __future__ import annotations
 
import copy
import logging
import os
import re
from typing import Any

import torch
from tqdm import tqdm

from ..datatypes import (
    Config,
    Question,
    ContextsForOneExample
)
from .. import utils
from .. import evaluation
from ..llms import HuggingFaceLLM, OpenAILLM


logger = logging.getLogger(__name__)


class LLMQA:

    def __init__(
        self,
        device: str | int = 0,
        # Initialization
        config: Config | str | None = None,
        # Loading
        path_snapshot: str | None = None,
        # Misc.
        model: HuggingFaceLLM | OpenAILLM | None = None
    ):
        logger.info("########## LLMQA Initialization Starts ##########")

        if isinstance(device, int):
            self.device = f"cuda:{0}"

        self.device = device
        self.path_snapshot = path_snapshot

        if path_snapshot is not None:
            assert config is None
            config = path_snapshot + "/config"

        # Load the configuration
        if isinstance(config, str):
            tmp = config
            config = utils.get_hocon_config(config_path=config, config_name=None)
            logger.info(f"Loaded configuration from {tmp}")
        self.config = config
        logger.info(utils.pretty_format_dict(self.config))

        # Initialize the prompt processor
        self.prompt_processor = PromptProcessor(
            prompt_template_name_or_path=config["prompt_template_name_or_path"],
            n_contexts=config["n_contexts"],
        )

        # Initialize the model
        self.model_name = config["model_name"]
        assert self.model_name in ["hf", "openai"]
        if model is not None:
            self.model = model
            logger.info("LLM is provided")
        elif self.model_name == "hf":
            self.model = HuggingFaceLLM(
                device=device,
                # Model
                llm_name_or_path=config["llm_name_or_path"],
                # Generation
                max_new_tokens=config["max_new_tokens"],
                quantization_bits=config["quantization_bits"],
            )
        else:
            self.model = OpenAILLM(
                openai_model_name=config["openai_model_name"],
                max_new_tokens=config["max_new_tokens"]
            )
        # self.model.llm.to(self.model.device)

        logger.info("########## LLMQA Initialization Ends ##########")

    def save(self, path_snapshot: str) -> None:
        path_config = path_snapshot + "/config"
        utils.write_json(path_config, self.config)

    def answer(
        self,
        question: Question,
        # optional: context augmentation
        contexts_for_question: ContextsForOneExample | None = None
    ) -> Question:
        with torch.no_grad():
            if self.model_name == "hf":
                # Switch to inference mode
                self.model.llm.eval()

            # Generate a prompt
            prompt = self.generate_prompt(
                question=question,
                contexts_for_question=contexts_for_question
            )

            # Generate a response
            generated_text = self.model.generate(prompt)

            # Parse the generated response
            answer, rationale, helpfulness_score = self.parse(
                question=question,
                generated_text=generated_text
            )

            # Integrate
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
        return self.prompt_processor.generate(
           question=question,
           contexts_for_question=contexts_for_question
        )

    def parse(
        self,
        question: Question,
        generated_text: str
    ) -> tuple[str, float, str]:
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
        results = []

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
            prompt_template_name_or_path=self.prompt_template_name_or_path
        )

        # Check requirements
        assert "{test_case_prompt}" in self.prompt_template

    def generate(
        self,
        question: Question,
        # optional: context augmentation
        contexts_for_question: ContextsForOneExample | None = None
    ) -> str:

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

        # Get prompt part for test case
        test_case_prompt = self.generate_test_case_prompt(question=question)

        # Combine the prompt parts
        prompt = self.prompt_template.format(
            contexts_prompt=contexts_prompt,
            test_case_prompt=test_case_prompt
        )
        return prompt

    def generate_contexts_prompt(self, context_texts: list[str]) -> str:
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
        return question["question"]

    def generate_output_prompt(self, question: Question) -> str:
        answer = ", ".join([a["answer"] for a in question["answers"]])
        score = 0.69 # FIXME
        prompt = ""
        prompt += f"Answer: {answer}\n"
        prompt += f"Score: {score}\n"
        return prompt.rstrip()


class LLMQATrainer:

    def __init__(self, base_output_path: str):
        self.base_output_path = base_output_path
        self.paths = self.get_paths()

    def get_paths(self) -> dict[str,str]:
        paths = {}

        # configurations
        paths["path_snapshot"] = self.base_output_path

        # evaluation outputs
        paths["path_dev_gold"] = os.path.join(self.base_output_path, "dev.gold.json")
        paths["path_dev_pred"] = os.path.join(self.base_output_path, "dev.pred.json")
        paths["path_dev_eval"] = os.path.join(self.base_output_path, "dev.eval.json")
        paths["path_test_gold"] = os.path.join(self.base_output_path, "test.gold.json")
        paths["path_test_pred"] = os.path.join(self.base_output_path, "test.pred.json")
        paths["path_test_eval"] = os.path.join(self.base_output_path, "test.eval.json")

        return paths

    def setup_dataset(
        self,
        answerer: LLMQA,
        questions: list[Question],
        split: str
    ) -> None:
        # Cache the gold annotations for evaluation
        path_gold = self.paths[f"path_{split}_gold"]
        if not os.path.exists(path_gold):
            gold_questions = []
            for question in tqdm(questions, desc="dataset setup"):
                gold_question = copy.deepcopy(question)
                gold_questions.append(gold_question)
            utils.write_json(path_gold, gold_questions)
            logger.info(f"Saved the gold annotations for evaluation in {path_gold}")

    def save_answerer(self, answerer: LLMQA) -> None:
        answerer.save(path_snapshot=self.paths["path_snapshot"])

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
        utils.write_json(self.paths[f"path_{split}_pred"], results)

        # Save the prompt-response pairs in plain text
        with open(self.paths[f"path_{split}_pred"].replace(".json", ".txt"), "w") as f:
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
                pred_path=self.paths[f"path_{split}_pred"],
                gold_path=self.paths[f"path_{split}_gold"],
                exact_match=False
            )
        elif metric == "llm4eval":
             scores = evaluation.qa.llm4eval(
                pred_path=self.paths[f"path_{split}_pred"],
                gold_path=self.paths[f"path_{split}_gold"],
            )
        else:
            scores = evaluation.qa.accuracy(
                pred_path=self.paths[f"path_{split}_pred"],
                gold_path=self.paths[f"path_{split}_gold"],
                exact_match=False
            )

        if get_scores_only:
            return scores

        # Save the evaluation results
        utils.write_json(self.paths[f"path_{split}_eval"], scores)
        logger.info(utils.pretty_format_dict(scores))
        return scores
