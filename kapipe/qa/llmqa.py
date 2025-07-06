from __future__ import annotations
 
import copy
import logging
import os
import re
from typing import Any

import torch
from tqdm import tqdm

from ..models import LLM, OpenAILLM
from .. import evaluation
from .. import utils
from ..datatypes import (
    Config,
    Question,
    DemonstrationsForOneExample,
    ContextsForOneExample
)


logger = logging.getLogger(__name__)


class LLMQA:

    def __init__(
        self,
        # General
        config: Config | str,
        device: int = 0,
        # Optional: few-shot setting
        path_demonstration_pool: str | None = None,
        # Misc.
        model: LLM | OpenAILLM | None = None,
        verbose: bool = True
    ):
        self.verbose = verbose
        if self.verbose:
            logger.info(">>>>>>>>>> LLMQA Initialization >>>>>>>>>>")

        self.device = device

        ######
        # Config
        ######

        if isinstance(config, str):
            tmp = config
            config = utils.get_hocon_config(config_path=config, config_name=None)
            if self.verbose:
                logger.info(f"Loaded configuration from {tmp}")
        self.config = config
        if self.verbose:
            logger.info(utils.pretty_format_dict(self.config))

        ######
        # Prompt Processor
        ######

        self.prompt_processor = PromptProcessor(
            prompt_template_name_or_path=config["prompt_template_name_or_path"],
            n_contexts=config["n_contexts"],
            path_demonstration_pool=path_demonstration_pool,
            n_demonstrations=config["n_demonstrations"]
        )

        ######
        # Model
        ######

        self.model_name = config["model_name"]
        assert self.model_name in ["llm", "openai"]

        if model is not None:
            self.model = model
            logger.info("LLM is provided")
        elif self.model_name == "llm":
            self.model = LLM(
                device=device,
                # Model
                llm_name_or_path=config["llm_name_or_path"],
                max_seg_len=config["max_seg_len"],
                quantization_bits=config["quantization_bits"],
                # Generation
                max_new_tokens=config["max_new_tokens"],
                beam_size=config["beam_size"],
                do_sample=config["do_sample"],
                num_return_sequences=config["num_return_sequences"],
                stop_list=config["stop_list"],
                clean_up_tokenization_spaces=config["clean_up_tokenization_spaces"],
            )
        else:
            self.model = OpenAILLM(
                # Model
                openai_model_name=config["openai_model_name"],
                # Generation
                max_new_tokens=config["max_new_tokens"]
            )
        # self.model.llm.to(self.model.device)

        self.map_reduce_generation = config["map_reduce_generation"]
        if self.map_reduce_generation:
            self.n_intermediate_answers = config["n_intermediate_answers"]

        if self.verbose:
            logger.info("<<<<<<<<<< LLMQA Initialization <<<<<<<<<<")

    def run(
        self,
        question: Question,
        # optional: few-shot setting
        demonstrations_for_question: DemonstrationsForOneExample | None = None,
        # optional: context augmentation
        contexts_for_question: ContextsForOneExample | None = None
    ) -> Question:
        with torch.no_grad():
            if self.model_name == "llm":
                # Switch to inference mode
                self.model.llm.eval()

            if self.map_reduce_generation:
                ###
                # Step 1. map generation
                ###

                intermediate_answers = []
                assert contexts_for_question is not None
                for passage in contexts_for_question["contexts"]:
                    # Prepare a new-formed context
                    new_contexts_for_question: ContextsForOneExample = {
                        "question_key": contexts_for_question["question_key"],
                        "contexts": [passage]
                    }

                    # Generate a prompt
                    prompt = self.prompt_processor.generate(
                        question=question,
                        demonstrations_for_question=demonstrations_for_question,
                        contexts_for_question=new_contexts_for_question
                    )

                    if self.model_name == "llm":
                        # Preprocess
                        preprocessed_data = self.model.preprocess(prompt=prompt)

                        # Tensorize
                        model_input = self.model.tensorize(
                            preprocessed_data=preprocessed_data,
                            compute_loss=False
                        )

                        # Forward
                        generated_text = self.model.generate(**model_input)[0] # str
                        generated_text = self.model.remove_prompt_from_generated_text(
                            generated_text=generated_text
                        )
                    else:
                        generated_text = self.model.generate(prompt) # str

                    # Structurize (post-processing)
                    answer, helpfulness_score = self.structurize(
                        question=question,
                        generated_text=generated_text
                    )
                    intermediate_answers.append({
                        "intermediate_answer": answer,
                        "helpfulness_score": helpfulness_score
                    })

                ###
                # Step 2. reduce generation
                ###

                # Select high-confident intermediate answers
                intermediate_answers = sorted(
                    intermediate_answers,
                    key=lambda x: -x["helpfulness_score"]
                )
                intermediate_answers = intermediate_answers[:self.n_intermediate_answers]

                # Prepare a new-formed context
                new_contexts_for_question: ContextsForOneExample = {
                    "question_key": contexts_for_question["question_key"],
                    "contexts": [
                        {"text": f"{x['intermediate_answer']} (score: {x['helpfulness_score']})"}
                        for x in intermediate_answers
                    ]
                }

                # Generate a prompt
                prompt = self.prompt_processor.generate(
                    question=question,
                    demonstrations_for_question=demonstrations_for_question,
                    contexts_for_question=new_contexts_for_question
                )

                if self.model_name == "llm":
                    # Preprocess
                    preprocessed_data = self.model.preprocess(prompt=prompt)

                    # Tensorize
                    model_input = self.model.tensorize(
                        preprocessed_data=preprocessed_data,
                        compute_loss=False
                    )

                    # Forward
                    generated_text = self.model.generate(**model_input)[0] # str
                    generated_text = self.model.remove_prompt_from_generated_text(
                        generated_text=generated_text
                    )
                else:
                    generated_text = self.model.generate(prompt) # str

                # Structurize (post-processing)
                answer, helpfulness_score = self.structurize(
                    question=question,
                    generated_text=generated_text
                )
            else:
                # Generate a prompt
                prompt = self.prompt_processor.generate(
                    question=question,
                    demonstrations_for_question=demonstrations_for_question,
                    contexts_for_question=contexts_for_question
                )

                if self.model_name == "llm":
                    # Preprocess
                    preprocessed_data = self.model.preprocess(prompt=prompt)

                    # Tensorize
                    model_input = self.model.tensorize(
                        preprocessed_data=preprocessed_data,
                        compute_loss=False
                    )

                    # Forward
                    generated_text = self.model.generate(**model_input)[0] # str
                    generated_text = self.model.remove_prompt_from_generated_text(
                        generated_text=generated_text
                    )
                else:
                    generated_text = self.model.generate(prompt) # str

                # Structurize (post-processing)
                answer, helpfulness_score = self.structurize(
                    question=question,
                    generated_text=generated_text
                )

            # Integrate
            result_question = copy.deepcopy(question)
            result_question["output_answer"] = answer
            result_question["helpfulness_score"] = helpfulness_score
            result_question["qa_prompt"] = (
                preprocessed_data["prompt"] if self.model_name == "llm" else prompt
            )
            result_question["qa_generated_text"] = generated_text
            if self.map_reduce_generation:
                result_question["intermediate_answers"] = intermediate_answers
            return result_question

    def structurize(self, question: Question, generated_text: str) -> tuple[str, float]:
        question_key = question["question_key"]

        # Parse each generated line
        generated_lines = generated_text.split("\n")
        answer = generated_text
        score = 0.0
        for generated_line in generated_lines:
            generated_text = generated_line.strip()
            if generated_line == "":
                continue
            
            # Parse the generated_line
            if generated_line.startswith("Answer:"):
                answer = generated_line[len("Answer:"):].strip()
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
        return answer, score
 
    def batch_run(
        self,
        questions: list[Question],
        # optional: few-shot setting
        demonstrations: list[DemonstrationsForOneExample] | None = None,
        # optional: context augmentation
        contexts: list[ContextsForOneExample] | None = None
    ) -> list[Question]:
        result_questions = []

        if demonstrations is None:
            demonstrations = [None] * len(questions)

        if contexts is None:
            contexts = [None] * len(questions)

        for question, demos_for_q, contexts_for_q in tqdm(
            zip(questions, demonstrations, contexts),
            total=len(questions),
            desc="answering steps"
        ):
            result_question = self.run(
                question=question,
                demonstrations_for_question=demos_for_q,
                contexts_for_question=contexts_for_q
            )
            result_questions.append(result_question)
        return result_questions


class PromptProcessor:
    
    def __init__(
        self,
        prompt_template_name_or_path: str,
        # optional: few-shot setting
        n_contexts: int | None = None,
        path_demonstration_pool: str | None = None,
        n_demonstrations: int | None = None
    ): 
        self.prompt_template_name_or_path = prompt_template_name_or_path
        self.path_demonstration_pool = path_demonstration_pool
        self.n_contexts = n_contexts
        self.n_demonstrations = n_demonstrations

        if self.path_demonstration_pool is not None:
            assert self.n_demonstrations is not None

        #####
        # Prompt template
        #####

        self.prompt_template = utils.read_prompt_template(prompt_template_name_or_path=self.prompt_template_name_or_path)

        # Check requirements
        if self.path_demonstration_pool is not None:
            assert "{demonstrations_prompt}" in self.prompt_template
        assert "{task_prompt}" in self.prompt_template

        #####
        # Demonstration pool
        #####

        if self.path_demonstration_pool is not None:
            self.demonstration_pool: dict[str, Question] = {
                demo_doc["question_key"]: demo_doc
                for demo_doc in utils.read_json(self.path_demonstration_pool)
            }
 
    def generate(
        self,
        question: Question,
        # optional: few-shot setting
        demonstrations_for_question: DemonstrationsForOneExample | None = None,
        # optional: context augmentation
        contexts_for_question: ContextsForOneExample | None = None
    ) -> str:
        if demonstrations_for_question is not None:
            # Prepare demonstrations
            demonstration_questions = [
                self.demonstration_pool[demo_key_info["question_key"]]
                for demo_key_info in demonstrations_for_question["demonstrations"][:self.n_demonstrations]
            ]
            # Get prompt part for demonstrations
            demonstrations_prompt = self.generate_demonstrations_prompt(
                demonstration_questions=demonstration_questions
            )
        else:
            demonstrations_prompt = ""

        if contexts_for_question is not None:
            # Prepare contexts
            if self.n_contexts is not None:
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

        # Get prompt part for task
        task_prompt = self.generate_task_prompt(question=question)

        # Combine the prompt parts
        prompt = self.prompt_template.format(
            demonstrations_prompt=demonstrations_prompt,
            contexts_prompt=contexts_prompt,
            task_prompt=task_prompt
        )
        return prompt

    def generate_demonstrations_prompt(self, demonstration_questions: list[Question]) -> str:
        prompt = ""
        n_demos = len(demonstration_questions)
        for demo_i, demo in enumerate(demonstration_questions):
            prompt += f"Example {demo_i+1}:\n"
            prompt += f"Question: {self.generate_input_question_prompt(question=demo)}\n"
            prompt += f"Answer: {self.generate_output_prompt(question=demo)}\n"
            if demo_i < n_demos - 1:
                prompt += "\n"
        return prompt.rstrip()

    def generate_contexts_prompt(self, context_texts: list[str]) -> str:
        n_contexts = len(context_texts)

        if n_contexts == 0:
            return ""

        prompt = ""
        for c_i, content_text in enumerate(context_texts):
            prompt += f"[{c_i+1}] {content_text.strip()} \n"
            if c_i < n_contexts - 1:
                prompt += "\n"
        return prompt.rstrip()

    def generate_task_prompt(self, question: Question) -> str:
        return f"Question: {self.generate_input_question_prompt(question)}".rstrip()
                   
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

        # Path to config file
        paths["path_config"] = os.path.join(self.base_output_path, "config")

        # Paths to validation outputs and scores
        paths["path_dev_gold"] = os.path.join(self.base_output_path, "dev.gold.json")
        paths["path_dev_pred"] = os.path.join(self.base_output_path, "dev.pred.json")
        paths["path_dev_eval"] = os.path.join(self.base_output_path, "dev.eval.json")

        # Paths to evaluation outputs and scores
        paths["path_test_gold"] = os.path.join(self.base_output_path, "test.gold.json")
        paths["path_test_pred"] = os.path.join(self.base_output_path, "test.pred.json")
        paths["path_test_eval"] = os.path.join(self.base_output_path, "test.eval.json")

        return paths

    def setup_dataset(self, answerer: LLMQA, questions: list[Question], split: str) -> None:
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
        # Since we do not finetune the model, we save the configuration
        # Save the config (only once)
        # utils.dump_hocon_config(self.paths["path_config"], answerer.config)
        utils.write_json(self.paths["path_config"], answerer.config)
        logger.info("Saved config file to %s" % self.paths["path_config"])

    def evaluate(
        self,
        answerer: LLMQA,
        questions: list[Question],
        demonstrations: list[DemonstrationsForOneExample] | None,
        contexts: list[ContextsForOneExample] | None,
        split: str,
        #
        metric: str = "accuracy",
        get_scores_only: bool = False
    ) -> dict[str, Any]:
        # Predict answers for the given questions,
        # optionally based on the demonstrations and contexts
        result_questions = answerer.batch_run(
            questions=questions,
            demonstrations=demonstrations,
            contexts=contexts
        )

        # Save the predicted answers
        utils.write_json(self.paths[f"path_{split}_pred"], result_questions)

        with open(self.paths[f"path_{split}_pred"].replace(".json", ".txt"), "w") as f:
            for result_question in result_questions:
                question_key = result_question["question_key"]
                prompt = result_question["qa_prompt"]
                generated_text = result_question["qa_generated_text"]
                f.write(f"--- QUESTION_KEY ({question_key}) ---\n\n")
                f.write(prompt + "\n\n")
                f.write(generated_text + "\n\n")
                f.flush()

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
