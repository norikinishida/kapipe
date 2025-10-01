import argparse
import logging
import os

import torch
import transformers

import sys
sys.path.insert(0, "../..")
from kapipe.qa import LLMQA, LLMQATrainer
from kapipe import utils
from kapipe.utils import StopWatch
from kapipe.datatypes import Question, ContextsForOneExample

import shared_functions


def main(args):
    transformers.logging.set_verbosity_error()

    sw = StopWatch()
    sw.start("main")

    ##################
    # Arguments
    ##################

    # Method
    device = torch.device(f"cuda:{args.gpu}")
    method_name = args.method
    config_path = args.config_path
    config_name = args.config_name

    # Input Data
    path_dev_questions = args.dev_questions
    path_test_questions = args.test_questions
    path_dev_contexts = args.dev_contexts
    path_test_contexts = args.test_contexts

    # Output Path
    path_results_dir = args.results_dir
    prefix = args.prefix
    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()
        args.prefix = prefix

    # Action
    actiontype = args.actiontype

    assert method_name in ["llmqa"]
    assert actiontype in ["evaluate", "check_prompt"]

    ##################
    # Logging Setup
    ##################

    # Set base output path
    base_output_path = os.path.join(
        path_results_dir,
        "qa",
        "llmqa",
        config_name,
        prefix
    )
    utils.mkdir(base_output_path)

    # Set logger
    if actiontype == "evaluate":
        shared_functions.set_logger(
            os.path.join(base_output_path, "evaluation.log"),
            # overwrite=True
        )

    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))

    ##################
    # Data
    ##################

    # Load questions
    dev_questions = utils.read_json(path_dev_questions)
    test_questions = utils.read_json(path_test_questions)

    # Load contexts
    if path_dev_contexts is not None:
        dev_contexts = utils.read_json(path_dev_contexts)
    else:
        dev_contexts = None
    if path_test_contexts is not None:
        test_contexts = utils.read_json(path_test_contexts)
    else:
        test_contexts = None

    # Show statistics
    shared_functions.show_questions_statistics(dev_questions, "Development")
    shared_functions.show_questions_statistics(test_questions, "Test")

    ##################
    # Method
    ##################

    if method_name == "llmqa":
        # Initialize the trainer (evaluator)
        trainer = LLMQATrainer(base_output_path=base_output_path)

        # Load the configuration
        config = utils.get_hocon_config(
            config_path=config_path,
            config_name=config_name
        )

        # Initialize the answerer
        answerer = LLMQA(
            device=device,
            config=config,
        )

    ##################
    # Training, Evaluation
    ##################

    if method_name == "llmqa":

        if actiontype == "check_prompt":
            # Show prompts
            with torch.no_grad():
                path_out = os.path.join(base_output_path, "output.txt")
                with open(path_out, "w") as f:
                    for i, (question, contexts_for_question) in enumerate(zip(
                        dev_questions, dev_contexts
                    )):
                        question_key = question["question_key"]
                        logging.info(f"Processing {question_key}")
                        question = answerer.answer(
                            question=question,
                            contexts_for_question=contexts_for_question
                        )
                        f.write(f"--- QUESTION_KEY ({question_key}) ---\n\n")
                        f.write("Prompt:\n")
                        f.write(question["qa_prompt"] + "\n\n")
                        f.write("-----\n")
                        f.write("Generated Text:\n")
                        f.write(question["qa_generated_text"] + "\n\n")
                        f.write("-----\n")
                        f.write("Output:\n")
                        f.write(f"{question['answer']}\n")
                        f.write("-----\n")
                        f.flush()
                        if i > 5:
                            break
                return

        # Set up the datasets for evaluation
        trainer.setup_dataset(
            answerer=answerer,
            questions=dev_questions,
            split="dev"
        )
        trainer.setup_dataset(
            answerer=answerer,
            questions=test_questions,
            split="test"
        )

        # Save the configurations of the answerer
        trainer.save_answerer(answerer=answerer)

        if actiontype == "evaluate":
            # Evaluate the answerer on the datasets
            trainer.evaluate(
                answerer=answerer,
                questions=dev_questions,
                demonstrations=None,
                contexts=dev_contexts,
                split="dev",
                metric="recall"
            )
            trainer.evaluate(
                answerer=answerer,
                questions=test_questions,
                demonstrations=None,
                contexts=test_contexts,
                split="test",
                metric="recall"
            )

    ##################
    # Closing
    ##################

    logging.info("Done.")
    sw.stop("main")
    logging.info("Time: %f min." % sw.get_time("main", minute=True))

    return prefix


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO
    )

    parser = argparse.ArgumentParser()

    # Method
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=True)

    # Input Data
    parser.add_argument("--dev_questions", type=str, required=True)
    parser.add_argument("--test_questions", type=str, required=True)
    parser.add_argument("--dev_contexts", type=str, default=None)
    parser.add_argument("--test_contexts", type=str, default=None)

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    # Action
    parser.add_argument("--actiontype", type=str, required=True)

    args = parser.parse_args()

    main(args=args)


