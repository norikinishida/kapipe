import argparse
import logging
import os

import transformers
from tqdm import tqdm

import sys
sys.path.insert(0, "../..")
from kapipe.qa import QA
from kapipe import utils
from kapipe.utils import StopWatch

import shared_functions


def main(args):
    transformers.logging.set_verbosity_error()

    sw = StopWatch()
    sw.start("main")

    ##################
    # Arguments
    ##################

    # Method
    identifier = args.identifier
    gpu = args.gpu

    # Input Data
    path_input_questions = args.input_questions
    path_input_contexts = args.input_contexts

    # Output Path
    path_results_dir = args.results_dir
    prefix = args.prefix
    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()
        args.prefix = prefix

    ##################
    # Logging Setup
    ##################

    # Set base output path
    base_output_path = os.path.join(
        path_results_dir,
        "qa",
        "qa",
        identifier,
        prefix
    )
    utils.mkdir(base_output_path)

    base_filename = os.path.splitext(os.path.basename(path_input_questions))[0]

    # Set logger
    shared_functions.set_logger(
        os.path.join(base_output_path, f"{base_filename}.answering.log"),
        # overwrite=True
    )

    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))

    ##################
    # Data
    ##################

    # Load questions
    questions = utils.read_json(path_input_questions)

    # Load contexts
    if path_input_contexts is not None:
        contexts = utils.read_json(path_input_contexts)
    else:
        contexts = None

    ##################
    # Method
    ##################

    # Initialilze the question answerer
    answerer = QA(identifier=identifier, gpu=gpu)

    ##################
    # QA
    ##################

    logging.info(f"Applying the QA component to {len(questions)} questions (+ contexts) in {path_input_questions} ({path_input_contexts}) ...")

    # Create the full output path
    path_output_questions = os.path.join(base_output_path, f"{base_filename}.pred.json")

    # Apply the question answerer to the questions
    result_questions = []
    for question, contexts_for_q in tqdm(
        zip(questions, contexts),
        total=len(questions)
    ):
        result_question = answerer.answer(
            question=question,
            contexts_for_question=contexts_for_q
        )
        result_questions.append(result_question)

    # Save the results
    utils.write_json(path_output_questions, result_questions)
    logging.info(f"Saved the prediction results to {path_output_questions}")

    # Save the prompt-response pairs in plain text
    if "qa_prompt" in result_questions[0] and "qa_generated_text" in result_questions[0]:
        path_output_text = os.path.join(base_output_path, "prompt_and_response.txt")
        with open(path_output_text, "w") as f:
            for q in result_questions:
                question_key = q["question_key"]
                prompt = q["qa_prompt"]
                generated_text = q["qa_generated_text"]
                f.write("-------------------------------------\n\n")
                f.write(f"QUESTION_KEY: {question_key}\n\n")
                f.write("PROMPT:\n")
                f.write(prompt + "\n\n")
                f.write("GENERATED TEXT:\n")
                f.write(generated_text + "\n\n")
                f.flush()

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
    parser.add_argument("--identifier", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)

    # Input Data
    parser.add_argument("--input_questions", type=str, required=True)
    parser.add_argument("--input_contexts", type=str, default=None)

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    args = parser.parse_args()

    main(args=args)


