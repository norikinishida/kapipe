import argparse
import logging
import os
import sys

from tqdm import tqdm
import transformers

from kapipe import evaluation
from kapipe import utils
from kapipe.llms import HuggingFaceLLM, OpenAILLM
from kapipe.qa import LLMQA
from kapipe.utils import StopWatch


def main(args):
    transformers.logging.set_verbosity_error()

    sw = StopWatch()
    sw.start("main")

    ##################
    # Arguments
    ##################

    # Method
    method_name = args.method
    config_path = args.config_path
    config_name = args.config_name

    # Input Data
    input_questions_path = args.input_questions
    input_contexts_path = args.input_contexts

    # Output Path
    results_dir = args.results_dir
    prefix = args.prefix
    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()
        args.prefix = prefix

    # Evaluation
    do_evaluation = args.do_evaluation
    gold_questions_path = args.gold

    # Batch API
    batch_mode: str | None = args.batch_mode

    ##################
    # Logging Setup
    ##################

    # Set base output path
    base_output_path = os.path.join(
        results_dir,
        "qa",
        method_name, 
        config_name,
        prefix
    )
    utils.mkdir(base_output_path)

    # Extract the base filename
    base_filename = os.path.splitext(os.path.basename(input_questions_path))[0]

    # Set logger
    if batch_mode is None:
        set_logger(
            os.path.join(base_output_path, f"{base_filename}.qa.log"),
            # overwrite=True
        )
    else:
        set_logger(
            os.path.join(base_output_path, f"{base_filename}.qa.{batch_mode}.log"),
            # overwrite=True
        )

    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))

    ##################
    # Data
    ##################

    # Load questions
    questions = utils.read_json(input_questions_path)

    # Load contexts
    if input_contexts_path is not None:
        contexts = utils.read_json(input_contexts_path)
    else:
        contexts = [None] * len(questions)

    ##################
    # Method Instantiation
    ##################

    # Load the experiment configuration
    config = utils.get_hocon_config(config_path=config_path, config_name=config_name)

    # Save the experiment configuration to the output path
    utils.write_json(os.path.join(base_output_path, "config.json"), config)

    # Instantiate the QA component
    if method_name == "llm_qa":
        # Instantiate the LLM wrapper
        if config["provider"] == "openai":
            model = OpenAILLM(
                model_name=config["model_name"],
                max_new_tokens=config["max_new_tokens"],
            )
        elif config["provider"] == "hf":
            model = HuggingFaceLLM(
                model_name=config["model_name"],
                max_new_tokens=config["max_new_tokens"],
                quantization_bits=config["quantization_bits"],
            )
        else:
            raise ValueError(f"Unknown LLM provider: {config['provider']}")
        logging.info("Instantiated the LLM model: %s" % repr(model))

        # Instantiate the LLM-based QA component
        answerer = LLMQA(
            model=model,
            prompt_template_name_or_path=config["prompt_template_name_or_path"],
            n_contexts=config["n_contexts"],
        )
        
    ##################
    # Method Execution
    ##################

    logging.info(
        f"Applying the QA component to {len(questions)} questions (+ contexts) "
        f"in {input_questions_path} ({input_contexts_path}) ..."
    )

    # Apply the QA component to the questions
    if batch_mode is None:
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

    elif batch_mode == "submit":
        # Submit prompts
        batch_ids: list[str] = answerer.submit_batch(
            questions=questions,
            contexts=contexts
        )
        utils.write_json(
            os.path.join(
                base_output_path,
                f"{base_filename}.batch_ids.json",
            ),
            batch_ids,
        )
        logging.info(f"Submitted batches: {batch_ids}")

    elif batch_mode == "fetch":
        # Fetch and process the responses
        batch_ids: list[str] = utils.read_json(
            os.path.join(
                base_output_path,
                f"{base_filename}.batch_ids.json",
            )
        )
        result_questions = answerer.fetch_and_process_batch(
            questions=questions,
            contexts=contexts,
            batch_ids=batch_ids,
        )

    else:
        raise ValueError(
            f"Invalid batch_mode: {batch_mode}. "
            "Expected None, 'submit', or 'fetch'."
        )

    if batch_mode != "submit":
        # Save the QA results
        output_questions_path = os.path.join(
            base_output_path,
            f"{base_filename}.pred.json"
        )
        utils.write_json(output_questions_path, result_questions)
        logging.info(f"Saved the prediction results to {output_questions_path}")

        # Save the prompt-response pairs in plain text
        if "qa_prompt" in result_questions[0] and "qa_generated_text" in result_questions[0]:
            output_text_path = os.path.join(
                base_output_path,
                f"{base_filename}.prompt_and_response.txt"
            )
            with open(output_text_path, "w") as f:
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
        # Evaluation
        ##################

        if do_evaluation:
            # Validate that the gold questions path is provided
            if gold_questions_path is None:
                raise ValueError("--gold is required when --do_evaluation is set")

            # Evaluate the prediction results
            scores = evaluation.qa.accuracy(
                pred_path=output_questions_path,
                gold_path=gold_questions_path,
                exact_match=False,
            )
            scores.update(
                evaluation.qa.token_level_f1(
                    pred_path=output_questions_path,
                    gold_path=gold_questions_path
                )
            )
            scores.update(
                evaluation.qa.recall(
                    pred_path=output_questions_path,
                    gold_path=gold_questions_path,
                    exact_match=False,
                )
            )
            logging.info(utils.pretty_format_dict(scores))

            # Save the evaluation result
            output_evaluation_path = os.path.join(
                base_output_path,
                f"{base_filename}.eval.json"
            )
            utils.write_json(output_evaluation_path, scores)
            logging.info(f"Saved the evaluation results to {output_evaluation_path}")

    ##################
    # Closing
    ##################

    logging.info("Done.")
    sw.stop("main")
    logging.info("Time: %f min." % sw.get_time("main", minute=True))

    return prefix


def set_logger(filename: str, overwrite: bool = False) -> None:
    if os.path.exists(filename) and not overwrite:
        logging.info("%s already exists." % filename)
        do_remove = input("Delete the existing log file? [y/n]: ")
        if (not do_remove.lower().startswith("y")) and (not len(do_remove) == 0):
            logging.info("Done.")
            sys.exit(0)

    root_logger = logging.getLogger()
    handler = logging.FileHandler(filename, "w")
    root_logger.addHandler(handler)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO
    )
    logging.getLogger("httpx").addFilter(
        lambda r: "huggingface.co" not in r.getMessage()
    )

    parser = argparse.ArgumentParser()

    # Method
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=True)

    # Input Data
    parser.add_argument("--input_questions", type=str, required=True)
    parser.add_argument("--input_contexts", type=str, default=None)

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    # Evaluation
    parser.add_argument("--do_evaluation", action="store_true")
    parser.add_argument("--gold", type=str, default=None)

    # Batch API
    parser.add_argument(
        "--batch_mode",
        type=str,
        default=None,
        choices=["submit", "fetch"],
    )

    args = parser.parse_args()

    main(args=args)


