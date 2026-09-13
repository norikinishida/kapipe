import argparse
import logging
import os
from typing import Any
import sys

import torch
import transformers

from kapipe import evaluation
from kapipe import utils
from kapipe.utils import StopWatch

from kapipe.pipelines import RAGPipeline

from kapipe.llms import BaseLLM, HuggingFaceLLM, OpenAILLM
from kapipe.passage_retrieval import (
    BasePassageRetriever,
    BM25,
    Contriever,
    Qwen3Embedding,
)
from kapipe.qa import (
    BaseQA,
    LLMQA,
)


def main(args):
    torch.autograd.set_detect_anomaly(True)
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
    input_passages_path = args.input_passages
    input_questions_path = args.input_questions

    # Output Path
    results_dir = args.results_dir
    prefix = args.prefix
    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()
        args.prefix = prefix

    # Action
    actiontype = args.actiontype

    # Evaluation
    do_evaluation = args.do_evaluation
    gold_questions_path = args.gold_answers
    gold_contexts_path = args.gold_contexts

    ##################
    # Logging Setup
    ##################

    # Set base output path
    base_output_path = os.path.join(
        results_dir,
        "rag_pipeline",
        method_name,
        config_name,
        prefix
    )
    utils.mkdir(base_output_path)

    # Set the base filename for query processing
    base_filename: str | None = None
    if actiontype == "inference":
        if input_questions_path is None:
            raise ValueError(
                f"--input_questions is required for {actiontype}"
            )
        base_filename = os.path.splitext(
            os.path.basename(input_questions_path)
        )[0]

    # Index will be saved to `index_dir``
    index_dir = os.path.join(base_output_path, "indexes")
    utils.mkdir(index_dir)

    # Set logger
    if actiontype != "inference":
        set_logger(
            os.path.join(index_dir, "indexing.log"),
            # overwrite=True
        )
    else:
        set_logger(
            os.path.join(
                base_output_path,
                base_filename + ".inference.log"
            ),
            # overwrite=True
        )

    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))
    logging.info(f"index dir: {index_dir}")

    ##################
    # Data
    ##################

    # Data are loaded in the corresponding actiontype section below

    ##################
    # Method Instantiation
    ##################

    # Load the experiment configuration
    config = utils.get_hocon_config(
        config_path=config_path,
        config_name=config_name
    )

    # Save the experiment configuration to the output path
    utils.write_json(os.path.join(base_output_path, "config.json"), config)

    # Instantiate the Passage Retrieval component
    passage_retrieval = instantiate_passage_retrieval_component(
        passage_retrieval_config=config["passage_retrieval"],
    )

    # Instantiate the QA component
    qa = instantiate_qa_component(
        qa_config=config["qa"],
    )

   # Instantiate the RAG pipeline
    rag = RAGPipeline(
        passage_retrieval=passage_retrieval,
        qa=qa,
    )

    ##################
    # Method Execution
    ##################

    if actiontype != "inference":
        # Validate that the input passages path is provided for indexing
        if input_passages_path is None:
            raise ValueError("--input_passages is required for indexing")

        # Load passages for indexing
        passages: list[dict[str, Any]] = utils.read_jsonl(input_passages_path)

        # Set component-specific arguments
        if config["passage_retrieval"]["method_name"] == "bm25":
            passage_retrieval_indexing_kwargs: dict[str, Any] = {}
        else:
            passage_retrieval_indexing_kwargs = {
                "batch_size": config["passage_retrieval"][
                    "indexing_batch_size"
                ],
            }

        # Build the index
        rag.make_index(
            passages=passages,
            index_dir=index_dir,
            **passage_retrieval_indexing_kwargs,
        )

    else:
        # Validate that the input questions path is provided
        if input_questions_path is None:
            raise ValueError("--input_questions is required for inference")

        # Load questions
        questions: list[dict[str, Any]] = utils.read_json(input_questions_path)

        logging.info(
            f"Applying the RAG pipeline to {len(questions)} questions "
            f"in {input_questions_path} ..."
        )

        # Load the index
        rag.load_index(index_dir=index_dir)

        # Run all inference components for every question
        result_questions: list[dict[str, Any]] = rag.infer(
            questions=questions,
            top_k=config["passage_retrieval"]["top_k"],
        )

        # Save the results
        output_questions_path = os.path.join(
            base_output_path,
            f"{base_filename}.pred.json",
        )
        utils.write_json(output_questions_path, result_questions)
        logging.info(f"Saved the prediction results to {output_questions_path}")

        # Save the prompts, raw responses, parsed answers, and optional gold answers
        output_prompt_and_responses_path: str = os.path.join(
            base_output_path,
            f"{base_filename}.prompt_and_responses.txt",
        )
        with open(
            output_prompt_and_responses_path,
            "w",
            encoding="utf-8",
        ) as fout:
            # Write one human-readable block for each question
            for result_question in result_questions:
                fout.write("=" * 80 + "\n\n")

                fout.write("QUESTION KEY:\n")
                fout.write(result_question["question_key"] + "\n\n")

                fout.write("PROMPT:\n")
                fout.write(result_question["qa_prompt"].rstrip() + "\n\n")

                fout.write("GENERATED TEXT:\n")
                fout.write(
                    result_question["qa_generated_text"].rstrip() + "\n\n"
                )

                fout.write("PARSED ANSWER:\n")
                fout.write(result_question["output_answer"].rstrip() + "\n\n")

                # Write gold answers only when they are included in the input question
                if "answers" in result_question:
                    fout.write("GOLD ANSWERS:\n")
                    for answer in result_question["answers"]:
                        fout.write(f"- {answer['answer']}\n")
                    fout.write("\n")

        logging.info(
            "Saved the prompts and responses to "
            f"{output_prompt_and_responses_path}"
        )

        ##################
        # Evaluation
        ##################

        if do_evaluation:
            # Validate that the gold questions and contexts paths are provided
            if gold_questions_path is None:
                raise ValueError(
                    "--gold_answers is required when --do_evaluation is set"
                )
            if gold_contexts_path is None:
                raise ValueError(
                    "--gold_contexts is required when --do_evaluation is set"
                )

            # Evaluate the prediction results
            qa_scores = evaluation.qa.accuracy(
                pred_path=output_questions_path,
                gold_path=gold_questions_path,
                exact_match=False,
            )
            qa_scores.update(
                evaluation.qa.token_level_f1(
                    pred_path=output_questions_path,
                    gold_path=gold_questions_path
                )
            )
            qa_scores.update(
                evaluation.qa.recall(
                    pred_path=output_questions_path,
                    gold_path=gold_questions_path,
                    exact_match=False,
                )
            )
            ret_scores = evaluation.passage_retrieval.precision_recall_at_k(
                pred_path=output_questions_path,
                gold_path=gold_contexts_path,
            )
            ret_scores.update(
                evaluation.passage_retrieval.ndcg_at_k(
                    pred_path=output_questions_path,
                    gold_path=gold_contexts_path,
                )
            )
            scores = {
                "qa": qa_scores,
                "passage_retrieval": ret_scores,
            }
            logging.info(utils.pretty_format_dict(scores))

            # Save the evaluation results
            output_evaluation_path = os.path.join(
                base_output_path,
                f"{base_filename}.eval.json",
            )
            utils.write_json(output_evaluation_path, scores)

            # Log the evaluation results
            logging.info(utils.pretty_format_dict(scores))
            logging.info(
                f"Saved the evaluation results to {output_evaluation_path}"
            )

    ##################
    # Closing
    ##################

    logging.info("Done.")
    sw.stop("main")
    logging.info("Time: %f min." % sw.get_time("main", minute=True))


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


def instantiate_passage_retrieval_component(
    passage_retrieval_config: dict[str, Any],
) -> BasePassageRetriever:

    # Instantiate the BM25-based Passage Retrieval component
    if passage_retrieval_config["method_name"] == "bm25":
        passage_retrieval = BM25(
            tokenizer=lambda text: text.lower().split(),
            k1=passage_retrieval_config["k1"],
            b=passage_retrieval_config["b"],
        )

    # Instantiate the Contriever-based Passage Retrieval component
    elif passage_retrieval_config["method_name"] == "contriever":
        passage_retrieval = Contriever(
            model_name=passage_retrieval_config["model_name"],
            max_passage_length=passage_retrieval_config["max_passage_length"],
            pooling_method=passage_retrieval_config["pooling_method"],
            normalize=passage_retrieval_config["normalize"],
            metric=passage_retrieval_config["metric"],
        )

    # Instantiate the Qwen3-Embedding-based Passage Retrieval component
    elif passage_retrieval_config["method_name"] == "qwen3_embedding":
        passage_retrieval = Qwen3Embedding(
            model_name=passage_retrieval_config["model_name"],
            max_passage_length=passage_retrieval_config["max_passage_length"],
            normalize=passage_retrieval_config["normalize"],
            metric=passage_retrieval_config["metric"],
            query_instruction=passage_retrieval_config["query_instruction"],
        )

    else:
        raise ValueError(
            f"Unknown passage retrieval method: {passage_retrieval_config['method_name']}"
        )

    return passage_retrieval


def instantiate_qa_component(
    qa_config: dict[str, Any],
) -> BaseQA:
    
    # Instantiate the LLM-based QA component
    if qa_config["method_name"] == "llm_qa":
        llm = instantiate_llm(config=qa_config)

        qa = LLMQA(
            model=llm,
            prompt_template_name_or_path=qa_config["prompt_template_name_or_path"],
            n_contexts=qa_config["n_contexts"],
        )

    else:
        raise ValueError(f"Unknown QA method: {qa_config['method_name']}")

    return qa

def instantiate_llm(
    config: dict[str, Any],
) -> BaseLLM:

    # Instantiate the LLM wrapper
    if config["llm_provider"] == "openai":
        llm = OpenAILLM(
            model_name=config["llm_model_name"],
            max_new_tokens=config["llm_max_new_tokens"],
        )
    elif config["llm_provider"] == "hf":
        llm = HuggingFaceLLM(
            model_name=config["llm_model_name"],
            max_new_tokens=config["llm_max_new_tokens"],
            quantization_bits=config["llm_quantization_bits"],
        )
    else:
        raise ValueError(f"Unknown LLM provider: {config['llm_provider']}")

    return llm


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
    parser.add_argument("--input_passages", type=str, default=None)
    parser.add_argument("--input_questions", type=str, default=None)

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    # Action
    parser.add_argument(
        "--actiontype",
        type=str,
        required=True,
        choices=[
            "indexing",
            "inference",
        ],
    )

    # Evaluation
    parser.add_argument("--do_evaluation", action="store_true")
    parser.add_argument("--gold_answers", type=str, default=None)
    parser.add_argument("--gold_contexts", type=str, default=None)

    args = parser.parse_args()

    main(args)
