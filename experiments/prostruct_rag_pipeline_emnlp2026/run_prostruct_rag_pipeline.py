import argparse
import logging
import os
import sys
from typing import Any

import torch
from tqdm import tqdm
import transformers

from kapipe import evaluation
from kapipe import utils
from kapipe.datatypes import Passage, Question
from kapipe.utils import StopWatch

from kapipe.pipelines import ProStructRAGPipeline

from kapipe.llms import BaseLLM, HuggingFaceLLM, OpenAILLM
from kapipe.proposition_extraction import (
    BasePropositionExtractor,
    LLMPropositionExtractor,
)
from kapipe.proposition_relation_extraction import (
    BasePropositionRelationExtractor,
    LLMPropositionRelationExtractor,
)
from kapipe.proposition_relation_refinement import (
    BasePropositionRelationRefiner,
    LLMPropositionRelationRefiner,
)
from kapipe.passage_graph_construction import (
    BasePassageGraphConstructor,
    PassageGraphConstructor,
)
from kapipe.passage_retrieval import (
    BasePassageRetriever,
    BM25,
    Contriever,
    Qwen3Embedding,
)
from kapipe.graph_retrieval import (
    BaseGraphRetriever,
    GraphRetriever,
)
from kapipe.context_formatting import (
    BaseContextFormatter,
    GraphVerbalizer,
)
from kapipe.qa import (
    BaseQA,
    LLMQA,
)


def main(args: argparse.Namespace) -> None:
    torch.autograd.set_detect_anomaly(True)
    transformers.logging.set_verbosity_error()

    sw: StopWatch = StopWatch()
    sw.start("main")

    ##################
    # Arguments
    ##################

    # Method
    method_name: str = args.method
    config_path: str = args.config_path
    config_name: str = args.config_name

    # Input Data
    input_passages_path: str | None = args.input_passages
    input_questions_path: str | None = args.input_questions

    # Output Path
    results_dir: str = args.results_dir
    prefix: str | None = args.prefix
    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()
        args.prefix = prefix

    # Action
    actiontype: str = args.actiontype

    # Evaluation
    do_evaluation: bool = args.do_evaluation
    gold_questions_path: str | None = args.gold

    ##################
    # Logging Setup
    ##################

    # Set base output path
    base_output_path: str = os.path.join(
        results_dir,
        "prostruct_rag_pipeline",
        method_name,
        config_name,
        prefix,
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

    # Index will be saved to `index_dir`
    index_dir: str = os.path.join(base_output_path, "indexes")
    utils.mkdir(index_dir)

    # Set logger
    if base_filename is None:
        set_logger(
            os.path.join(base_output_path, f"{actiontype}.log"),
            # overwrite=True
        )
    else:
        set_logger(
            os.path.join(
                base_output_path,
                f"{base_filename}.{actiontype}.log",
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
    config: dict[str, Any] = utils.get_hocon_config(
        config_path=config_path,
        config_name=config_name,
    )

    # Save the experiment configuration to the output path
    utils.write_json(
        os.path.join(base_output_path, "config.json"),
        config,
    )

    # Initialize the loaded LLM map
    loaded_llm_map: dict[str, BaseLLM] = {}

    # Instantiate the Proposition Extraction component
    proposition_extraction: BasePropositionExtractor | None = None
    if actiontype == "proposition_extraction":
        proposition_extraction, loaded_llm_map = (
            instantiate_proposition_extraction_component(
                proposition_extraction_config=config["proposition_extraction"],
                loaded_llm_map=loaded_llm_map,
            )
        )

    # Instantiate the Proposition Relation Extraction component
    proposition_relation_extraction: (
        BasePropositionRelationExtractor | None
    ) = None
    if actiontype == "proposition_relation_extraction":
        proposition_relation_extraction, loaded_llm_map = (
            instantiate_proposition_relation_extraction_component(
                proposition_relation_extraction_config=(
                    config["proposition_relation_extraction"]
                ),
                loaded_llm_map=loaded_llm_map,
            )
        )

    # Instantiate the Proposition Relation Refinement component
    proposition_relation_refinement: (
        BasePropositionRelationRefiner | None
    ) = None
    if actiontype == "proposition_relation_refinement":
        proposition_relation_refinement, loaded_llm_map = (
            instantiate_proposition_relation_refinement_component(
                proposition_relation_refinement_config=(
                    config["proposition_relation_refinement"]
                ),
                loaded_llm_map=loaded_llm_map,
            )
        )

    # Instantiate the Passage Graph Construction component
    passage_graph_construction: BasePassageGraphConstructor | None = None
    if actiontype == "passage_graph_construction":
        passage_graph_construction = (
            instantiate_passage_graph_construction_component(
                passage_graph_construction_config=(
                    config["passage_graph_construction"]
                ),
            )
        )

    # Instantiate the Passage Retrieval component
    passage_retrieval: BasePassageRetriever | None = None
    if (
        actiontype == "passage_retrieval_indexing"
        or actiontype == "inference"
    ):
        passage_retrieval = instantiate_passage_retrieval_component(
            passage_retrieval_config=config["passage_retrieval"],
        )

    # Instantiate the Graph Retrieval component
    graph_retrieval: BaseGraphRetriever | None = None
    if actiontype == "inference":
        graph_retrieval = instantiate_graph_retrieval_component(
            graph_retrieval_config=config["graph_retrieval"],
        )

    # Instantiate the Context Formatting component
    context_formatting: BaseContextFormatter | None = None
    if actiontype == "inference":
        context_formatting = instantiate_context_formatting_component(
            context_formatting_config=config["context_formatting"],
        )

    # Instantiate the QA component
    qa: BaseQA | None = None
    if actiontype == "inference":
        qa, loaded_llm_map = instantiate_qa_component(
            qa_config=config["qa"],
            loaded_llm_map=loaded_llm_map,
        )

    # Instantiate the ProStruct-RAG pipeline
    prostruct_rag: ProStructRAGPipeline = ProStructRAGPipeline(
        proposition_extraction=proposition_extraction,
        proposition_relation_extraction=proposition_relation_extraction,
        proposition_relation_refinement=proposition_relation_refinement,
        passage_graph_construction=passage_graph_construction,
        passage_retrieval=passage_retrieval,
        graph_retrieval=graph_retrieval,
        context_formatting=context_formatting,
        qa=qa,
    )

    ##################
    # Method Execution
    ##################

    node_id_key: str = config["passage_graph_construction"]["node_id_key"]

    # Require Passage Graph Construction and Graph Retrieval to use the same node ID key
    if config["graph_retrieval"]["node_id_key"] != node_id_key:
        raise ValueError(
            "Passage Graph Construction and Graph Retrieval must use "
            "the same node_id_key."
        )

    if actiontype != "inference":
        # Load passages only when Proposition Extraction is selected
        if actiontype == "proposition_extraction":
            if input_passages_path is None:
                raise ValueError(
                    "--input_passages is required for proposition_extraction"
                )
            passages: list[Passage] = utils.read_jsonl(input_passages_path)
        else:
            passages = None

        # Set component-specific arguments
        if config["proposition_relation_extraction"]["retriever"]["method_name"] == (
            "bm25"
        ):
            proposition_relation_extraction_indexing_kwargs = {}
            passage_retrieval_indexing_kwargs = {}
        else:
            proposition_relation_extraction_indexing_kwargs={
                "batch_size": (
                    config["proposition_relation_extraction"]["retriever"][
                        "indexing_batch_size"
                    ]
                ),
            }
            passage_retrieval_indexing_kwargs = {
                "batch_size": (
                    config["passage_retrieval"]["indexing_batch_size"]
                ),
            }

        # Run the selected ProStruct-RAG indexing component
        prostruct_rag.make_index(
            # Input
            passages=passages,
            # Output directory
            index_dir=index_dir,
            # Component-specific arguments
            top_k=(
                config["proposition_relation_extraction"]["retriever"][
                    "top_k"
                ]
            ),
            prefilter_k=(
                config["proposition_relation_extraction"]["retriever"][
                    "prefilter_k"
                ]
            ),
            search_batch_size=(
                config["proposition_relation_extraction"]["retriever"][
                    "search_batch_size"
                ]
            ),
            node_id_key=node_id_key,
            source_id_key=config["proposition_extraction"]["source_id_key"],
            proposition_relation_extraction_indexing_kwargs=(
                proposition_relation_extraction_indexing_kwargs
            ),
            passage_retrieval_indexing_kwargs=passage_retrieval_indexing_kwargs,
            # Target component for indexing
            target_component=actiontype,
        )

    else:
        assert base_filename is not None

        # Load questions
        if input_questions_path is None:
            raise ValueError("--input_questions is required for inference")
        questions: list[Question] = utils.read_json(input_questions_path)

        logging.info(
            f"Applying the ProStruct-RAG pipeline to "
            f"{len(questions)} questions in {input_questions_path} ..."
        )

        # Load the built index
        prostruct_rag.load_index(index_dir=index_dir)

        # Run all inference components for every question
        result_questions: list[Question] = []
        for question in tqdm(questions):
            result_question: Question = prostruct_rag.infer(
                question=question,
                top_k=config["passage_retrieval"]["top_k"],
                hop_size=config["graph_retrieval"]["hop_size"],
                node_id_key=node_id_key,
            )
            result_questions.append(result_question)

        # Save the results
        output_questions_path: str = os.path.join(
            base_output_path,
            f"{base_filename}.pred.json",
        )
        utils.write_json(output_questions_path, result_questions)
        logging.info(f"Saved QA results to {output_questions_path}")

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
            # Require gold answers only when evaluation is requested
            if gold_questions_path is None:
                raise ValueError(
                    "--gold is required when --do_evaluation is set"
                )

            # Evaluate the prediction results
            qa_scores: dict[str, Any] = evaluation.qa.accuracy(
                pred_path=output_questions_path,
                gold_path=gold_questions_path,
                exact_match=False,
            )
            qa_scores.update(
                evaluation.qa.token_level_f1(
                    pred_path=output_questions_path,
                    gold_path=gold_questions_path,
                )
            )
            # qa_scores.update(
            #     evaluation.qa.recall(
            #         pred_path=output_questions_path,
            #         gold_path=gold_questions_path,
            #         exact_match=False,
            #     )
            # )
            scores: dict[str, Any] = {
                "qa": qa_scores,
            }
            logging.info(utils.pretty_format_dict(scores))

            # Save the evaluation results
            output_evaluation_path: str = os.path.join(
                base_output_path,
                f"{base_filename}.eval.json",
            )
            utils.write_json(output_evaluation_path, scores)

            # Log the evaluation results
            logging.info(utils.pretty_format_dict(scores))
            logging.info(
                f"Saved evaluation results to {output_evaluation_path}"
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
        do_remove: str = input("Delete the existing log file? [y/n]: ")
        if (not do_remove.lower().startswith("y")) and (not len(do_remove) == 0):
            logging.info("Done.")
            sys.exit(0)

    root_logger: logging.Logger = logging.getLogger()
    handler: logging.FileHandler = logging.FileHandler(filename, "w")
    root_logger.addHandler(handler)


def instantiate_proposition_extraction_component(
    proposition_extraction_config: dict[str, Any],
    loaded_llm_map: dict[str, BaseLLM] | None,
) -> tuple[BasePropositionExtractor, dict[str, BaseLLM]]:

    # Instantiate the Proposition Extraction component
    if proposition_extraction_config["method_name"] == "llm_proposition_extractor":
        # Instantiate the LLM wrapper
        llm: BaseLLM
        llm, loaded_llm_map = instantiate_llm(
            config=proposition_extraction_config,
            loaded_llm_map=loaded_llm_map,
        )

        # Instantiate the LLM-based Proposition Extraction component
        proposition_extraction: BasePropositionExtractor = LLMPropositionExtractor(
            model=llm,
            prompt_template_name_or_path=(
                proposition_extraction_config["prompt_template_name_or_path"]
            ),
            include_title_as_proposition=(
                proposition_extraction_config["include_title_as_proposition"]
            ),
        )
    else:
        raise ValueError(
            "Unknown Proposition Extraction method: "
            f"{proposition_extraction_config['method_name']}"
        )

    return proposition_extraction, loaded_llm_map


def instantiate_proposition_relation_extraction_component(
    proposition_relation_extraction_config: dict[str, Any],
    loaded_llm_map: dict[str, BaseLLM] | None,
) -> tuple[BasePropositionRelationExtractor, dict[str, BaseLLM]]:

    # Instantiate the Proposition Relation Extraction component
    if (
        proposition_relation_extraction_config["method_name"]
        == "llm_proposition_relation_extractor"
    ):
        # Instantiate the LLM wrapper
        llm: BaseLLM
        llm, loaded_llm_map = instantiate_llm(
            config=proposition_relation_extraction_config,
            loaded_llm_map=loaded_llm_map,
        )

        # Instantiate the Passage Retrieval component for
        # candidate proposition retrieval.
        retriever: BasePassageRetriever = instantiate_passage_retrieval_component(
            passage_retrieval_config=(
                proposition_relation_extraction_config["retriever"]
            ),
        )

        # Instantiate the LLM-based Proposition Relation Extraction component
        proposition_relation_extraction: BasePropositionRelationExtractor = (
            LLMPropositionRelationExtractor(
                model=llm,
                retriever=retriever,
                prompt_template_name_or_path=(
                    proposition_relation_extraction_config[
                        "prompt_template_name_or_path"
                    ]
                ),
                use_timestamp=(
                    proposition_relation_extraction_config["use_timestamp"]
                ),
            )
        )
    else:
        raise ValueError(
            "Unknown Proposition Relation Extraction method: "
            f"{proposition_relation_extraction_config['method_name']}"
        )

    return proposition_relation_extraction, loaded_llm_map


def instantiate_proposition_relation_refinement_component(
    proposition_relation_refinement_config: dict[str, Any],
    loaded_llm_map: dict[str, BaseLLM] | None,
) -> tuple[BasePropositionRelationRefiner, dict[str, BaseLLM]]:

    # Instantiate the Proposition Relation Refinement component
    if (
        proposition_relation_refinement_config["method_name"]
        == "llm_proposition_relation_refiner"
    ):
        # Instantiate the LLM wrapper
        llm: BaseLLM
        llm, loaded_llm_map = instantiate_llm(
            config=proposition_relation_refinement_config,
            loaded_llm_map=loaded_llm_map,
        )

        # Instantiate the LLM-based Proposition Relation Refinement component
        proposition_relation_refinement: BasePropositionRelationRefiner = (
            LLMPropositionRelationRefiner(
                model=llm,
                prompt_template_name_or_path=(
                    proposition_relation_refinement_config[
                        "prompt_template_name_or_path"
                    ]
                ),
                use_timestamp=(
                    proposition_relation_refinement_config["use_timestamp"]
                ),
            )
        )
    else:
        raise ValueError(
            "Unknown Proposition Relation Refinement method: "
            f"{proposition_relation_refinement_config['method_name']}"
        )

    return proposition_relation_refinement, loaded_llm_map


def instantiate_passage_graph_construction_component(
    passage_graph_construction_config: dict[str, Any],
) -> BasePassageGraphConstructor:

    # Instantiate the Passage Graph Construction component
    if (
        passage_graph_construction_config["method_name"]
        == "passage_graph_constructor"
    ):
        passage_graph_construction: BasePassageGraphConstructor = (
            PassageGraphConstructor()
        )
    else:
        raise ValueError(
            "Unknown Passage Graph Construction method: "
            f"{passage_graph_construction_config['method_name']}"
        )

    return passage_graph_construction


def instantiate_passage_retrieval_component(
    passage_retrieval_config: dict[str, Any],
) -> BasePassageRetriever:

    # Instantiate the BM25-based Passage Retrieval component
    if passage_retrieval_config["method_name"] == "bm25":
        passage_retrieval: BasePassageRetriever = BM25(
            tokenizer=lambda text: text.lower().split(),
            k1=passage_retrieval_config["k1"],
            b=passage_retrieval_config["b"],
        )

    # Instantiate the Contriever-based Passage Retrieval component
    elif passage_retrieval_config["method_name"] == "contriever":
        passage_retrieval: BasePassageRetriever = Contriever(
            model_name=passage_retrieval_config["model_name"],
            max_passage_length=passage_retrieval_config["max_passage_length"],
            pooling_method=passage_retrieval_config["pooling_method"],
            normalize=passage_retrieval_config["normalize"],
            metric=passage_retrieval_config["metric"],
        )

    # Instantiate the Qwen3-Embedding-based Passage Retrieval component
    elif passage_retrieval_config["method_name"] == "qwen3_embedding":
        passage_retrieval: BasePassageRetriever = Qwen3Embedding(
            model_name=passage_retrieval_config["model_name"],
            max_passage_length=passage_retrieval_config["max_passage_length"],
            normalize=passage_retrieval_config["normalize"],
            metric=passage_retrieval_config["metric"],
            query_instruction=passage_retrieval_config["query_instruction"],
        )

    else:
        raise ValueError(
            "Unknown passage retrieval method: "
            f"{passage_retrieval_config['method_name']}"
        )

    return passage_retrieval


def instantiate_graph_retrieval_component(
    graph_retrieval_config: dict[str, Any],
) -> BaseGraphRetriever:

    # Instantiate the Graph Retrieval component
    if graph_retrieval_config["method_name"] == "graph_retriever":
        graph_retrieval: BaseGraphRetriever = GraphRetriever(
            use_timestamp=graph_retrieval_config["use_timestamp"],
        )
    else:
        raise ValueError(
            f"Unknown Graph Retrieval method: {graph_retrieval_config['method_name']}"
        )

    return graph_retrieval


def instantiate_context_formatting_component(
    context_formatting_config: dict[str, Any],
) -> BaseContextFormatter:

    # Instantiate the Context Formatting component
    if context_formatting_config["method_name"] == "graph_verbalizer":
        context_formatting: BaseContextFormatter = GraphVerbalizer(
            use_timestamp=context_formatting_config["use_timestamp"],
        )
    else:
        raise ValueError(
            "Unknown Context Formatting method: "
            f"{context_formatting_config['method_name']}"
        )

    return context_formatting


def instantiate_qa_component(
    qa_config: dict[str, Any],
    loaded_llm_map: dict[str, BaseLLM] | None,
) -> tuple[BaseQA, dict[str, BaseLLM]]:

    # Instantiate the QA component
    if qa_config["method_name"] == "llm_qa":
        # Instantiate the LLM wrapper
        llm: BaseLLM
        llm, loaded_llm_map = instantiate_llm(
            config=qa_config,
            loaded_llm_map=loaded_llm_map,
        )

        # Instantiate the LLM-based QA component
        qa: BaseQA = LLMQA(
            model=llm,
            prompt_template_name_or_path=qa_config["prompt_template_name_or_path"],
            n_contexts=qa_config["n_contexts"],
        )

    else:
        raise ValueError(f"Unknown QA method: {qa_config['method_name']}")

    return qa, loaded_llm_map


def instantiate_llm(
    config: dict[str, Any],
    loaded_llm_map: dict[str, BaseLLM]
) -> tuple[BaseLLM, dict[str, BaseLLM]]:

    # Create a key for reusing the same LLM instance
    key_parts: list[str] = [
        config["llm_provider"],
        config["llm_model_name"],
        str(config["llm_max_new_tokens"]),
    ]
    if config["llm_provider"] == "hf":
        key_parts.append(str(config["llm_quantization_bits"]))
    key: str = "___".join(key_parts)

    # If the key is found, reuse the corresponding LLM
    if key in loaded_llm_map:
        return loaded_llm_map[key], loaded_llm_map

    # Instantiate the LLM wrapper
    llm: BaseLLM
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

    # Add the key-LLM record to the dict
    loaded_llm_map[key] = llm

    return llm, loaded_llm_map


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO
    )
    logging.getLogger("httpx").addFilter(
        lambda r: "huggingface.co" not in r.getMessage()
    )

    parser: argparse.ArgumentParser = argparse.ArgumentParser()

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
            "proposition_extraction",
            "proposition_relation_extraction",
            "proposition_relation_refinement",
            "passage_graph_construction",
            "passage_retrieval_indexing",
            "inference",
        ],
    )

    # Evaluation
    parser.add_argument("--do_evaluation", action="store_true")
    parser.add_argument("--gold", type=str, default=None)

    args: argparse.Namespace = parser.parse_args()

    main(args)
