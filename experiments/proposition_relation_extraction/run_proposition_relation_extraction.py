import argparse
import json
import logging
import os
import sys
from typing import Any

from tqdm import tqdm

from kapipe import utils
from kapipe.llms import HuggingFaceLLM, OpenAILLM
from kapipe.passage_retrieval import BM25, Contriever, Qwen3Embedding
from kapipe.proposition_relation_extraction import LLMPropositionRelationExtractor
from kapipe.utils import StopWatch


def main(args):
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
    input_propositions_path = args.input_propositions

    # Output Path
    results_dir = args.results_dir
    prefix = args.prefix
    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()
        args.prefix = prefix

    # Batch API
    batch_mode: str | None = args.batch_mode

    assert method_name in ["llm_proposition_relation_extractor"], f"Unknown method: {method_name}"

    ##################
    # Logging Setup
    ##################

    # Set base output path
    base_output_path = os.path.join(
        results_dir,
        "proposition_relation_extraction",
        method_name,
        config_name,
        prefix
    )
    utils.mkdir(base_output_path)

    # Extract the base filename
    base_filename = os.path.splitext(
        os.path.basename(input_propositions_path)
    )[0]
 
    # Set logger
    if batch_mode is None:
        set_logger(
            os.path.join(
                base_output_path,
                f"{base_filename}.proposition_relation_extraction.log",
            ),
            # overwrite=True
        )
    else:
        set_logger(
            os.path.join(
                base_output_path,
                f"{base_filename}.proposition_relation_extraction.{batch_mode}.log",
            ),
            # overwrite=True
        )
 
    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))

    ##################
    # Data
    ##################

    # Load propositions
    logging.info("Loading propositions ...")
    propositions = []
    with open(input_propositions_path) as f:
        for line in f:
            proposition = json.loads(line.strip())
            propositions.append(proposition)
    logging.info(f"Number of propositions: {len(propositions)}")

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

    # Instantiate the LLM wrapper
    if method_name == "llm_proposition_relation_extractor":
        if config["llm_provider"] == "openai":
            model = OpenAILLM(
                model_name=config["llm_model_name"],
                max_new_tokens=config["llm_max_new_tokens"],
            )
        elif config["llm_provider"] == "hf":
            model = HuggingFaceLLM(
                model_name=config["llm_model_name"],
                max_new_tokens=config["llm_max_new_tokens"],
                quantization_bits=config["llm_quantization_bits"],
            )
        else:
            raise ValueError(f"Unknown LLM provider: {config['llm_provider']}")
        logging.info("Instantiated the LLM model: %s" % repr(model))

    # Instantiate the Passage Retrieval component
    if config["retriever"]["method_name"] == "bm25":
        retriever = BM25(
            tokenizer=lambda text: text.lower().split(),
            k1=config["retriever"]["k1"],
            b=config["retriever"]["b"],
        )
    elif config["retriever"]["method_name"] == "contriever":
        retriever = Contriever(
            model_name=config["retriever"]["model_name"],
            max_passage_length=config["retriever"]["max_passage_length"],
            pooling_method=config["retriever"]["pooling_method"],
            normalize=config["retriever"]["normalize"],
            metric=config["retriever"]["metric"],
        )
    elif config["retriever"]["method_name"] == "qwen3_embedding":
        retriever = Qwen3Embedding(
            model_name=config["retriever"]["model_name"],
            max_passage_length=config["retriever"]["max_passage_length"],
            normalize=config["retriever"]["normalize"],
            metric=config["retriever"]["metric"],
            query_instruction=config["retriever"]["query_instruction"],
        )
    else:
        raise ValueError(
            "Invalid retrieval method name: %s"
            % config["retriever"]["method_name"]
        )

    # Instantiate the Proposition Relation Extraction component
    if method_name == "llm_proposition_relation_extractor":
        extractor = LLMPropositionRelationExtractor(
            model=model,
            retriever=retriever,
            prompt_template_name_or_path=config["prompt_template_name_or_path"],
            use_timestamp=config["use_timestamp"],
        )
    else:
        raise ValueError(f"Unknown method name: {method_name}")

    ##################
    # Method Execution
    ##################

    logging.info(
        f"Applying the Proposition Relation Extraction component "
        f"to {len(propositions)} propositions in {input_propositions_path} ..."
    )

    # Apply the Proposition Relation Extraction component to the propositions

    # [Step 1-1] Build index for propositions
    index_dir = os.path.join(
        base_output_path,
        "intermediate_proposition_retrieval"
    )
    if config["retriever"]["method_name"] == "bm25":
        extractor.make_index(
            propositions=propositions,
            index_dir=index_dir,
        )
    else:
        extractor.make_index(
            propositions=propositions,
            index_dir=index_dir,
            batch_size=config["retriever"]["indexing_batch_size"],
        )

    # [Step 1-2] Retrieve tail propositions for each proposition
    batch_tail_propositions = extractor.batch_retrieve_tail_propositions(
        head_propositions=propositions,
        top_k=config["retriever"]["top_k"],
        prefilter_k=config["retriever"]["prefilter_k"],
        batch_size=config["retriever"]["search_batch_size"],
    )

    # [Step 2] Extract proposition relations for each proposition
    if batch_mode is None:
        triples = []
        for head_proposition, tail_propositions in tqdm(
            zip(propositions, batch_tail_propositions),
            total=len(propositions)
        ):
            triples_for_head = extractor.extract(
                head_proposition=head_proposition,
                tail_propositions=tail_propositions,
            )
            triples.extend(triples_for_head)

        logging.info(
            f"Extracted {len(triples)} triples from {len(propositions)} propositions"
        )

        # Save the Proposition Relation Extraction results
        output_triples_path = os.path.join(
            base_output_path,
            f"{base_filename}.triples.json",
        )
        utils.write_json(output_triples_path, triples)
        logging.info(f"Saved triples to {output_triples_path}")

    elif batch_mode == "submit":
        # Submit prompts
        batch_ids: list[str] = extractor.submit_batch(
            head_propositions=propositions,
            batch_tail_propositions=batch_tail_propositions,
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
        triples: list[dict[str, Any]] = extractor.fetch_and_process_batch(
            head_propositions=propositions,
            batch_tail_propositions=batch_tail_propositions,
            batch_ids=batch_ids,
        )

        logging.info(
            f"Extracted {len(triples)} triples from {len(propositions)} propositions"
        )

        # Save the Proposition Relation Extraction results
        output_triples_path: str = os.path.join(
            base_output_path,
            f"{base_filename}.triples.json",
        )
        utils.write_json(output_triples_path, triples)
        logging.info(f"Saved triples to {output_triples_path}")

    else:
        raise ValueError(
            f"Invalid batch_mode: {batch_mode}. "
            "Expected None, 'submit', or 'fetch'."
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


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO
    )
    parser = argparse.ArgumentParser()

    # Method
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=True)

    # Input Data
    parser.add_argument("--input_propositions", type=str, required=True)

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    # Batch API
    parser.add_argument(
        "--batch_mode",
        type=str,
        default=None,
        choices=["submit", "fetch"],
    )

    args = parser.parse_args()
    main(args)
