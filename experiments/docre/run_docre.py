import argparse
import logging
import os
import sys

from tqdm import tqdm
import transformers

from kapipe import evaluation
from kapipe import utils
from kapipe.docre import ATLOP, LLMDocRE
from kapipe.llms import HuggingFaceLLM, OpenAILLM
from kapipe.utils import StopWatch


def main(args):
    # torch.autograd.set_detect_anomaly(True)
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
    input_documents_path = args.input_documents

    # Output Path
    results_dir = args.results_dir
    prefix = args.prefix
    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()
        args.prefix = prefix

    # Evaluation
    do_evaluation = args.do_evaluation
    gold_documents_path = args.gold

    # Batch API
    batch_mode: str | None = args.batch_mode

    ##################
    # Logging Setup
    ##################

    # Set base output path
    base_output_path = os.path.join(
        results_dir,
        "docre",
        method_name,
        config_name,
        prefix
    )
    utils.mkdir(base_output_path)

    # Extract the base filename
    base_filename = os.path.splitext(os.path.basename(input_documents_path))[0]

    # Set logger
    if batch_mode is None:
        set_logger(
            os.path.join(
                base_output_path,
                f"{base_filename}.docre.log",
            ),
            # overwrite=True
        )
    else:
        set_logger(
            os.path.join(
                base_output_path,
                f"{base_filename}.docre.{batch_mode}.log",
            ),
            # overwrite=True
        )
 
    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))

    ##################
    # Data
    ##################

    # Load documents
    documents = utils.read_json(input_documents_path)

    ##################
    # Method Instantiation
    ##################

   # Load the experiment configuration
    config = utils.get_hocon_config(config_path=config_path, config_name=config_name)

    # Save the experiment configuration to the output path
    utils.write_json(os.path.join(base_output_path, "config.json"), config)

    # Instantiate the DocRE component
    if method_name == "atlop":
        # Load the ATLOP extractor from the public snapshot
        extractor = ATLOP.from_identifier(
            identifier=config["identifier"]
        )
    elif method_name == "llm_docre":
        # Instantiate the LLM wrapper
        if config["llm_provider"] == "openai":
            model = OpenAILLM(
                model_name=config["llm_model_name"],
                max_new_tokens=config["llm_max_new_tokens"],
            )
        elif config["llm_provider"] == "hf":
            model = HuggingFaceLLM(
                model_name=config["llm_model_name"],
                max_new_tokens=config["llm_max_new_tokens"],
                quantization_bits=config["llm_quantization_bits"]
            )
        else:
            raise ValueError(f"Unknown LLM provider: {config['llm_provider']}")
        logging.info("Instantiated the LLM model: %s" % repr(model))

        if "identifier" in config: 
            # Load the LLM-based DocRE extractor from the public snapshot
            extractor = LLMDocRE.from_identifier(
                model=model,
                identifier=config["identifier"]
            )
        else:
            # Load the user-defined schema
            possible_head_entity_types = config["possible_head_entity_types"]
            possible_tail_entity_types = config["possible_tail_entity_types"]
            vocab_relation: dict[str, int] = {
                rel: rel_i
                for rel_i, rel in enumerate(config["relations"])
            }
            rel_meta_info: dict[str, dict[str, str]] = config["rel_meta_info"]
            entity_dict_path = config.get("entity_dict_path", None)

            # Instantiate the LLM-based DocRE component with the user-defined schema
            extractor = LLMDocRE(
                model=model,
                prompt_template_name_or_path=config["prompt_template_name_or_path"],
                knowledge_base_name=config["knowledge_base_name"],
                mention_style=config["mention_style"],
                with_span_annotation=config["with_span_annotation"],
                possible_head_entity_types=possible_head_entity_types,
                possible_tail_entity_types=possible_tail_entity_types,
                vocab_relation=vocab_relation,
                rel_meta_info=rel_meta_info,
                entity_dict_path=entity_dict_path,
            )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    ##################
    # Method Execution
    ##################

    logging.info(
        f"Applying the DocRE component to {len(documents)} documents "
        f"in {input_documents_path} ..."
    )

    # Apply the DocRE component to the documents
    if batch_mode is None:
        result_documents = []
        for document in tqdm(documents):
            result_document = extractor.extract(document=document)
            result_documents.append(result_document)

    elif batch_mode == "submit":
        # Submit prompts
        batch_ids: list[str] = extractor.submit_batch(documents=documents)
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
        result_documents = extractor.fetch_and_process_batch(
            documents=documents,
            batch_ids=batch_ids,
        )

    else:
        raise ValueError(
            f"Invalid batch_mode: {batch_mode}. "
            "Expected None, 'submit', or 'fetch'."
        )

    if batch_mode != "submit":
        # Save the results
        output_documents_path = os.path.join(base_output_path, f"{base_filename}.pred.json")
        utils.write_json(output_documents_path, result_documents)
        logging.info(f"Saved the prediction results to {output_documents_path}")

        # Save the prompt-response pairs visually in plain text
        if "docre_prompt" in result_documents[0] and "docre_generated_text" in result_documents[0]:
            output_text_path = os.path.join(base_output_path, f"{base_filename}.prompt_and_response.txt")
            with open(output_text_path, "w") as f:
                for doc in result_documents:
                    doc_key = doc["doc_key"]
                    prompt = doc["docre_prompt"]
                    generated_text = doc["docre_generated_text"]
                    f.write("-------------------------------------\n\n")
                    f.write(f"DOC_KEY: {doc_key}\n\n")
                    f.write("PROMPT:\n")
                    f.write(prompt + "\n\n")
                    f.write("GENERATED TEXT:\n")
                    f.write(generated_text + "\n\n")
                    f.flush()

        ##################
        # Evaluation
        ##################

        if do_evaluation:
            # Validate that the gold documents path is provided
            if gold_documents_path is None:
                raise ValueError("--gold is required when --do_evaluation is set")

            # Calculate standard DocRE scores
            scores = evaluation.docre.fscore(
                pred_path=output_documents_path,
                gold_path=gold_documents_path,
                skip_intra_inter=True,
                skip_ign=True,
            )
            logging.info(utils.pretty_format_dict(scores))

            # Save the evaluation scores
            output_evaluation_path = os.path.join(
                base_output_path,
                f"{base_filename}.eval.json",
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
    parser.add_argument("--input_documents", type=str, required=True)

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
