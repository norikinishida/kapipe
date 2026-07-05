import argparse
import logging
import os
from typing import Any
import sys

import torch
from tqdm import tqdm
import transformers

from kapipe import evaluation
from kapipe import utils
from kapipe.utils import StopWatch

from kapipe.pipelines import TripleExtractionPipeline

from kapipe.llms import BaseLLM, HuggingFaceLLM, OpenAILLM
from kapipe.ner import (
    BaseNER,
    BiaffineNER,
    LLMNER,
)
from kapipe.ed_retrieval import (
    BaseEDRetriever,
    MentionNameEntityRetriever,
    BlinkBiEncoder,
)
from kapipe.ed_reranking import (
    BaseEDReranker,
    IdenticalEntityReranker,
    BlinkCrossEncoder,
    LLMED,
)
from kapipe.docre import (
    BaseDocRE,
    ATLOP,
    LLMDocRE,
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

    ##################
    # Logging Setup
    ##################

    # Set base output path
    base_output_path = os.path.join(
        results_dir,
        "triple_extraction_pipeline",
        method_name,
        config_name,
        prefix
    )
    utils.mkdir(base_output_path)

    base_filename = os.path.splitext(os.path.basename(input_documents_path))[0]

    # Set logger
    set_logger(
        os.path.join(
            base_output_path,
            f"{base_filename}.triple_extraction_pipeline.log"
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

    # Initialize the loaded LLM map
    loaded_llm_map: dict[str, BaseLLM] = {}

    # Instantiate the NER component
    ner, loaded_llm_map = instantiate_ner_component(
        ner_config=config["ner"],
        loaded_llm_map=loaded_llm_map,
    )

    # Instantiate the ED-Retrieval component
    ed_retrieval, loaded_llm_map = instantiate_ed_retrieval_component(
        ed_retrieval_config=config["ed_retrieval"],
        loaded_llm_map=loaded_llm_map,
    )

    # Instantiate the ED-Reranking component
    ed_reranking, loaded_llm_map = instantiate_ed_reranking_component(
        ed_reranking_config=config["ed_reranking"],
        loaded_llm_map=loaded_llm_map,
    )

    # Instantiate the DocRE component
    docre, loaded_llm_map = instantiate_docre_component(
        docre_config=config["docre"],
        loaded_llm_map=loaded_llm_map,
    )

    # Instantiate the Triple Extraction pipeline
    extractor = TripleExtractionPipeline(
        ner=ner,
        ed_retrieval=ed_retrieval,
        ed_reranking=ed_reranking,
        docre=docre,
    )

    ##################
    # Method Execution
    ##################

    logging.info(f"Applying the Triple Extraction pipeline to {len(documents)} documents in {input_documents_path} ...")

    # Apply the Triple Extraction pipeline to the documents
    result_documents = []
    for document in tqdm(documents):
        result_document = extractor.extract_triples(
            document=document,
            retrieval_size=config["ed_retrieval"]["retrieval_size"]
        )
        result_documents.append(result_document)

    # Save the results
    output_documents_path = os.path.join(base_output_path, f"{base_filename}.pred.json")
    utils.write_json(output_documents_path, result_documents)
    logging.info(f"Saved the prediction results to {output_documents_path}")

    ##################
    # Evaluation
    ##################

    if do_evaluation:
        # Require gold documents only when evaluation is requested
        if gold_documents_path is None:
            raise ValueError("--gold is required when --do_evaluation is set")

        # Evaluate the prediction results
        ner_scores = evaluation.ner.fscore(
            pred_path=output_documents_path,
            gold_path=gold_documents_path
        )
        ed_scores_mention_level = evaluation.ed.fscore(
            pred_path=output_documents_path,
            gold_path=gold_documents_path,
            inkb=False,
            skip_normalization=True,
            on_predicted_spans=True
        )
        ed_scores_entity_level = evaluation.ed.entity_level_fscore(
            pred_path=output_documents_path,
            gold_path=gold_documents_path
        )
        docre_scores = evaluation.docre.fscore(
            pred_path=output_documents_path,
            gold_path=gold_documents_path,
            skip_intra_inter=True,
            skip_ign=True
        )
        scores = {
            "ner": ner_scores,
            "ed_mention_level": ed_scores_mention_level,
            "ed_entity_level": ed_scores_entity_level,
            "docre": docre_scores,
        }
        logging.info(utils.pretty_format_dict(scores))

        # Save the evaluation results
        output_evaluation_path = os.path.join(base_output_path, f"{base_filename}.eval.json")
        utils.write_json(output_evaluation_path, scores)
        logging.info(f"Saved the evaluation results to {output_evaluation_path}")

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


def instantiate_ner_component(
    ner_config: dict[str, Any],
    loaded_llm_map: dict[str, BaseLLM],
) -> tuple[BaseNER, dict[str, BaseLLM]]:

    # Instantiate the NER component
    if ner_config["method_name"] == "biaffine_ner":
        # Load the Biaffine NER component from the public snapshot
        ner = BiaffineNER.from_identifier(
            identifier=ner_config["identifier"]
        )
    elif ner_config["method_name"] == "llm_ner":
        # Instantiate the LLM wrapper
        llm, loaded_llm_map = instantiate_llm(
            config=ner_config,
            loaded_llm_map=loaded_llm_map
        )

        if "identifier" in ner_config:
            # Load the LLM-based NER component from the public snapshot
            ner = LLMNER.from_identifier(
                model=llm,
                identifier=ner_config["identifier"],
            )

        else:
            # Load the user-defined schema
            vocab_etype: dict[str, int] = {
                etype: etype_i
                for etype_i, etype in enumerate(ner_config["entity_types"])
            }
            etype_meta_info: dict[str, dict[str, str]] = ner_config["etype_meta_info"]

            # Instantiate the LLM-based NER component with the user-defined schema
            ner = LLMNER(
                model=llm,
                prompt_template_name_or_path=ner_config["prompt_template_name_or_path"],
                vocab_etype=vocab_etype,
                etype_meta_info=etype_meta_info,
            )

    else:
        raise ValueError(f"Unknown NER method: {ner_config['method_name']}")

    return ner, loaded_llm_map


def instantiate_ed_retrieval_component(
    ed_retrieval_config: dict[str, Any],
    loaded_llm_map: dict[str, BaseLLM],
) -> tuple[BaseEDRetriever, dict[str, BaseLLM]]:

    # Instantiate the ED-Retrieval component.
    # Also, re-build the index over entities.
    if ed_retrieval_config["method_name"] == "mention_name_entity_retriever":
        # Instantiate the ED-Retrieval component using simple mention-name assignment
        ed_retrieval = MentionNameEntityRetriever()
        ed_retrieval.make_index()
    elif ed_retrieval_config["method_name"] == "blink_bi_encoder":
        # Load the BLINK Bi-Encoder ED-Retrieval component from the public snapshot
        ed_retrieval = BlinkBiEncoder.from_identifier(
            identifier=ed_retrieval_config["identifier"]
        )
        ed_retrieval.make_index(use_precomputed_entity_vectors=True)
    else:
        raise ValueError(
            f"Unknown ED-Retrieval method: {ed_retrieval_config['method_name']}"
        )

    return ed_retrieval, loaded_llm_map


def instantiate_ed_reranking_component(
    ed_reranking_config: dict[str, Any],
    loaded_llm_map: dict[str, BaseLLM],
) -> tuple[BaseEDReranker, dict[str, BaseLLM]]:

    # Instantiate the ED-Reranking component
    if ed_reranking_config["method_name"] == "identical_entity_reranker":
        # Instantiate the ED-Reranking component using identical function
        ed_reranking = IdenticalEntityReranker()
    elif ed_reranking_config["method_name"] == "blink_cross_encoder":
        # Load the BLINK Cross-Encoder ED-Reranking component from the public snapshot
        ed_reranking = BlinkCrossEncoder.from_identifier(
            identifier=ed_reranking_config["identifier"]
        )
    elif ed_reranking_config["method_name"] == "llm_ed":
        # Instantiate the LLM wrapper
        llm, loaded_llm_map = instantiate_llm(
            config=ed_reranking_config,
            loaded_llm_map=loaded_llm_map
        )
        # Load the LLM-based ED-Reranking component from the public snapshot
        ed_reranking = LLMED.from_identifier(
            model=llm,
            identifier=ed_reranking_config["identifier"],
        )
    else:
        raise ValueError(
            f"Unknown ED-Reranking method: {ed_reranking_config['method_name']}"
        )

    return ed_reranking, loaded_llm_map


def instantiate_docre_component(
    docre_config: dict[str, Any],
    loaded_llm_map: dict[str, BaseLLM],
) -> tuple[BaseDocRE, dict[str, BaseLLM]]:

    # Instantiate the DocRE component
    if docre_config["method_name"] == "atlop":
        # Load the ATLOP-based DocRE component from the public snapshot
        docre = ATLOP.from_identifier(
            identifier=docre_config["identifier"]
        )
    elif docre_config["method_name"] == "llm_docre":
        # Instantiate the LLM wrapper
        llm, loaded_llm_map = instantiate_llm(
            config=docre_config,
            loaded_llm_map=loaded_llm_map
        )

        if "identifier" in docre_config:
            # Load the LLM-based DocRE component from the public snapshot
            docre = LLMDocRE.from_identifier(
                model=llm,
                identifier=docre_config["identifier"],
            )

        else:
            # Load the user-defined schema
            possible_head_entity_types = docre_config["possible_head_entity_types"]
            possible_tail_entity_types = docre_config["possible_tail_entity_types"]
            vocab_relation: dict[str, int] = {
                rel: rel_i
                for rel_i, rel in enumerate(docre_config["relations"])
            }
            rel_meta_info: dict[str, dict[str, str]] = docre_config["rel_meta_info"]
            entity_dict_path = docre_config.get("entity_dict_path", None)

            # Instantiate the LLM-based DocRE component with the user-defined schema
            docre = LLMDocRE(
                model=llm,
                prompt_template_name_or_path=docre_config["prompt_template_name_or_path"],
                knowledge_base_name=docre_config["knowledge_base_name"],
                mention_style=docre_config["mention_style"],
                with_span_annotation=docre_config["with_span_annotation"],
                possible_head_entity_types=possible_head_entity_types,
                possible_tail_entity_types=possible_tail_entity_types,
                vocab_relation=vocab_relation,
                rel_meta_info=rel_meta_info,
                entity_dict_path=entity_dict_path,
            )
    else:
        raise ValueError(f"Unknown DocRE method: {docre_config['method_name']}")

    return docre, loaded_llm_map


def instantiate_llm(
    config: dict[str, Any],
    loaded_llm_map: dict[str, BaseLLM]
) -> tuple[BaseLLM, dict[str, BaseLLM]]:

    # Create a key for reusing the same LLM instance
    key_parts = [
        config["llm_provider"],
        config["llm_model_name"],
        str(config["llm_max_new_tokens"]),
    ]
    if config["llm_provider"] == "hf":
        key_parts.append(str(config["llm_quantization_bits"]))
    key = "___".join(key_parts)

    # If the key is found, reuse the corresponding LLM
    if key in loaded_llm_map:
        return loaded_llm_map[key], loaded_llm_map

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

    # Add the key-LLM record to the dict
    loaded_llm_map[key] = llm

    return llm, loaded_llm_map


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
    parser.add_argument("--input_documents", type=str, required=True)

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    # Evaluation
    parser.add_argument("--do_evaluation", action="store_true")
    parser.add_argument("--gold", type=str, default=None)

    args = parser.parse_args()

    main(args)
