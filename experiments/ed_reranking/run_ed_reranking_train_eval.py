import argparse
from collections import defaultdict
import copy
import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import tabulate
import torch
from tqdm import tqdm
import transformers

from kapipe import evaluation
from kapipe import utils
from kapipe.ed_reranking import (
    BlinkCrossEncoder, BlinkCrossEncoderTrainer,
    LLMED, LLMEDTrainer
)
from kapipe.llms import HuggingFaceLLM, OpenAILLM
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
    train_documents_path = args.train_documents
    dev_documents_path = args.dev_documents
    test_documents_path = args.test_documents

    train_candidate_entities_path = args.train_candidate_entities
    dev_candidate_entities_path = args.dev_candidate_entities
    test_candidate_entities_path = args.test_candidate_entities

    entity_dict_path = args.entity_dict

    n_demonstrations = args.n_demonstrations

    # Output Path
    results_dir = args.results_dir
    prefix = args.prefix
    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()
        args.prefix = prefix

    # Action
    actiontype = args.actiontype

    assert method_name in ["blink_cross_encoder", "llm_ed"]
    assert actiontype in ["train", "evaluate", "check_prompt"]

    ##################
    # Logging Setup
    ##################

    # Set base output path
    base_output_path = os.path.join(
        results_dir,
        "ed_reranking",
        method_name,
        config_name,
        prefix
    )
    utils.mkdir(base_output_path)

    # Set logger
    if actiontype == "train":
        set_logger(
            os.path.join(base_output_path, "training.log"),
            # overwrite=True
        )
    elif actiontype == "evaluate":
        set_logger(
            os.path.join(base_output_path, "evaluation.log"),
            # overwrite=True
        )

    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))

    ##################
    # Data
    ##################

    # Load documents
    train_documents = utils.read_json(train_documents_path)
    dev_documents = utils.read_json(dev_documents_path)
    test_documents = utils.read_json(test_documents_path)

    # Load candidate entities
    train_candidate_entities = utils.read_json(train_candidate_entities_path)
    dev_candidate_entities = utils.read_json(dev_candidate_entities_path)
    test_candidate_entities = utils.read_json(test_candidate_entities_path)

    # Load demonstrations for LLM-based ED-Reranking
    if method_name == "llm_ed":
        demonstration_documents, demonstration_candidate_entities = (
            create_demonstrations(
                train_documents=train_documents,
                train_candidate_entities=train_candidate_entities,
                n_demonstrations=n_demonstrations,
            )
        )

    # Show statistics
    show_ed_documents_statistics(
        documents=train_documents,
        title="Training"
    )
    show_ed_documents_statistics(
        documents=dev_documents,
        title="Development"
    )
    show_ed_documents_statistics(
        documents=test_documents,
        title="Test"
    )    

    ##################
    # Method Instantiation
    ##################

    # Load the experiment configuration
    config = utils.get_hocon_config(config_path=config_path, config_name=config_name)

    # Save the experiment configuration to the output path
    utils.write_json(os.path.join(base_output_path, "config.json"), config)

    # Instantiate the ED-Reranking component
    if method_name == "blink_cross_encoder":
        # Instantiate the trainer (evaluator)
        trainer = BlinkCrossEncoderTrainer(base_output_path=base_output_path)

        if actiontype == "train":
            # Instantiate the Blink Bi-Encoder reranker
            reranker = BlinkCrossEncoder(
                **config,
                entity_dict_path=entity_dict_path
            )
        else:
            # Load the Blink Bi-Encoder reranker from the snapshot
            reranker = BlinkCrossEncoder.from_snapshot(
                snapshot_path=trainer.paths["snapshot_path"]
            )

    elif method_name == "llm_ed":
        assert actiontype != "train"

        # Instantiate the trainer (evaluator)
        trainer = LLMEDTrainer(base_output_path=base_output_path)

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

        # Instantiate the LLM-based ED reranker
        reranker = LLMED(
            model=model,
            **config,
            entity_dict_path=entity_dict_path,
            demonstration_documents=demonstration_documents,
            demonstration_candidate_entities=demonstration_candidate_entities,
        )

    else:
        raise ValueError(f"Unknown method: {method_name}")

    ##################
    # Method Execution
    ##################

    if method_name == "blink_cross_encoder":

        # Remove out-of-kb mentions in the training dataset
        (
            processed_train_documents,
            processed_train_candidate_entities
        ) = remove_out_of_kb_mentions(
            train_documents=train_documents,
            train_candidate_entities=train_candidate_entities,
            entity_dict=reranker.entity_dict
        )
        show_ed_documents_statistics(
            documents=processed_train_documents,
            title="Training after Out-of-Kb Removal"
        )

        # Evaluate the candidate entities for the training dataset
        logging.info(utils.pretty_format_dict(
            evaluation.ed.recall_at_k(
                pred_path=processed_train_candidate_entities,
                gold_path=processed_train_documents,
                inkb=False
            )
        ))

        # Add/move gold entities in the candidate entities for the training dataset
        processed_train_candidate_entities = add_or_move_gold_entity_in_candidates(
            documents=processed_train_documents,
            candidate_entities=processed_train_candidate_entities
        )

        # Re-evaluate the candidate entities for the training dataset
        logging.info(utils.pretty_format_dict(
            evaluation.ed.recall_at_k(
                pred_path=processed_train_candidate_entities,
                gold_path=processed_train_documents,
                inkb=False
            )
        ))

        # Set up the datasets for evaluation
        trainer.setup_dataset(
            reranker=reranker,
            documents=dev_documents,
            candidate_entities=dev_candidate_entities,
            split="dev"
        )
        trainer.setup_dataset(
            reranker=reranker,
            documents=test_documents,
            candidate_entities=test_candidate_entities,
            split="test"
        )

        # Evaluate the candidate entities for the development dataset
        if (
            ".pred_candidate_entities." in dev_candidate_entities_path
            and
            os.path.exists(dev_candidate_entities_path.replace(
                ".pred_candidate_entities.", ".pred."
            ))
        ):
            logging.info(utils.pretty_format_dict(
                evaluation.ed.recall_at_k(
                    pred_path=dev_candidate_entities,
                    gold_path=trainer.paths["dev_gold_path"],
                    inkb=True
                ) | evaluation.ed.accuracy(
                    pred_path=dev_candidate_entities_path.replace(
                        ".pred_candidate_entities.", ".pred."
                    ),
                    gold_path=trainer.paths["dev_gold_path"],
                    inkb=True
                ) | evaluation.ed.fscore(
                    pred_path=dev_candidate_entities_path.replace(
                        ".pred_candidate_entities.", ".pred."
                    ),
                    gold_path=trainer.paths["dev_gold_path"],
                    inkb=True
                )
            ))

        # Evaluate the candidate entities for the test dataset
        if (
            ".pred_candidate_entities." in test_candidate_entities_path
            and
            os.path.exists(test_candidate_entities_path.replace(
                ".pred_candidate_entities.", ".pred."
            ))
        ):
            logging.info(utils.pretty_format_dict(
                evaluation.ed.recall_at_k(
                    pred_path=test_candidate_entities,
                    gold_path=trainer.paths["test_gold_path"],
                    inkb=True
                ) | evaluation.ed.accuracy(
                    pred_path=test_candidate_entities_path.replace(
                        ".pred_candidate_entities.", ".pred."
                    ),
                    gold_path=trainer.paths["test_gold_path"],
                    inkb=True
                ) | evaluation.ed.fscore(
                    pred_path=test_candidate_entities_path.replace(
                        ".pred_candidate_entities.", ".pred."
                    ),
                    gold_path=trainer.paths["test_gold_path"],
                    inkb=True
                )
            )) 

        if actiontype == "train":
            # Train the reranker
            trainer.train(
                reranker=reranker,
                train_documents=processed_train_documents,
                train_candidate_entities=processed_train_candidate_entities,
                dev_documents=dev_documents,
                dev_candidate_entities=dev_candidate_entities,
                **config,
            )

        if actiontype == "evaluate":
            # Evaluate the reranker on the datasets
            trainer.evaluate(
                reranker=reranker,
                documents=dev_documents,
                candidate_entities=dev_candidate_entities,
                split="dev"
            )
            trainer.evaluate(
                reranker=reranker,
                documents=test_documents,
                candidate_entities=test_candidate_entities,
                split="test"
            )

    if method_name == "llm_ed":

        if actiontype == "check_prompt":
            # Show prompts
            with torch.no_grad():
                out_path = os.path.join(base_output_path, "output.txt")
                with open(out_path, "w") as f:
                    for i, (document, cands) in enumerate(zip(
                        dev_documents, dev_candidate_entities
                    )):
                        doc_key = document["doc_key"]
                        logging.info(f"Processing {doc_key}")
                        document = reranker.rerank(
                            document=document,
                            candidate_entities_for_doc=cands
                        )
                        f.write(f"--- DOC_KEY ({doc_key}) ---\n\n")
                        f.write("Prompt:\n")
                        f.write(document["ed_prompt"] + "\n\n")
                        f.write("-----\n")
                        f.write("Generated Text:\n")
                        f.write(document["ed_generated_text"] + "\n\n")
                        f.write("-----\n")
                        f.write("Parsed mention-entity pairs:\n")
                        for m in document["mentions"]:
                            f.write(f"{m}\n")
                        f.write("-----\n")
                        f.flush()
                        if i > 5:
                            break
                return

        # Set up the datasets for evaluation
        trainer.setup_dataset(
            reranker=reranker,
            documents=dev_documents,
            candidate_entities=dev_candidate_entities,
            split="dev"
        )
        trainer.setup_dataset(
            reranker=reranker,
            documents=test_documents,
            candidate_entities=test_candidate_entities,
            split="test"
        )

        # Evaluate the candidate entities for the development dataset
        if (
            ".pred_candidate_entities." in dev_candidate_entities_path
            and
            os.path.exists(dev_candidate_entities_path.replace(
                ".pred_candidate_entities.", ".pred."
            ))
        ):
            logging.info(utils.pretty_format_dict(
                evaluation.ed.recall_at_k(
                    pred_path=dev_candidate_entities,
                    gold_path=trainer.paths["dev_gold_path"],
                    inkb=True
                ) | evaluation.ed.accuracy(
                    pred_path=dev_candidate_entities_path.replace(
                        ".pred_candidate_entities.", ".pred."
                    ),
                    gold_path=trainer.paths["dev_gold_path"],
                    inkb=True
                ) | evaluation.ed.fscore(
                    pred_path=dev_candidate_entities_path.replace(
                        ".pred_candidate_entities.", ".pred."
                    ),
                    gold_path=trainer.paths["dev_gold_path"],
                    inkb=True
                )
            ))

        # Evaluate the candidate entities for the test dataset
        if (
            ".pred_candidate_entities." in test_candidate_entities_path
            and
            os.path.exists(test_candidate_entities_path.replace(
                ".pred_candidate_entities.", ".pred."
            ))
        ):
            logging.info(utils.pretty_format_dict(
                evaluation.ed.recall_at_k(
                    pred_path=test_candidate_entities,
                    gold_path=trainer.paths["test_gold_path"],
                    inkb=True
                ) | evaluation.ed.accuracy(
                    pred_path=test_candidate_entities_path.replace(
                        ".pred_candidate_entities.", ".pred."
                    ),
                    gold_path=trainer.paths["test_gold_path"],
                    inkb=True
                ) | evaluation.ed.fscore(
                    pred_path=test_candidate_entities_path.replace(
                        ".pred_candidate_entities.", ".pred."
                    ),
                    gold_path=trainer.paths["test_gold_path"],
                    inkb=True
                )
            )) 

        # Save the configurations of the reranker
        trainer.save_reranker(reranker=reranker)

        if actiontype == "evaluate":
            # Evaluate the reranker on the datasets
            trainer.evaluate(
                reranker=reranker,
                documents=dev_documents,
                candidate_entities=dev_candidate_entities,
                split="dev"
            )
            trainer.evaluate(
                reranker=reranker,
                documents=test_documents,
                candidate_entities=test_candidate_entities,
                split="test"
            )

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


def pop_logger_handler():
    root_logger = logging.getLogger()
    assert len(root_logger.handlers) > 1
    handler = root_logger.handlers.pop()
    root_logger.removeHandler(handler)
    handler.close()
    logging.info(f"Removed {handler} from the root logger {root_logger}.")


def create_demonstrations(
    train_documents: list[dict],
    train_candidate_entities: list[dict],
    n_demonstrations: int,
) -> tuple[list[dict], list[dict]]:
    # Keep scored document-candidate pairs
    scored_items = []

    # Score each training document with its retrieval candidates
    for document, candidate_entities_for_doc in zip(
        train_documents,
        train_candidate_entities,
    ):
        # Check document alignment
        assert document["doc_key"] == candidate_entities_for_doc["doc_key"]

        # Check mention-candidate alignment
        assert len(document["mentions"]) == len(
            candidate_entities_for_doc["candidate_entities"]
        )

        # Count mentions whose gold entity appears in the retrieved candidates
        n_covered_mentions = 0
        for mention, candidate_entities_for_mention in zip(
            document["mentions"],
            candidate_entities_for_doc["candidate_entities"],
        ):
            # Collect candidate entity IDs
            candidate_entity_ids = {
                candidate_entity["entity_id"]
                for candidate_entity in candidate_entities_for_mention
            }

            # Count usable ED demonstration mentions
            if mention["entity_id"] in candidate_entity_ids:
                n_covered_mentions += 1

        # Keep documents with at least one usable ED demonstration mention
        if n_covered_mentions > 0:
            scored_items.append((
                n_covered_mentions,
                len(document["mentions"]),
                document,
                candidate_entities_for_doc,
            ))

    # Prefer documents with more covered mentions
    scored_items = sorted(
        scored_items,
        key=lambda item: (-item[0], -item[1]),
    )

    # Select top-k aligned document-candidate pairs
    selected_items = scored_items[:n_demonstrations]

    # Extract demonstration documents
    demonstration_documents = [item[2] for item in selected_items]

    # Extract aligned demonstration candidate entities
    demonstration_candidate_entities = [item[3] for item in selected_items]

    return demonstration_documents, demonstration_candidate_entities


def show_ed_documents_statistics(
    documents,
    title
):
    """Show ED documents statistics

    Parameters
    ----------
    documents : list[Document]
    title : str

    Summarize the following statistics
        - Number of documents
        - Number of sentences
            - Average number of sentences per document
        - Number of words
            - Average number of words per document
        - Number of mentions (without/with redundancy)
            - Average number of mentions (without/with redundancy) per document
        - Number of mentions (without/with redundancy) for each entity type
        - Number of entities
            - Average number of entities per document
        - Number of entites for each entity type
    """
    n_documents = len(documents)
    n_sentences_list = []
    n_words_list = []

    n_mentions_list = []
    n_mentions_dict = defaultdict(int)

    n_mentions2_list = []
    n_mentions2_dict = defaultdict(int)

    n_entities_list = []
    n_entities_dict = defaultdict(int)

    for doc in tqdm(documents):
        n_sentences_list.append(len(doc["sentences"]))

        n_words_list.append(len(utils.flatten_lists(
            [s.split() for s in doc["sentences"]]
        )))

        n_mentions_list.append(len(doc["mentions"]))
        for mention in doc["mentions"]:
            etype = mention["entity_type"]
            n_mentions_dict[etype] += 1

        n_mentions2_list.append(sum(
            [len(e["mention_indices"]) for e in doc["entities"]]
        ))
        for entity in doc["entities"]:
            ms = entity["mention_indices"]
            etype = entity["entity_type"]
            n_mentions2_dict[etype] += len(ms)

        n_entities_list.append(len(doc["entities"]))
        for entity in doc["entities"]:
            etype = entity["entity_type"]
            n_entities_dict[etype] += 1

    results = {}
    results["Number of documents"] = n_documents
    results["Number of sentences"] = get_statistics_text(n_sentences_list)
    results["Number of words"] = get_statistics_text(n_words_list)

    results["Number of mentions"] = get_statistics_text(n_mentions_list)
    for key, value in sorted(list(n_mentions_dict.items()), key=lambda tpl: tpl[0]):
        results[f"\tNumber of mentions for {key}"] = value

    results["Number of mentions (with redundancy)"] = get_statistics_text(n_mentions2_list)
    for key, value in sorted(list(n_mentions2_dict.items()), key=lambda tpl: tpl[0]):
        results[f"\tNumber of mentions (with redundancy) for {key}"] = value

    results["Number of entities"] = get_statistics_text(n_entities_list)
    for key, value in sorted(list(n_entities_dict.items()), key=lambda tpl: tpl[0]):
        results[f"\tNumber of entities for {key}"] = value

    table = {}
    table[title] = results.keys()
    table["Statistics"] = results.values()
    df = pd.DataFrame.from_dict(table)
    logging.info("\n" + tabulate.tabulate(df, headers="keys", tablefmt="psql", floatfmt=".1f"))


def get_statistics_text(xs):
    if len(xs) == 0:
        sum_ = mean_ = max_ = min_ = 0
    else:
        sum_ = np.sum(xs)
        mean_ = np.mean(xs)
        max_ = np.max(xs)
        min_ = np.min(xs)
    return f"Total: {sum_} / Average per instance: {mean_} / Max: {max_} / Min: {min_}"


def remove_out_of_kb_mentions(
    train_documents,
    train_candidate_entities,
    entity_dict
):
    new_train_documents = []
    new_train_candidate_entities = []

    kb_entity_ids = set(list(entity_dict.keys()))

    n_prev_mentions = 0
    n_new_mentions = 0
    n_prev_entities = 0
    n_new_entities = 0

    for doc_i in tqdm(
        range(len(train_documents)),
        desc="removing out-of-kb mentions"
    ):
        # Copy data
        doc = copy.deepcopy(train_documents[doc_i])
        candidate_entities_for_doc = copy.deepcopy(train_candidate_entities[doc_i])

        assert doc["doc_key"] == candidate_entities_for_doc["doc_key"]

        # Remove out-of-kb mentions and their candidate entities
        n_prev_mentions += len(doc["mentions"])
        candidate_entities_for_mentions = [
            cs for cs, m in zip(
                candidate_entities_for_doc["candidate_entities"], doc["mentions"]
            )
            if m["entity_id"] in kb_entity_ids
        ]
        mentions = [m for m in doc["mentions"] if m["entity_id"] in kb_entity_ids]
        n_new_mentions += len(mentions)

        # Reset mentions and candidate entities
        doc["mentions"] = mentions
        candidate_entities_for_doc["candidate_entities"] = candidate_entities_for_mentions

        assert len(mentions) == len(candidate_entities_for_mentions)
        if len(mentions) == 0:
            continue

        # Re-aggregate mentions for entities
        n_prev_entities += len(doc["entities"])
        entities = utils.aggregate_mentions_to_entities(
            document=doc,
            mentions=mentions
        )
        n_new_entities += len(entities)

        # Reset entities
        doc["entities"] = entities
 
        # new_train_documents[doc_i] = doc
        # new_train_candidate_entities[doc_i] = candidate_entities
        new_train_documents.append(doc)
        new_train_candidate_entities.append(candidate_entities_for_doc)

    logging.info(f"Removed {n_prev_mentions - n_new_mentions}({n_prev_entities - n_new_entities}) out-of-kb mentions (entities) in the training dataset: {n_prev_mentions} ({n_prev_entities}) -> {n_new_mentions} ({n_new_entities})")
    logging.info(f"Removed {len(train_documents) - len(new_train_documents)} documents with only out-of-kb mentions/entities in the training dataset: {len(train_documents)} -> {len(new_train_documents)}")

    return new_train_documents, new_train_candidate_entities


def add_or_move_gold_entity_in_candidates(
    documents,
    candidate_entities
):
    result_candidate_entities = []
    count_add = 0
    count_move = 0
    count_mentions = 0
    top_k = 0
    for document, cands_for_doc in zip(documents, candidate_entities):
        result_cands_for_mentions = []

        mentions = document["mentions"]
        cands_for_mentions = cands_for_doc["candidate_entities"]
        assert len(mentions) == len(cands_for_mentions)

        original_gold_entity_rank_list = [] # list[float]

        for mention, cands_for_mention in zip(mentions, cands_for_mentions):
            top_k = len(cands_for_mention)

            # Remove the gold entity in the candidates
            gold_entity_id = mention["entity_id"]
            cand_entity_ids = [c["entity_id"] for c in cands_for_mention]
            if gold_entity_id in cand_entity_ids:
                if gold_entity_id != cand_entity_ids[0]:
                    count_move += 1
                index = cand_entity_ids.index(gold_entity_id)
                gold_canonical_name = cands_for_mention[index]["canonical_name"]
                result_cands_for_mention = cands_for_mention[:index] + cands_for_mention[index+1:]
                original_gold_entity_rank_list.append(index + random.random())
            else:
                count_add += 1
                gold_canonical_name = None # Not used
                result_cands_for_mention = cands_for_mention[:-1]
                original_gold_entity_rank_list.append(top_k+1000 + random.random()) # Should be larger than top_k
            count_mentions += 1

            # Append the gold entity to the top of the candidate list
            result_cands_for_mention = [{
                "entity_id": gold_entity_id,
                "canonical_name": gold_canonical_name,
                "score": 1000000.0
            }] + result_cands_for_mention
            result_cands_for_mentions.append(result_cands_for_mention)

        assert len(document["mentions"]) == len(result_cands_for_mentions) == len(original_gold_entity_rank_list)

        result_cands_for_doc = {
            "doc_key": document["doc_key"],
            "candidate_entities": result_cands_for_mentions,
            "original_gold_entity_rank_list": original_gold_entity_rank_list
        }
        result_candidate_entities.append(result_cands_for_doc)

    logging.info(f"Added (or changed the position of) gold entities to the list of top-{top_k} candidate entities for {count_add} ({count_move}) / {count_mentions} mentions")
    return result_candidate_entities


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
    parser.add_argument("--train_documents", type=str, required=True)
    parser.add_argument("--dev_documents", type=str, required=True)
    parser.add_argument("--test_documents", type=str, required=True)

    parser.add_argument("--train_candidate_entities", type=str, required=True)
    parser.add_argument("--dev_candidate_entities", type=str, required=True)
    parser.add_argument("--test_candidate_entities", type=str, required=True)

    parser.add_argument("--entity_dict", type=str, required=True)

    parser.add_argument("--n_demonstrations", type=int, default=3)

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    # Action
    parser.add_argument("--actiontype", type=str, required=True)

    args = parser.parse_args()

    if args.actiontype == "train_and_evaluate":
        # Training
        args.actiontype = "train"
        prefix = main(args=args)
        # Evaluation
        args.actiontype = "evaluate"
        args.prefix = prefix
        pop_logger_handler()
        main(args=args)
    else:
        # Training or Evaluation
        main(args=args)
        

