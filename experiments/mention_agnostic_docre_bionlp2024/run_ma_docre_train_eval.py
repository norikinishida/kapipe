import argparse
from collections import defaultdict
import logging
import os
import sys

import numpy as np
import pandas as pd
import tabulate
import torch
from tqdm import tqdm
import transformers

from kapipe import utils
from kapipe.docre import (
    MAATLOP, MAATLOPTrainer,
    MAQA, MAQATrainer
)
from kapipe.utils import StopWatch


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
    dataset_name = args.dataset_name
    train_documents_path = args.train_documents
    dev_documents_path = args.dev_documents
    test_documents_path = args.test_documents
    entity_dict_path = args.entity_dict

    # Output Path
    results_dir = args.results_dir
    prefix = args.prefix
    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()
        args.prefix = prefix

    # Action
    actiontype = args.actiontype

    assert method_name in ["ma_atlop", "ma_qa"]
    assert dataset_name in ["cdr", "hoip"]
    if entity_dict_path is None:
        raise ValueError("--entity_dict is required for mention-agnostic DocRE")
    assert actiontype in ["train", "evaluate"]

    ##################
    # Logging Setup
    ##################

    # Set base output path
    base_output_path = os.path.join(
        results_dir,
        "ma_docre",
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

    # Create vocabulary of relation labels
    if dataset_name == "cdr":
        vocab_relation = get_vocab_relation_for_cdr(method_name=method_name)
    elif dataset_name == "hoip":
        vocab_relation = get_vocab_relation_for_hoip(method_name=method_name)
    else:
        vocab_relation = get_vocab_relation(
            documents_list=[
                train_documents,
                dev_documents,
                test_documents
            ],
            method_name=method_name
        )

    # Create vocabulary of answer labels for MA-QA
    if method_name == "ma_qa":
        vocab_answer = get_vocab_answer()
 
    # Show statistics
    show_docre_documents_statistics(
        documents=train_documents,
        vocab_relation=vocab_relation,
        with_supervision=True,
        title="Training"
    )
    show_docre_documents_statistics(
        documents=dev_documents,
        vocab_relation=vocab_relation,
        with_supervision=True,
        title="Development"
    )
    show_docre_documents_statistics(
        documents=test_documents,
        vocab_relation=vocab_relation,
        with_supervision=True,
        title="Test"
    )

    ##################
    # Method Instantiation
    ##################

    # Load the experiment configuration
    config = utils.get_hocon_config(config_path=config_path, config_name=config_name)

    # Save the experiment configuration to the output path
    utils.write_json(os.path.join(base_output_path, "config.json"), config)

    # Instantiate the DocRE component
    if method_name == "ma_atlop":
        # Instantiate the trainer (evaluator)
        trainer = MAATLOPTrainer(base_output_path=base_output_path)

        if actiontype == "train":
            # Instantiate the MAATLOP-based DocRE component
            extractor = MAATLOP(
                **config,
                vocab_relation=vocab_relation,
                entity_dict_path=entity_dict_path
            )
        else:
            # Load the MAATLOP-based DocRE component from the snapshot
            extractor = MAATLOP.from_snapshot(
                snapshot_path=trainer.paths["snapshot_path"]
            )

    elif method_name == "ma_qa":
        # Instantiate the trainer (evaluator)
        trainer = MAQATrainer(base_output_path=base_output_path)

        if actiontype == "train":
            # Instantiate the MAQA-based DocRE component
            extractor = MAQA(
                **config,
                vocab_answer=vocab_answer,
                entity_dict_path=entity_dict_path
            )
        else:
            # Load the MAQA-based DocRE component from the snapshot
            extractor = MAQA.from_snapshot(
                snapshot_path=trainer.paths["snapshot_path"]
            )

        # Set the preprocessor for the current MAQATrainer interface
        extractor.preprocessor = extractor.model.preprocessor
    
    else:
        raise ValueError(f"Unknown method: {method_name}")

    ##################
    # Method Execution
    ##################

    if method_name in ["ma_atlop", "ma_qa"]:

        # Set up the datasets for evaluation
        trainer.setup_dataset(
            extractor=extractor,
            documents=train_documents,
            split="train",
            with_gold_annotations=True
        )
        trainer.setup_dataset(
            extractor=extractor,
            documents=dev_documents,
            split="dev",
            with_gold_annotations=True
        )
        trainer.setup_dataset(
            extractor=extractor,
            documents=test_documents,
            split="test",
            with_gold_annotations=True
        )

        if actiontype == "train":
            # Train the extractor
            trainer.train(
                extractor=extractor,
                train_documents=train_documents,
                dev_documents=dev_documents,
                supplemental_info={},
                **config,
            )

        if actiontype == "evaluate":
            # Evaluate the extractor on the datasets
            trainer.evaluate(
                extractor=extractor,
                documents=dev_documents,
                split="dev",
                supplemental_info={},
                #
                skip_intra_inter=True,
                skip_ign=True,
            )
            trainer.evaluate(
                extractor=extractor,
                documents=test_documents,
                split="test",
                supplemental_info={},
                #
                skip_intra_inter=True,
                skip_ign=True,
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


def get_vocab_relation(documents_list, method_name):
    relations = set()
    for documents in documents_list:
        for document in documents:
            for triple in document["relations"]:
                relations.add(triple["relation"])
    relations = sorted(list(relations))
    if method_name in ["ma_atlop"]:
        relations = ["NO-REL"] + relations
    vocab_relation = {rel: rel_id for rel_id, rel in enumerate(relations)}
    return vocab_relation


def get_vocab_relation_for_cdr(method_name):
    relations = ["CID"] # CDR contains only one relationship
    if method_name in ["ma_atlop"]:
        relations = ["NO-REL"] + relations
    vocab_relation = {rel: rel_id for rel_id, rel in enumerate(relations)}
    return vocab_relation


def get_vocab_relation_for_hoip(method_name):
    # relations = [
    #     "has result", "has part", "has molecular reaction", "part of"
    # ]
    relations = ["has result", "has part"]
    if method_name in ["ma_atlop"]:
        relations = ["NO-REL"] + relations
    vocab_relation = {rel: rel_id for rel_id, rel in enumerate(relations)}
    return vocab_relation


def get_vocab_answer():
    answers = ["No", "Yes"]
    vocab_answer = {ans: ans_id for ans_id, ans in enumerate(answers)}
    return vocab_answer


def show_docre_documents_statistics(
    documents,
    vocab_relation,
    with_supervision,
    title
):
    """Show DocRE documents statistics

    Parameters
    ----------
    documents : list[Document]
    vocab_relation : Dict[str, int]
    with_supervision : bool
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
        - Number of pairs
            - Average number of pairs per document
        - Number of positive pairs
            - Average number of positive pairs per document
        - Number of positive pairs for each relation class
        - Number of negative pairs (triples)
            - Average number of negative pairs (triples) per document
    """
    if not "NO-REL" in vocab_relation:
        tmp = {"NO-REL": 0}
        for k, v in vocab_relation.items():
            tmp[k] = v + 1
        vocab_relation = tmp

    ivocab_relation = {i:l for l, i in vocab_relation.items()}

    n_documents = len(documents)
    n_sentences_list = []
    n_words_list = []

    n_mentions_list = []
    n_mentions_dict = defaultdict(int)

    n_mentions2_list = []
    n_mentions2_dict = defaultdict(int)

    n_entities_list = []
    n_entities_dict = defaultdict(int)

    n_pairs_list = []
    n_positive_pairs_list = []
    n_negative_pairs_list = []
    # n_positive_pairs_dict = defaultdict(int)
    n_positive_pairs_dict = {r: 0 for r in vocab_relation.keys()}

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

        (
            pair_head_entity_indices,
            pair_tail_entity_indices,
            pair_gold_relation_labels
        ) = get_docre_pairs(
            document=doc,
            vocab_relation=vocab_relation,
            with_supervision=with_supervision,
            possible_head_entity_types=None,
            possible_tail_entity_types=None
        )

        n_pairs_list.append(len(pair_head_entity_indices))
        if with_supervision:
            n_positive_pairs = 0
            n_negative_pairs = 0
            for labels in pair_gold_relation_labels:
                if labels[0] == 1:
                    n_negative_pairs += 1
                else:
                    n_positive_pairs += 1
            n_positive_pairs_list.append(n_positive_pairs)
            n_negative_pairs_list.append(n_negative_pairs)
        else:
            n_positive_pairs_list.append(0)
            n_negative_pairs_list.append(0)

        # Class-wise
        if with_supervision:
            count = 0
            for labels in pair_gold_relation_labels:
                for l_i, l in enumerate(labels):
                    if l_i == 0:
                        continue
                    if l == 1:
                        n_positive_pairs_dict[ivocab_relation[l_i]] += 1
                        count += 1
            # assert count == len(doc["relations"]), (count, len(doc["relations"]))

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

    results["Number of pairs"] = get_statistics_text(n_pairs_list)
    results["Number of positive pairs"] = get_statistics_text(n_positive_pairs_list)
    for key, value in n_positive_pairs_dict.items():
        results[f"\tNumber of positive pairs for {key}"] = value
    results["Number of negative pairs"] = get_statistics_text(n_negative_pairs_list)

    table = {}
    table[title] = results.keys()
    table["Statistics"] = results.values()
    df = pd.DataFrame.from_dict(table)
    logging.info("\n" + tabulate.tabulate(df, headers="keys", tablefmt="psql", floatfmt=".1f"))


def get_docre_pairs(
    document,
    vocab_relation,
    with_supervision,
    possible_head_entity_types=None,
    possible_tail_entity_types=None
):
    not_include_entity_pairs = None
    if "not_include_pairs" in document:
        # list[tuple[EntityIndex, EntityIndex]]
        epairs = [
            (epair["arg1"], epair["arg2"])
            for epair in document["not_include_pairs"]
        ]
        not_include_entity_pairs \
            = [(e1,e2) for e1,e2 in epairs] + [(e2,e1) for e1,e2 in epairs]

    pair_head_entity_indices = [] # list[int]
    pair_tail_entity_indices = [] # list[int]
    pair_gold_relation_labels = [] # list[list[int]]

    for head_entity_i in range(len(document["entities"])):
        for tail_entity_i in range(len(document["entities"])):
            # Skip diagonal
            if head_entity_i == tail_entity_i:
                continue

            # Skip based on entity types if specified
            # e.g, Skip chemical-chemical, disease-disease,
            #      and disease-chemical pairs for CDR.
            if (
                (possible_head_entity_types is not None)
                and
                (possible_tail_entity_types is not None)
            ):
                head_entity_type = document["entities"][head_entity_i]["entity_type"]
                tail_entity_type = document["entities"][tail_entity_i]["entity_type"]
                if not (
                    (head_entity_type in possible_head_entity_types)
                    and
                    (tail_entity_type in possible_tail_entity_types)
                ):
                    continue

            # Skip "not_include" pairs if specified
            if not_include_entity_pairs is not None:
                if (head_entity_i, tail_entity_i) \
                        in not_include_entity_pairs:
                    continue

            pair_head_entity_indices.append(head_entity_i)
            pair_tail_entity_indices.append(tail_entity_i)

            if with_supervision:
                rels = find_relations(
                    arg1=head_entity_i,
                    arg2=tail_entity_i,
                    relations=document["relations"]
                )
                multilabel_positive_indicators \
                    = [0] * len(vocab_relation)
                if len(rels) == 0:
                    # Found no gold relation for this entity pair
                    multilabel_positive_indicators[0] = 1
                else:
                    for rel in rels:
                        rel_id = vocab_relation[rel]
                        multilabel_positive_indicators[rel_id] = 1
                pair_gold_relation_labels.append(
                    multilabel_positive_indicators
                )
            else:
                pair_gold_relation_labels.append(None)

    return (
        pair_head_entity_indices,
        pair_tail_entity_indices,
        pair_gold_relation_labels
    )


def find_relations(arg1, arg2, relations):
    rels = [] # list[str]
    for triple in relations:
        if triple["arg1"] == arg1 and triple["arg2"] == arg2:
            rels.append(triple["relation"])
    return rels


def get_statistics_text(xs):
    if len(xs) == 0:
        sum_ = mean_ = max_ = min_ = 0
    else:
        sum_ = np.sum(xs)
        mean_ = np.mean(xs)
        max_ = np.max(xs)
        min_ = np.min(xs)
    return f"Total: {sum_} / Average per instance: {mean_} / Max: {max_} / Min: {min_}"


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
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--train_documents", type=str, required=True)
    parser.add_argument("--dev_documents", type=str, required=True)
    parser.add_argument("--test_documents", type=str, required=True)
    parser.add_argument("--entity_dict", type=str, default=None)
    
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

