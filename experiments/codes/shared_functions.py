# from collections import defaultdict
import logging
import os
import random
import sys

import numpy as np
import torch
# import pandas as pd
# import tabulate
# from tqdm import tqdm

from kapipe import evaluation
from kapipe import utils


####################################
# Logging
####################################


def set_logger(filename, overwrite=False):
    """
    Parameters
    ----------
    filename: str
    overwrite: bool, default False
    """
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


####################################
# Random Seed Setting
####################################


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    logging.info("Set random seed: %d" % seed)


####################################
# Dataset
####################################


def evaluate_candidate_entity_retriever(
    trainer,
    path_candidate_entities,
    split
):
    scores = evaluation.ed.accuracy(
        pred_path=path_candidate_entities.replace(
            ".pred_candidate_entities.", ".pred."
        ),
        gold_path=trainer.paths[f"path_{split}_gold"],
        inkb=True
    )
    scores.update(evaluation.ed.fscore(
        pred_path=path_candidate_entities.replace(
            ".pred_candidate_entities.", ".pred."
        ),
        gold_path=trainer.paths[f"path_{split}_gold"],
        inkb=True
    ))
    logging.info(f"Evaluation results of candidate entity retriever for the {split} set:")
    logging.info(utils.pretty_format_dict(scores))


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
                result_cands_for_mention \
                    = cands_for_mention[:index] + cands_for_mention[index+1:]
            else:
                count_add += 1
                gold_canonical_name = None
                result_cands_for_mention = cands_for_mention[:-1]
            count_mentions += 1
            # Append the gold entity to the top of the candidate list
            result_cands_for_mention = [{
                "entity_id": gold_entity_id,
                "canonical_name": gold_canonical_name,
                "score": 1000000.0
            }] + result_cands_for_mention
            result_cands_for_mentions.append(result_cands_for_mention)
        result_cands_for_doc = {
            "doc_key": document["doc_key"],
            "candidate_entities": result_cands_for_mentions
        }
        result_candidate_entities.append(result_cands_for_doc)
    logging.info(f"Added (or changed the position of) gold entities to the list of top-{top_k} candidate entities for {count_add} ({count_move}) / {count_mentions} mentions")
    return result_candidate_entities


def get_docred_info(dataset_name):
    if dataset_name == "docred":
        train_file_name = "train_annotated.json"
        dev_file_name = "dev.json"
        test_file_name = None
    elif dataset_name == "redocred":
        train_file_name = "train_revised.json"
        dev_file_name = "dev_revised.json"
        test_file_name = "test_revised.json"
    else:
        raise Exception(f"Invalid dataset_name: {dataset_name}")
    return train_file_name, dev_file_name, test_file_name


# def show_ner_dataset_statistics(config, dataset, title):
#     """Show dataset statistics

#     Parameters
#     ----------
#     config: ConfigTree
#     dataset : numpy.ndarray
#     title: str

#     Summarize the following statistics
#         - Number of documents
#         - Number of sentences
#             - Average number of sentences per document
#         - Number of words
#             - Average number of words per document
#         - Number of subtokens
#             - Average number of subtokens per document
#         - Number of segments
#             - Average number of segments per document
#         - Number of mentions (without/with redundancy)
#             - Average number of mentions (without/with redundancy) per document
#         - Number of mentions (without/with redundancy) for each entity type
#     """
#     n_documents = len(dataset)
#     n_sentences_list = []
#     n_words_list = []
#     n_subtokens_list = []
#     n_segments_list = []

#     n_mentions_list = []
#     n_mentions_dict = defaultdict(int)

#     for data in tqdm(dataset):
#         n_sentences_list.append(len(data.sentences))
#         n_words_list.append(len(utils.flatten_lists(data.sentences)))
#         n_subtokens_list.append(len(utils.flatten_lists(data.segments)))
#         n_segments_list.append(len(data.segments))

#         n_mentions_list.append(len(data.mentions))
#         for span, name, etype in data.mentions:
#             n_mentions_dict[etype] += 1

#     results = {}
#     results["Number of documents"] = n_documents
#     results["Number of sentences"] = get_statistics_text(n_sentences_list)
#     results["Number of words"] = get_statistics_text(n_words_list)
#     results["Number of subtokens"] = get_statistics_text(n_subtokens_list)
#     results["Number of segments"] = get_statistics_text(n_segments_list)

#     results["Number of mentions"] = get_statistics_text(n_mentions_list)
#     for key, value in n_mentions_dict.items():
#         results[f"\tNumber of mentions for {key}"] = value

#     table = {}
#     table[title] = results.keys()
#     table["Statistics"] = results.values()
#     df = pd.DataFrame.from_dict(table)
#     logging.info("\n" + tabulate.tabulate(df, headers="keys", tablefmt="psql", floatfmt=".1f"))


# def show_el_dataset_statistics(config, dataset, title):
#     """Show dataset statistics

#     Parameters
#     ----------
#     config: ConfigTree
#     dataset : numpy.ndarray
#     title: str

#     Summarize the following statistics
#         - Number of documents
#         - Number of sentences
#             - Average number of sentences per document
#         - Number of words
#             - Average number of words per document
#         - Number of subtokens
#             - Average number of subtokens per document
#         - Number of segments
#             - Average number of segments per document
#         - Number of mentions (without/with redundancy)
#             - Average number of mentions (without/with redundancy) per document
#         - Number of mentions (without/with redundancy) for each entity type
#         - Number of entities
#             - Average number of entities per document
#         - Number of entites for each entity type
#         - Number of pairs
#             - Average number of pairs per document
#         - Number of positive pairs
#             - Average number of positive pairs per document
#         - Number of positive pairs for each relation class
#         - Number of negative pairs (triples)
#             - Average number of negative pairs (triples) per document
#     """
#     n_documents = len(dataset)
#     n_sentences_list = []
#     n_words_list = []
#     n_subtokens_list = []
#     n_segments_list = []

#     n_mentions_list = []
#     n_mentions_dict = defaultdict(int)
#     n_mentions2_list = []
#     n_mentions2_dict = defaultdict(int)

#     n_entities_list = []
#     n_entities_dict = defaultdict(int)

#     # n_pairs_list = []
#     # n_positive_pairs_list = []
#     # n_negative_pairs_list = []
#     # n_positive_pairs_dict = defaultdict(int)

#     for data in tqdm(dataset):
#         n_sentences_list.append(len(data.sentences))
#         n_words_list.append(len(utils.flatten_lists(data.sentences)))
#         n_subtokens_list.append(len(utils.flatten_lists(data.segments)))
#         n_segments_list.append(len(data.segments))

#         n_mentions_list.append(len(data.mentions))
#         for span, name, etype, eid in data.mentions:
#             n_mentions_dict[etype] += 1
#         n_mentions2_list.append(sum([len(ms) for ms, etype, eid in data.entities]))
#         for ms, etype, eid in data.entities:
#             n_mentions2_dict[etype] += len(ms)

#         n_entities_list.append(len(data.entities))
#         for ms, etype, eid in data.entities:
#             n_entities_dict[etype] += 1

#         # n_pairs_list.append(len(data.pair_head_entity_indices))
#         # if hasattr(data, "pair_gold_relation_labels"):
#         #     n_positive_pairs = 0
#         #     n_negative_pairs = 0
#         #     for labels in data.pair_gold_relation_labels:
#         #         if labels[0] == 1:
#         #             n_negative_pairs += 1
#         #         else:
#         #             n_positive_pairs += 1
#         #     n_positive_pairs_list.append(n_positive_pairs)
#         #     n_negative_pairs_list.append(n_negative_pairs)

#         #     # Class-wise
#         #     count = 0
#         #     for labels in data.pair_gold_relation_labels:
#         #         for l_i, l in enumerate(labels):
#         #             if l_i == 0:
#         #                 continue
#         #             if l == 1:
#         #                 n_positive_pairs_dict[ivocab_relation[l_i]] += 1
#         #                 count += 1
#         #     assert count == len(data.relations)
#         # else:
#         #     n_positive_pairs_list.append(0)
#         #     n_negative_pairs_list.append(0)

#     results = {}
#     results["Number of documents"] = n_documents
#     results["Number of sentences"] = get_statistics_text(n_sentences_list)
#     results["Number of words"] = get_statistics_text(n_words_list)
#     results["Number of subtokens"] = get_statistics_text(n_subtokens_list)
#     results["Number of segments"] = get_statistics_text(n_segments_list)

#     results["Number of mentions"] = get_statistics_text(n_mentions_list)
#     for key, value in n_mentions_dict.items():
#         results[f"\tNumber of mentions for {key}"] = value
#     results["Number of mentions (with redundancy)"] = get_statistics_text(n_mentions2_list)
#     for key, value in n_mentions2_dict.items():
#         results[f"\tNumber of mentions (with redundancy) for {key}"] = value

#     results["Number of entities"] = get_statistics_text(n_entities_list)
#     for key, value in n_entities_dict.items():
#         results[f"\tNumber of entities for {key}"] = value

#     # results["Number of pairs"] = get_statistics_text(n_pairs_list)
#     # results["Number of positive pairs"] = get_statistics_text(n_positive_pairs_list)
#     # for key, value in n_positive_pairs_dict.items():
#     #     results[f"\tNumber of positive pairs for {key}"] = value
#     # results["Number of netative pairs"] = get_statistics_text(n_negative_pairs_list)

#     table = {}
#     table[title] = results.keys()
#     table["Statistics"] = results.values()
#     df = pd.DataFrame.from_dict(table)
#     logging.info("\n" + tabulate.tabulate(df, headers="keys", tablefmt="psql", floatfmt=".1f"))


# def show_docre_dataset_statistics(config, dataset, vocab_relation, title):
#     """Show dataset statistics

#     Parameters
#     ----------
#     config: ConfigTree
#     dataset : numpy.ndarray
#     vocab_relation: Dict[str, int]
#     title: str

#     Summarize the following statistics
#         - Number of documents
#         - Number of sentences
#             - Average number of sentences per document
#         - Number of words
#             - Average number of words per document
#         - Number of subtokens
#             - Average number of subtokens per document
#         - Number of segments
#             - Average number of segments per document
#         - Number of mentions (without/with redundancy)
#             - Average number of mentions (without/with redundancy) per document
#         - Number of mentions (without/with redundancy) for each entity type
#         - Number of entities
#             - Average number of entities per document
#         - Number of entites for each entity type
#         - Number of pairs
#             - Average number of pairs per document
#         - Number of positive pairs
#             - Average number of positive pairs per document
#         - Number of positive pairs for each relation class
#         - Number of negative pairs (triples)
#             - Average number of negative pairs (triples) per document
#     """
#     ivocab_relation = {i:l for l, i in vocab_relation.items()}

#     n_documents = len(dataset)
#     n_sentences_list = []
#     n_words_list = []
#     n_subtokens_list = []
#     n_segments_list = []

#     n_mentions_list = []
#     n_mentions_dict = defaultdict(int)
#     n_mentions2_list = []
#     n_mentions2_dict = defaultdict(int)

#     n_entities_list = []
#     n_entities_dict = defaultdict(int)

#     n_pairs_list = []
#     n_positive_pairs_list = []
#     n_negative_pairs_list = []
#     n_positive_pairs_dict = defaultdict(int)

#     for data in tqdm(dataset):
#         n_sentences_list.append(len(data.sentences))
#         n_words_list.append(len(utils.flatten_lists(data.sentences)))
#         n_subtokens_list.append(len(utils.flatten_lists(data.segments)))
#         n_segments_list.append(len(data.segments))

#         n_mentions_list.append(len(data.mentions))
#         for span, name, etype, eid in data.mentions:
#             n_mentions_dict[etype] += 1
#         n_mentions2_list.append(sum([len(ms) for ms, etype, eid in data.entities]))
#         for ms, etype, eid in data.entities:
#             n_mentions2_dict[etype] += len(ms)

#         n_entities_list.append(len(data.entities))
#         for ms, etype, eid in data.entities:
#             n_entities_dict[etype] += 1

#         n_pairs_list.append(len(data.pair_head_entity_indices))
#         if hasattr(data, "pair_gold_relation_labels"):
#             n_positive_pairs = 0
#             n_negative_pairs = 0
#             for labels in data.pair_gold_relation_labels:
#                 if labels[0] == 1:
#                     n_negative_pairs += 1
#                 else:
#                     n_positive_pairs += 1
#             n_positive_pairs_list.append(n_positive_pairs)
#             n_negative_pairs_list.append(n_negative_pairs)

#             # Class-wise
#             count = 0
#             for labels in data.pair_gold_relation_labels:
#                 for l_i, l in enumerate(labels):
#                     if l_i == 0:
#                         continue
#                     if l == 1:
#                         n_positive_pairs_dict[ivocab_relation[l_i]] += 1
#                         count += 1
#             assert count == len(data.relations)
#         else:
#             n_positive_pairs_list.append(0)
#             n_negative_pairs_list.append(0)

#     results = {}
#     results["Number of documents"] = n_documents
#     results["Number of sentences"] = get_statistics_text(n_sentences_list)
#     results["Number of words"] = get_statistics_text(n_words_list)
#     results["Number of subtokens"] = get_statistics_text(n_subtokens_list)
#     results["Number of segments"] = get_statistics_text(n_segments_list)

#     results["Number of mentions"] = get_statistics_text(n_mentions_list)
#     for key, value in n_mentions_dict.items():
#         results[f"\tNumber of mentions for {key}"] = value
#     results["Number of mentions (with redundancy)"] = get_statistics_text(n_mentions2_list)
#     for key, value in n_mentions2_dict.items():
#         results[f"\tNumber of mentions (with redundancy) for {key}"] = value

#     results["Number of entities"] = get_statistics_text(n_entities_list)
#     for key, value in n_entities_dict.items():
#         results[f"\tNumber of entities for {key}"] = value

#     results["Number of pairs"] = get_statistics_text(n_pairs_list)
#     results["Number of positive pairs"] = get_statistics_text(n_positive_pairs_list)
#     for key, value in n_positive_pairs_dict.items():
#         results[f"\tNumber of positive pairs for {key}"] = value
#     results["Number of netative pairs"] = get_statistics_text(n_negative_pairs_list)

#     table = {}
#     table[title] = results.keys()
#     table["Statistics"] = results.values()
#     df = pd.DataFrame.from_dict(table)
#     logging.info("\n" + tabulate.tabulate(df, headers="keys", tablefmt="psql", floatfmt=".1f"))


# def get_statistics_text(xs):
#     sum_ = np.sum(xs)
#     mean_ = np.mean(xs)
#     max_ = np.max(xs)
#     min_ = np.min(xs)
#     return f"Total: {sum_} / Average per doc: {mean_} / Max: {max_} / Min: {min_}"

