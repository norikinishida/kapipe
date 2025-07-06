from collections import defaultdict
import logging
import os
import random
import sys

import numpy as np
import torch
import pandas as pd
import tabulate
from tqdm import tqdm

# import sys
# sys.path.append("../..")
# from kapipe import evaluation
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


# def evaluate_candidate_entity_retriever(
#     trainer,
#     path_candidate_entities,
#     split
# ):
#     scores = evaluation.ed.accuracy(
#         pred_path=path_candidate_entities.replace(
#             ".pred_candidate_entities.", ".pred."
#         ),
#         gold_path=trainer.paths[f"path_{split}_gold"],
#         inkb=True
#     )
#     scores.update(evaluation.ed.fscore(
#         pred_path=path_candidate_entities.replace(
#             ".pred_candidate_entities.", ".pred."
#         ),
#         gold_path=trainer.paths[f"path_{split}_gold"],
#         inkb=True
#     ))
#     logging.info(f"Evaluation results of candidate entity retriever for the {split} set:")
#     logging.info(utils.pretty_format_dict(scores))

   
def show_ner_documents_statistics(documents, title):
    """Show NER documents statistics

    Parameters
    ----------
    documents : list[Document]
    title: str

    Summarize the following statistics
        - Number of documents
        - Number of sentences
            - Average number of sentences per document
        - Number of words
            - Average number of words per document
        - Number of mentions (without/with redundancy)
            - Average number of mentions (without/with redundancy) per document
        - Number of mentions (without/with redundancy) for each entity type
    """
    n_documents = len(documents)
    n_sentences_list = []
    n_words_list = []

    n_mentions_list = []
    n_mentions_dict = defaultdict(int)

    for doc in tqdm(documents):
        n_sentences_list.append(len(doc["sentences"]))

        n_words_list.append(len(utils.flatten_lists(
            [s.split() for s in doc["sentences"]]
        )))

        n_mentions_list.append(len(doc["mentions"]))
        for mention in doc["mentions"]:
            etype = mention["entity_type"]
            n_mentions_dict[etype] += 1

    results = {}

    results["Number of documents"] = n_documents
    results["Number of sentences"] = get_statistics_text(n_sentences_list)
    results["Number of words"] = get_statistics_text(n_words_list)

    results["Number of mentions"] = get_statistics_text(n_mentions_list)
    for key, value in sorted(list(n_mentions_dict.items()), key=lambda tpl: tpl[0]):
        results[f"\tNumber of mentions for {key}"] = value

    table = {}
    table[title] = results.keys()
    table["Statistics"] = results.values()
    df = pd.DataFrame.from_dict(table)
    logging.info("\n" + tabulate.tabulate(df, headers="keys", tablefmt="psql", floatfmt=".1f"))


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


def show_questions_statistics(questions, title):
    """Show QA questions statistics

    Parameters
    ----------
    questions: list[Question]
    title: str

    Summarize the following statistics
        - Number of questions
        - Number of yes-no/list/long answers
            - Average number of answers per question
    """
    n_questions = len(questions)
    n_yesno_answers_list = []
    n_long_answers_list = []
    n_list_answers_list = []
    n_synonyms_list = []

    for q in tqdm(questions):
        answers = q.get("answers", None)
        if answers is None:
            continue

        n_yesno_answers = 0
        n_long_answers = 0
        for a in answers:
            if a["answer_type"] == "yesno":
                n_yesno_answers += 1
            elif a["answer_type"] == "long":
                n_long_answers += 1
        n_yesno_answers_list.append(n_yesno_answers)
        n_long_answers_list.append(n_long_answers)

        index_to_synonyms = {} # dict[int, list[str]]
        for a in answers:
            if a["answer_type"] == "list":
                list_index = a["list_index"]
                if not list_index in index_to_synonyms:
                    index_to_synonyms[list_index] = []
                index_to_synonyms[list_index].append(a["answer"])
        n_list_answers = len(index_to_synonyms)
        n_list_answers_list.append(n_list_answers)
        for index, synonyms in index_to_synonyms.items():
            n_synonyms = len(synonyms)
            n_synonyms_list.append(n_synonyms)

    results = {}

    results["Number of questions"] = n_questions
    results["Number of yes-no answers"] = get_statistics_text(n_yesno_answers_list)
    results["Number of factoid answers"] = get_statistics_text(n_list_answers_list)
    results["Number of synonyms"] = get_statistics_text(n_synonyms_list)
    results["Number of long answers"] = get_statistics_text(n_long_answers_list)

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

