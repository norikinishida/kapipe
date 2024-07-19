import argparse
import logging
import os

import kapipe
from kapipe import utils


def train(args):
    # Read documents and entity dictionary
    train_documents = utils.read_json(args.train_documents)
    dev_documents = utils.read_json(args.dev_documents)
    entity_dict = utils.read_json(args.entity_dict)

    # Create a blank pipeline
    ka = kapipe.blank(
        gpu_map={
            "ner": 0,
            "ed_retrieval": 1,
            "ed_reranking": 2,
            "docre": 3,
        }
    )

    # NER
    ka.ner.fit(
        train_documents,
        dev_documents,
        optional_config={
            "bert_pretrained_name_or_path": "allenai/scibert_scivocab_uncased",
            "bert_learning_rate": 2e-5,
            "task_learning_rate": 1e-4,
            "dataset_name": "example_dataset",
            "allow_nested_entities": True,
            "max_epoch": 10,
        }
    )

    # ED-Retrieval
    ka.ed_ret.fit(
        entity_dict,
        train_documents,
        dev_documents,
        optional_config={
            "bert_pretrained_name_or_path": "allenai/scibert_scivocab_uncased",
            "bert_learning_rate": 2e-5,
            "task_learning_rate": 1e-4,
            "dataset_name": "example_dataset",
            "max_epoch": 10,
        }
    )

    # ED-Reranking
    train_candidate_entities = [
        ka.ed_ret(d, num_candidate_entities=128)[1]
        for d in train_documents
    ]
    dev_candidate_entities = [
        ka.ed_ret(d, num_candidate_entities=128)[1]
        for d in dev_documents
    ]
    ka.ed_rank.fit(
        entity_dict,
        train_documents, train_candidate_entities,
        dev_documents, dev_candidate_entities,
        optional_config={
            "bert_pretrained_name_or_path": "allenai/scibert_scivocab_uncased",
            "bert_learning_rate": 2e-5,
            "task_learning_rate": 1e-4,
            "dataset_name": "example_dataset",
            "max_epoch": 10,
        }
    )

    # DocRE
    ka.docre.fit(
        train_documents,
        dev_documents,
        optional_config={
            "bert_pretrained_name_or_path": "allenai/scibert_scivocab_uncased",
            "bert_learning_rate": 2e-5,
            "task_learning_rate": 1e-4,
            "dataset_name": "example_dataset",
            "max_epoch": 10,
            "possible_head_entity_types": ["Chemical"], # or None
            "possible_tail_entity_types": ["Disease"], # or None
        }
    )

    ka.save(identifier="hello_world")


def inference(args):
    logging.set_verbosity_error()

    ka = kapipe.load(identifier="hello_world")

    documents = utils.read_json(args.dev_documents)

    documents = [ka(d) for d in documents]

    path_output = os.path.splitext(args.dev_document)[0]
    utils.write_json(path_output + ".pipe.json", documents)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_documents", type=str, required=True)
    parser.add_argument("--dev_documents", type=str, required=True)
    parser.add_argument("--entity_dict", type=str, required=True)
    args = parser.parse_args()

    train(args)
    inference(args)