import argparse
import copy
import logging
import os

import numpy as np
import torch
import transformers
from tqdm import tqdm

import sys
sys.path.insert(0, "../..")
from kapipe.ed_retrieval import (
    BlinkBiEncoder, BlinkBiEncoderTrainer,
    LexicalEntityRetriever, LexicalEntityRetrieverTrainer
)
from kapipe import utils
from kapipe.utils import StopWatch

import shared_functions


def main(args):
    transformers.logging.set_verbosity_error()

    sw = StopWatch()
    sw.start("main")

    ##################
    # Arguments
    ##################

    # Method
    device = torch.device(f"cuda:{args.gpu}")
    method_name = args.method
    config_path = args.config_path
    config_name = args.config_name

    # Input Data
    path_train_documents = args.train_documents
    path_dev_documents = args.dev_documents
    path_test_documents = args.test_documents
    path_entity_dict = args.entity_dict

    # Output Path
    path_results_dir = args.results_dir
    prefix = args.prefix
    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()
        args.prefix = prefix

    # Action
    actiontype = args.actiontype

    assert method_name in ["blink_bi_encoder", "lexical_entity_retriever"]
    assert actiontype in ["train", "evaluate", "check_preprocessing"]

    ##################
    # Logging Setup
    ##################

    # Set base output path
    base_output_path = os.path.join(
        path_results_dir,
        "ed_retrieval",
        method_name,
        config_name,
        prefix
    )
    utils.mkdir(base_output_path)

    # Set logger
    if actiontype == "train":
        shared_functions.set_logger(
            os.path.join(base_output_path, "training.log"),
            # overwrite=True
        )
    elif actiontype == "evaluate":
        shared_functions.set_logger(
            os.path.join(base_output_path, "evaluation.log"),
            # overwrite=True
        )

    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))

    ##################
    # Data
    ##################

    # Load documents
    train_documents = utils.read_json(path_train_documents)
    dev_documents = utils.read_json(path_dev_documents)
    test_documents = utils.read_json(path_test_documents)

    # Show statistics
    shared_functions.show_ed_documents_statistics(
        documents=train_documents,
        title="Training"
    )
    shared_functions.show_ed_documents_statistics(
        documents=dev_documents,
        title="Development"
    )
    shared_functions.show_ed_documents_statistics(
        documents=test_documents,
        title="Test"
    )    

    ##################
    # Method
    ##################

    if method_name == "blink_bi_encoder":
        # Initialize the trainer (evaluator)
        trainer = BlinkBiEncoderTrainer(
            base_output_path=base_output_path
        )

        if actiontype == "train" or actiontype == "check_preprocessing":
            # Initialize the retriever
            config = utils.get_hocon_config(
                config_path=config_path,
                config_name=config_name
            )
            retriever = BlinkBiEncoder(
                device=device,
                config=config,
                path_entity_dict=path_entity_dict
            )
        else:
            # Load the retriever
            retriever = BlinkBiEncoder(
                device=device,
                path_snapshot=trainer.paths["path_snapshot"]
            )
            # Re-build index
            retriever.make_index(use_precomputed_entity_vectors=True)

    elif method_name == "lexical_entity_retriever":
        assert actiontype == "evaluate"

        # Initialize the trainer (evaluator)
        trainer = LexicalEntityRetrieverTrainer(
            base_output_path=base_output_path
        )

        # Initialize the retriever
        config = utils.get_hocon_config(
            config_path=config_path,
            config_name=config_name
        )
        retriever = LexicalEntityRetriever(
            config=config,
            path_entity_dict=path_entity_dict
        )

    ##################
    # Training, Evaluation
    ##################

    if method_name == "blink_bi_encoder":

        # Remove out-of-KB mentions in the training dataset
        processed_train_documents = remove_out_of_kb_mentions(
            train_documents=train_documents,
            entity_dict=retriever.entity_dict
        )
        shared_functions.show_ed_documents_statistics(
            documents=processed_train_documents,
            title="Training after Out-of-Kb Removal"
        )

        # Split the training documents that contain too many mentions by duplicating and partitioning their mentions.
        processed_train_documents = split_documents_by_mention_limit(
            train_documents=processed_train_documents,
            n_candidate_entities=retriever.config["n_candidate_entities"]
        )
        shared_functions.show_ed_documents_statistics(
            documents=processed_train_documents,
            title="Training after duplication"
        )

        # Set up the datasets for evaluation
        trainer.setup_dataset(
            retriever=retriever,
            documents=dev_documents,
            split="dev"
        )
        trainer.setup_dataset(
            retriever=retriever,
            documents=test_documents,
            split="test"
        )

        if actiontype == "train":
            # Train the retriever
            trainer.train(
                retriever=retriever,
                train_documents=processed_train_documents,
                dev_documents=dev_documents
            )

        elif actiontype == "evaluate":
            # Evaluate the retriever on the datasets
            trainer.evaluate(
                retriever=retriever,
                documents=dev_documents,
                split="dev"
            )
            trainer.evaluate(
                retriever=retriever,
                documents=test_documents,
                split="test"
            )
            trainer.evaluate(
                retriever=retriever,
                documents=train_documents,
                split="train",
                #
                prediction_only=True
            )

        elif actiontype == "check_preprocessing":
            # Save preprocessed data
            results = []
            for document in dev_documents:
                preprocessed_data = retriever.model.preprocessor.preprocess(document)
                results.append(preprocessed_data)
            utils.write_json(os.path.join(base_output_path, "dev.check_preprocessing.json"), results)

    elif method_name == "lexical_entity_retriever":

        # Set up the datasets for evaluation
        trainer.setup_dataset(
            retriever=retriever,
            documents=dev_documents,
            split="dev"
        )
        trainer.setup_dataset(
            retriever=retriever,
            documents=test_documents,
            split="test"
        )

        # Save the configurations of the retriever
        trainer.save_retriever(retriever=retriever)

        if actiontype == "evaluate":
            # Evaluate the retriever on the datasets
            trainer.evaluate(
                retriever=retriever,
                documents=dev_documents,
                split="dev"
            )
            trainer.evaluate(
                retriever=retriever,
                documents=test_documents,
                split="test"
            )
            trainer.evaluate(
                retriever=retriever,
                documents=train_documents,
                split="train",
                #
                prediction_only=True
            )

    ##################
    # Closing
    ##################

    logging.info("Done.")
    sw.stop("main")
    logging.info("Time: %f min." % sw.get_time("main", minute=True))

    return prefix


def remove_out_of_kb_mentions(train_documents, entity_dict):
    # new_train_documents = copy.deepcopy(train_documents)
    new_train_documents = []

    kb_entity_ids = set(list(entity_dict.keys()))

    n_prev_mentions = 0
    n_new_mentions = 0
    n_prev_entities = 0
    n_new_entities = 0

    for doc_i in tqdm(
        range(len(train_documents)),
        desc="removing out-of-kb mentions"
    ):
        # Copy the data
        doc = copy.deepcopy(train_documents[doc_i])

        # Remove out-of-kb mentions
        n_prev_mentions += len(doc["mentions"])
        mentions = [m for m in doc["mentions"] if m["entity_id"] in kb_entity_ids]
        n_new_mentions += len(mentions)

        # Reset mentions
        doc["mentions"] = mentions

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

        # processed_train_documents[doc_i] = doc
        new_train_documents.append(doc)

    logging.info(f"Removed {n_prev_mentions - n_new_mentions}({n_prev_entities - n_new_entities}) out-of-kb mentions (entities) in the training dataset: {n_prev_mentions} ({n_prev_entities}) -> {n_new_mentions} ({n_new_entities})")
    logging.info(f"Removed {len(train_documents) - len(new_train_documents)} documents with only out-of-kb mentions/entities in the training dataset: {len(train_documents)} -> {len(new_train_documents)}")

    return new_train_documents

    
def split_documents_by_mention_limit(
    train_documents,
    n_candidate_entities
):
    new_train_documents = [] # list[Document]

    max_num_mentions = n_candidate_entities // 2

    n_total_original_documents = 0
    # n_total_original_mentions = 0
    n_total_duplicated_documents = 0
    # n_total_duplicated_mentions = 0

    for doc_i in tqdm(
        range(len(train_documents)),
        desc=f"Duplicating documents with more than {n_candidate_entities}/2 mentions"
    ):
        original_doc = train_documents[doc_i]
        original_mentions = original_doc["mentions"]
        n_total_original_documents += 1
        # n_total_original_mentions += len(original_mentions)
        
        if len(original_mentions) <= max_num_mentions:
            doc = copy.deepcopy(original_doc)
            new_train_documents.append(doc)
            n_total_duplicated_documents += 1
            # n_total_duplicated_mentions += len(doc["mentions"])

        else:
            n_mentions = len(original_mentions)
            perm = np.random.permutation(n_mentions)
            for begin_i in range(0, n_mentions, max_num_mentions):
                doc = copy.deepcopy(original_doc)
                doc["mentions"] = sorted(
                    [
                        original_mentions[i]
                        for i in perm[begin_i: begin_i + max_num_mentions]
                    ],
                    key=lambda m: tuple(m["span"])
                ) 
                doc["entities"] = utils.aggregate_mentions_to_entities(
                    document=doc,
                    mentions=doc["mentions"]
                )
                new_train_documents.append(doc)
                n_total_duplicated_documents += 1
                # n_total_duplicated_mentions += len(doc["mentions"])

    logging.info(f"Num. original training documents: {n_total_original_documents}")
    # logging.info(f"Num. original training mentions: {n_total_original_mentions}")
    logging.info(f"Num. duplicated training documents: {n_total_duplicated_documents}")
    # logging.info(f"Num. duplicated training mentions: {n_total_duplicated_mentions}")

    return new_train_documents


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO
    )

    parser = argparse.ArgumentParser()

    # Method
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=True)

    # Input Data
    parser.add_argument("--train_documents", type=str, required=True)
    parser.add_argument("--dev_documents", type=str, required=True)
    parser.add_argument("--test_documents", type=str, required=True)
    parser.add_argument("--entity_dict", type=str, required=True)

    # Output Data
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
        shared_functions.pop_logger_handler()
        main(args=args)
    else:
        # Training or Evaluation
        main(args=args)

