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
from kapipe.triple_extraction import BlinkBiEncoder, BlinkBiEncoderTrainer
from kapipe.triple_extraction import LexicalEntityRetriever, LexicalEntityRetrieverTrainer
from kapipe import utils
from kapipe.utils import StopWatch

import shared_functions


def main(args):
    transformers.logging.set_verbosity_error()

    sw = StopWatch()
    sw.start("main")

    device = torch.device(f"cuda:{args.gpu}")
    
    method_name = args.method

    path_train_documents = args.train
    path_dev_documents = args.dev
    path_test_documents = args.test

    path_entity_dict = args.entity_dict

    path_results_dir = args.results_dir

    config_path = args.config_path
    config_name = args.config_name
    prefix = args.prefix

    actiontype = args.actiontype

    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()
        args.prefix = prefix

    assert method_name in ["blinkbiencoder", "lexicalentityretriever"]
    assert actiontype in ["train", "evaluate", "check_preprocessing"]

    ##################
    # Set logger
    ##################

    base_output_path = os.path.join(
        path_results_dir,
        "edret",
        method_name,
        config_name,
        prefix
    )
    utils.mkdir(base_output_path)

    if actiontype == "train":
        shared_functions.set_logger(base_output_path + "/training.log")
    elif actiontype == "evaluate":
        shared_functions.set_logger(base_output_path + "/evaluation.log")

    logging.info(utils.pretty_format_dict(vars(args)))

    ##################
    # Get documents
    ##################

    train_documents = utils.read_json(path_train_documents)
    dev_documents = utils.read_json(path_dev_documents)
    test_documents = utils.read_json(path_test_documents)

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
    # Get extractor
    ##################

    if method_name == "blinkbiencoder":

        trainer = BlinkBiEncoderTrainer(
            base_output_path=base_output_path
        )

        config = utils.get_hocon_config(
            config_path=config_path,
            config_name=config_name
        )

        if actiontype == "train" or actiontype == "check_preprocessing":
            extractor = BlinkBiEncoder(
                device=device,
                config=config,
                path_entity_dict=path_entity_dict,
                path_model=None
            )
        else:
            extractor = BlinkBiEncoder(
                device=device,
                config=config,
                path_entity_dict=path_entity_dict,
                path_model=trainer.paths["path_snapshot"]
            )
            extractor.make_index(use_precomputed_entity_vectors=True)

    elif method_name == "lexicalentityretriever":

        trainer = LexicalEntityRetrieverTrainer(
            base_output_path=base_output_path
        )

        config = utils.get_hocon_config(
            config_path=config_path,
            config_name=config_name
        )

        assert actiontype == "evaluate"

        extractor = LexicalEntityRetriever(
            config=config,
            path_entity_dict=path_entity_dict
        )

    ##################
    # Train or evaluate
    ##################

    if method_name == "blinkbiencoder":

        processed_train_documents = remove_out_of_kb_mentions(
            train_documents=train_documents,
            entity_dict=extractor.entity_dict
        )
        shared_functions.show_ed_documents_statistics(
            documents=processed_train_documents,
            title="Training after Out-of-Kb Removal"
        )

        processed_train_documents = replicate_documents_with_too_many_mentions(
            train_documents=processed_train_documents,
            n_candidate_entities=extractor.config["n_candidate_entities"]
        )
        shared_functions.show_ed_documents_statistics(
            documents=processed_train_documents,
            title="Training after replication"
        )

        trainer.setup_dataset(
            extractor=extractor,
            documents=dev_documents,
            split="dev"
        )
        trainer.setup_dataset(
            extractor=extractor,
            documents=test_documents,
            split="test"
        )

        if actiontype == "train":
            trainer.train(
                extractor=extractor,
                train_documents=processed_train_documents,
                dev_documents=dev_documents
            )

        elif actiontype == "evaluate":
            trainer.evaluate(
                extractor=extractor,
                documents=dev_documents,
                split="dev"
            )
            trainer.evaluate(
                extractor=extractor,
                documents=test_documents,
                split="test"
            )
            trainer.evaluate(
                extractor=extractor,
                documents=train_documents,
                split="train",
                #
                prediction_only=True
            )

        elif actiontype == "check_preprocessing":
            results = []
            for document in dev_documents:
                preprocessed_data = extractor.model.preprocessor.preprocess(document)
                results.append(preprocessed_data)
            utils.write_json(base_output_path + "/dev.check_preprocessing.json", results)

    elif method_name == "lexicalentityretriever":

        trainer.setup_dataset(
            extractor=extractor,
            documents=dev_documents,
            split="dev"
        )
        trainer.setup_dataset(
            extractor=extractor,
            documents=test_documents,
            split="test"
        )

        trainer.save_extractor(extractor=extractor)

        if actiontype == "evaluate":
            trainer.evaluate(
                extractor=extractor,
                documents=dev_documents,
                split="dev"
            )
            trainer.evaluate(
                extractor=extractor,
                documents=test_documents,
                split="test"
            )
            trainer.evaluate(
                extractor=extractor,
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

    
def replicate_documents_with_too_many_mentions(
    train_documents,
    n_candidate_entities
):
    new_train_documents = [] # list[Document]

    max_num_mentions = n_candidate_entities // 2

    n_total_original_documents = 0
    # n_total_original_mentions = 0
    n_total_replicated_documents = 0
    # n_total_replicated_mentions = 0

    for doc_i in tqdm(
        range(len(train_documents)),
        desc=f"Replicating documents with more than {n_candidate_entities}/2 mentions"
    ):
        original_doc = train_documents[doc_i]
        original_mentions = original_doc["mentions"]
        n_total_original_documents += 1
        # n_total_original_mentions += len(original_mentions)
        
        if len(original_mentions) <= max_num_mentions:
            doc = copy.deepcopy(original_doc)
            new_train_documents.append(doc)
            n_total_replicated_documents += 1
            # n_total_replicated_mentions += len(doc["mentions"])

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
                n_total_replicated_documents += 1
                # n_total_replicated_mentions += len(doc["mentions"])

    logging.info(f"Num. original training documents: {n_total_original_documents}")
    # logging.info(f"Num. original training mentions: {n_total_original_mentions}")
    logging.info(f"Num. replicated training documents: {n_total_replicated_documents}")
    # logging.info(f"Num. replicated training mentions: {n_total_replicated_mentions}")

    return new_train_documents


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO
    )

    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--method", type=str, required=True)

    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--dev", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)

    parser.add_argument("--entity_dict", type=str, required=True)

    parser.add_argument("--results_dir", type=str, required=True)

    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

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

