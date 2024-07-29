import argparse
import logging
import os

# import numpy as np
import torch
import transformers

import sys
sys.path.insert(0, "../..")
from kapipe.systems import EntityRerankingCrossEncoderSystem
from kapipe.trainers import EntityRerankingCrossEncoderTrainer
from kapipe import evaluation
from kapipe import utils
from kapipe.utils import StopWatch

import shared_functions


def main(args):
    transformers.logging.set_verbosity_error()

    sw = StopWatch()
    sw.start("main")

    path_train_documents = args.train
    path_dev_documents = args.dev
    path_test_documents = args.test
    path_train_candidate_entities = args.train_candidate_entities
    path_dev_candidate_entities = args.dev_candidate_entities
    path_test_candidate_entities = args.test_candidate_entities
    path_entity_dict = args.entity_dict
    path_results_dir = args.results_dir
    device = torch.device(f"cuda:{args.gpu}")
    config_path = args.config_path
    config_name = args.config_name
    prefix = args.prefix
    actiontype = args.actiontype
    path_pretrained_model = args.pretrained_model
    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()
        args.prefix = prefix
    assert actiontype in ["train", "evaluate"]

    ##################
    # Set logger
    ##################

    base_output_path = os.path.join(
        path_results_dir,
        "entityrerankingcrossencoder",
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

    # Get documents
    train_documents = utils.read_json(path_train_documents)
    dev_documents = utils.read_json(path_dev_documents)
    test_documents = utils.read_json(path_test_documents)

    # Get candidate entities
    train_candidate_entities = utils.read_json(path_train_candidate_entities)
    dev_candidate_entities = utils.read_json(path_dev_candidate_entities)
    test_candidate_entities = utils.read_json(path_test_candidate_entities)

    logging.info(utils.pretty_format_dict(
        evaluation.ed.recall_at_k(
            pred_path=path_train_candidate_entities,
            gold_path=path_train_documents
        )
    ))
    train_candidate_entities \
        = shared_functions.add_or_move_gold_entity_in_candidates(
            documents=train_documents,
            candidate_entities=train_candidate_entities
        )
    logging.info(utils.pretty_format_dict(
        evaluation.ed.recall_at_k(
            pred_path=train_candidate_entities,
            gold_path=path_train_documents
        )
    ))
    logging.info(utils.pretty_format_dict(
        evaluation.ed.recall_at_k(
            pred_path=path_dev_candidate_entities,
            gold_path=path_dev_documents
        )
    ))
    logging.info(utils.pretty_format_dict(
        evaluation.ed.recall_at_k(
            pred_path=path_test_candidate_entities,
            gold_path=path_test_documents
        )
    ))

    ##################
    # Get system
    ##################

    trainer = EntityRerankingCrossEncoderTrainer(
        base_output_path=base_output_path
    )

    config = utils.get_hocon_config(
        config_path=config_path,
        config_name=config_name
    )
    if actiontype == "train":
        if path_pretrained_model is not None:
            system = EntityRerankingCrossEncoderSystem(
                device=device,
                config=config,
                path_entity_dict=path_entity_dict,
                path_model=path_pretrained_model
            )
        else:
            system = EntityRerankingCrossEncoderSystem(
                device=device,
                config=config,
                path_entity_dict=path_entity_dict,
                path_model=None
            )
    else:
        system = EntityRerankingCrossEncoderSystem(
            device=device,
            config=config,
            path_entity_dict=path_entity_dict,
            path_model=trainer.paths["path_snapshot"]
        )

    ##################
    # Train or evaluate
    ##################

    trainer.setup_dataset(
        system=system,
        documents=dev_documents,
        candidate_entities=dev_candidate_entities,
        split="dev"
    )
    trainer.setup_dataset(
        system=system,
        documents=test_documents,
        candidate_entities=test_candidate_entities,
        split="test"
    )

    # Show candidate-entity scores
    shared_functions.evaluate_candidate_entity_retriever(
        trainer=trainer,
        path_candidate_entities=path_dev_candidate_entities,
        split="dev"
    )
    shared_functions.evaluate_candidate_entity_retriever(
        trainer=trainer,
        path_candidate_entities=path_test_candidate_entities,
        split="test"
    )

    if actiontype == "train":
        trainer.train(
            system=system,
            train_documents=train_documents,
            train_candidate_entities=train_candidate_entities,
            dev_documents=dev_documents,
            dev_candidate_entities=dev_candidate_entities
        )

    if actiontype == "evaluate":
        trainer.evaluate(
            system=system,
            documents=dev_documents,
            candidate_entities=dev_candidate_entities,
            split="dev"
        )
        trainer.evaluate(
            system=system,
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


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--dev", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--train_candidate_entities", type=str, required=True)
    parser.add_argument("--dev_candidate_entities", type=str, required=True)
    parser.add_argument("--test_candidate_entities", type=str, required=True)
    parser.add_argument("--entity_dict", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--actiontype", type=str, required=True)
    parser.add_argument("--pretrained_model", type=str, default=None)
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
        

