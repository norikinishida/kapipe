import argparse
import logging
import os

import torch
import transformers

from kapipe.systems import MAATLOPSystem
from kapipe.trainers import MAATLOPTrainer
from kapipe import utils
from kapipe.utils import StopWatch

import shared_functions


def main(args):
    # torch.autograd.set_detect_anomaly(True)
    transformers.logging.set_verbosity_error()

    sw = StopWatch()
    sw.start("main")

    path_train_documents = args.train
    path_dev_documents = args.dev
    path_test_documents = args.test
    path_entity_dict = args.entity_dict
    path_results_dir = args.results_dir
    device = torch.device(f"cuda:{args.gpu}")
    config_path = args.config_path
    config_name = args.config_name
    prefix = args.prefix
    actiontype = args.actiontype
    path_pretrained_model = args.pretrained_model
    path_pretrained_model_vocab_relation = args.pretrained_model_vocab_relation
    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()
        args.prefix = prefix
    assert actiontype in ["train", "evaluate", "check_preprocessing"]

    ##################
    # Set logger
    ##################

    base_output_path = os.path.join(
        path_results_dir,
        "maatlop",
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
    # Get documents, vocabulary, and supplemental_info
    ##################

    # Get documents
    train_documents = utils.read_json(path_train_documents)
    dev_documents = utils.read_json(path_dev_documents)
    test_documents = utils.read_json(path_test_documents)

    # Get vocabulary
    if path_train_documents.endswith("cdr/train.json"):
        vocab_relation = get_vocab_relation_for_cdr()
    elif path_train_documents.endswith("gda/train.json"):
        vocab_relation = get_vocab_relation_for_gda()
    elif path_train_documents.endswith("docred/train.json"):
        vocab_relation = get_vocab_relation_for_docred(
            path=os.path.join(
                path_train_documents, "..", "meta", "rel2id.json"
            )
        )
    elif path_train_documents.endswith("redocred/train.json"):
        vocab_relation = get_vocab_relation_for_redocred(
            path=os.path.join(
                path_train_documents, "..", "meta", "rel2id.json"
            )
        )
    else:
        vocab_relation = get_vocab_relation(
            documents_list=[train_documents, dev_documents, test_documents]
        )

    # Get supplemental information
    if path_train_documents.endswith("docred/train.json"):
        train_file_name, dev_file_name, test_file_name \
            = shared_functions.get_docred_info(dataset_name="docred")
    elif path_train_documents.endswith("redocred/train.json"):
        train_file_name, dev_file_name, test_file_name \
            = shared_functions.get_docred_info(dataset_name="redocred")
    else:
        train_file_name = dev_file_name = test_file_name = None
    supplemental_info = {
        # Information for DocRED/Re-DocRED official evaluation
        "original_data_dir": \
            os.path.join(path_train_documents, "..", "original"),
        "train_file_name": train_file_name,
        "dev_file_name": dev_file_name,
        "test_file_name": test_file_name,
    }

    ##################
    # Get system
    ##################

    trainer = MAATLOPTrainer(base_output_path=base_output_path)

    config = utils.get_hocon_config(
        config_path=config_path,
        config_name=config_name
    )
    if actiontype == "train":
        system = MAATLOPSystem(
            device=device,
            config=config,
            path_entity_dict=path_entity_dict,
            vocab_relation=vocab_relation,
            path_model=None
        )
        # if path_pretrained_model is not None:
        #     logging.info(f"Loaded pretrained parameters from {path_pretrained_model}")
        #     system.load_model(
        #         path_pretrained_model,
        #         ignored_names=["block_bilinear"]
        #     )
    else:
        system = MAATLOPSystem(
            device=device,
            config=config,
            path_entity_dict=path_entity_dict,
            vocab_relation=trainer.paths["path_vocab_relation"],
            path_model=trainer.paths["path_snapshot"]
        )

    ##################
    # Train or evaluate
    ##################

    trainer.setup_dataset(
        system=system,
        documents=train_documents,
        split="train",
        with_gold_annotations=True
    )
    trainer.setup_dataset(
        system=system,
        documents=dev_documents,
        split="dev",
        with_gold_annotations=True
    )
    if config["dataset_name"] == "docred":
        trainer.setup_dataset(
            system=system,
            documents=test_documents,
            split="test",
            with_gold_annotations=False
        )
    else:
        trainer.setup_dataset(
            system=system,
            documents=test_documents,
            split="test",
            with_gold_annotations=True
        )

    if actiontype == "train":
        trainer.train(
            system=system,
            train_documents=train_documents,
            dev_documents=dev_documents,
            supplemental_info=supplemental_info
        )

    if actiontype == "evaluate":
        if config["dataset_name"] == "docred":
            # Dev
            trainer.official_evaluate(
                system=system,
                documents=dev_documents,
                split="dev",
                supplemental_info=supplemental_info
            )
            # Test
            trainer.official_evaluate(
                system=system,
                documents=test_documents,
                split="test",
                supplemental_info=supplemental_info,
                #
                prediction_only=True,
            )
        elif config["dataset_name"] == "redocred":
            # Dev
            trainer.official_evaluate(
                system=system,
                documents=dev_documents,
                split="dev",
                supplemental_info=supplemental_info
            )
            # Test
            trainer.official_evaluate(
                system=system,
                documents=test_documents,
                split="test",
                supplemental_info=supplemental_info
            )
        else:
            # Dev
            trainer.evaluate(
                system=system,
                documents=dev_documents,
                split="dev",
                supplemental_info=supplemental_info,
                #
                skip_intra_inter=True
            )
            # Test
            trainer.evaluate(
                system=system,
                documents=test_documents,
                split="test",
                supplemental_info=supplemental_info,
                #
                skip_intra_inter=True
            )

    if actiontype == "check_preprocessing":
        results = []
        for document in dev_documents:
            if system.config["do_negative_entity_sampling"]:
                document = system.sample_negative_entities_randomly(
                    document=document,
                    sample_size=round(
                        len(document["entities"])
                        * system.config["negative_entity_ratio"]
                    )
                )
            preprocessed_data = system.preprocessor.preprocess(document)
            preprocessed_data["pair_head_entity_indices"] = preprocessed_data["pair_head_entity_indices"].tolist()
            preprocessed_data["pair_tail_entity_indices"] = preprocessed_data["pair_tail_entity_indices"].tolist()
            preprocessed_data["pair_gold_relation_labels"] = preprocessed_data["pair_gold_relation_labels"].tolist()
            results.append(preprocessed_data)
        utils.write_json(base_output_path + "/dev.check_preprocessing.json", results)

    ##################
    # Closing
    ##################

    logging.info("Done.")
    sw.stop("main")
    logging.info("Time: %f min." % sw.get_time("main", minute=True))

    return prefix


def get_vocab_relation(documents_list):
    relations = set()
    for documents in documents_list:
        for document in documents:
            for triple in document["relations"]:
                relations.add(triple["relation"])
    relations = sorted(list(relations))
    relations = ["NO-REL"] + relations
    vocab_relation = {rel: rel_id for rel_id, rel in enumerate(relations)}
    return vocab_relation


def get_vocab_relation_for_cdr():
    relations = ["NO-REL", "CID"] # CDR contains only one relation type: CID
    vocab_relation = {rel: rel_id for rel_id, rel in enumerate(relations)}
    return vocab_relation


def get_vocab_relation_for_gda():
    relations = ["NO-REL", "GDA"] # GDA contains only one relation type: GDA
    vocab_relation = {rel: rel_id for rel_id, rel in enumerate(relations)}
    return vocab_relation


def get_vocab_relation_for_docred(path):
    original_rel2id = utils.read_json(path)
    relations = [(rel, rel_i) for rel, rel_i in original_rel2id.items()]
    relations = sorted(relations, key=lambda tpl: tpl[1])
    assert relations[0] == ("Na", 0)
    relations = ["NO-REL"] + [rel for rel, rel_i in relations[1:]]
    vocab_relation = {rel: rel_id for rel_id, rel in enumerate(relations)}
    return vocab_relation


def get_vocab_relation_for_redocred(path):
    original_rel2id = utils.read_json(path)
    relations = [(rel, rel_i) for rel, rel_i in original_rel2id.items()]
    relations = sorted(relations, key=lambda tpl: tpl[1])
    assert relations[0] == ("Na", 0)
    relations = ["NO-REL"] + [rel for rel, rel_i in relations[1:]]
    vocab_relation = {rel: rel_id for rel_id, rel in enumerate(relations)}
    return vocab_relation


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--dev", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--entity_dict", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--actiontype", type=str, required=True)
    parser.add_argument("--pretrained_model", type=str, default=None)
    parser.add_argument(
        "--pretrained_model_vocab_relation",
        type=str,
        default=None
    )
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


