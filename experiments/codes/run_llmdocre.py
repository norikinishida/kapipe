import argparse
import logging
import os

import torch
import transformers

import sys
sys.path.insert(0, "../..")
from kapipe.systems import LexicalEntityRetrievalSystem
from kapipe.systems import LLMDocRESystem
from kapipe.trainers import LLMDocRETrainer
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
    path_entity_dict = args.entity_dict
    path_train_demonstrations = args.train_demonstrations
    path_dev_demonstrations = args.dev_demonstrations
    path_test_demonstrations = args.test_demonstrations
    path_results_dir = args.results_dir
    device = torch.device(f"cuda:{args.gpu}")
    config_path = args.config_path
    config_name = args.config_name
    prefix = args.prefix
    actiontype = args.actiontype
    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()
        args.prefix = prefix
    assert actiontype in ["evaluate", "prompt_check"]

    ##################
    # Set logger
    ##################

    base_output_path = os.path.join(
        path_results_dir,
        "llmdocre",
        config_name,
        prefix
    )
    utils.mkdir(base_output_path)

    # if actiontype == "train":
    #     shared_functions.set_logger(base_output_path + "/training.log")
    if actiontype == "evaluate":
        shared_functions.set_logger(base_output_path + "/evaluation.log")

    logging.info(utils.pretty_format_dict(vars(args)))

    ##################
    # Get documents, vocabulary, and supplemental_info
    ##################

    # Get documents
    train_documents = utils.read_json(path_train_documents)
    dev_documents = utils.read_json(path_dev_documents)
    test_documents = utils.read_json(path_test_documents)

    # Get demonstration information
    train_demonstrations = utils.read_json(path_train_demonstrations)
    dev_demonstrations = utils.read_json(path_dev_demonstrations)
    test_demonstrations = utils.read_json(path_test_demonstrations)

    # Get vocabulary
    if path_train_documents.endswith("cdr/train.json"):
        vocab_relation = get_vocab_relation_for_cdr()
    elif path_train_documents.endswith("gda/train.json"):
        vocab_relation = get_vocab_relation_for_gda()
    elif path_train_documents.endswith("docred/train.json"):
        vocab_relation = get_vocab_relation_for_docred(
            path=os.path.join(path_train_documents, "..", "meta", "rel2id.json")
        )
    elif path_train_documents.endswith("redocred/train.json"):
        vocab_relation = get_vocab_relation_for_redocred(
            path=os.path.join(path_train_documents, "..", "meta", "rel2id.json")
        )
    elif "hoip" in path_train_documents:
        vocab_relation = get_vocab_relation_for_hoip()
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

    trainer = LLMDocRETrainer(base_output_path=base_output_path)

    config = utils.get_hocon_config(
        config_path=config_path,
        config_name=config_name
    )
    system = LLMDocRESystem(
        device=device,
        config=config,
        path_entity_dict=path_entity_dict,
        vocab_relation=vocab_relation,
        path_demonstration_pool=path_train_documents
    )

    ##################
    # Train or evaluate
    ##################

    if actiontype == "prompt_check":
        with torch.no_grad():
            path_out = base_output_path + "/output.txt"
            with open(path_out, "w") as f:
                for i, (document, demos) in enumerate(zip(
                    dev_documents,
                    dev_demonstrations
                )):
                    doc_key = document["doc_key"]
                    logging.info(f"Processing {doc_key}")
                    document = system.extract(
                        document=document,
                        demonstrations_for_doc=demos
                    )
                    f.write(f"--- DOC_KEY ({doc_key}) ---\n\n")
                    f.write("Prompt:\n")
                    f.write(document["docre_prompt"] + "\n\n")
                    f.write("-----\n")
                    f.write("Generated Text:\n")
                    f.write(document["docre_generated_text"] + "\n\n")
                    f.write("-----\n")
                    f.write("Parsed Triples:\n")
                    for t in document["relations"]:
                        f.write(f"{t}\n")
                    f.write("-----\n")
                    f.flush()
                    if i > 5:
                        break
            return

    trainer.setup_dataset(
        system=system,
        documents=train_documents,
        demonstrations=train_demonstrations,
        split="train",
        with_gold_annotations=True
    )
    trainer.setup_dataset(
        system=system,
        documents=dev_documents,
        demonstrations=dev_demonstrations,
        split="dev",
        with_gold_annotations=True
    )
    if config["dataset_name"] == "docred":
        trainer.setup_dataset(
            system=system,
            documents=test_documents,
            demonstrations=test_demonstrations,
            split="test",
            with_gold_annotations=False
        )
    else:
        trainer.setup_dataset(
            system=system,
            documents=test_documents,
            demonstrations=test_demonstrations,
            split="test",
            with_gold_annotations=True
        )

    trainer.save_system(system=system)

    if actiontype == "evaluate":
        if config["dataset_name"] == "docred":
            # Dev
            trainer.official_evaluate(
                system=system,
                documents=dev_documents,
                demonstrations=dev_demonstrations,
                split="dev",
                supplemental_info=supplemental_info
            )
            # Test
            trainer.official_evaluate(
                system=system,
                documents=test_documents,
                demonstrations=test_demonstrations,
                split="test",
                supplemental_info=supplemental_info,
                #
                prediction_only=True
            )
        elif config["dataset_name"] == "redocred":
            # Dev
            trainer.official_evaluate(
                system=system,
                documents=dev_documents,
                demonstrations=dev_demonstrations,
                split="dev",
                supplemental_info=supplemental_info
            )
            # Test
            trainer.official_evaluate(
                system=system,
                documents=test_documents,
                demonstrations=test_demonstrations,
                split="test",
                supplemental_info=supplemental_info
            )
        else:
            # Dev
            trainer.evaluate(
                system=system,
                documents=dev_documents,
                demonstrations=dev_demonstrations,
                split="dev",
                supplemental_info=supplemental_info,
                #
                skip_intra_inter=True,
                skip_ign=True
            )
            # Test
            trainer.evaluate(
                system=system,
                documents=test_documents,
                demonstrations=test_demonstrations,
                split="test",
                supplemental_info=supplemental_info,
                #
                skip_intra_inter=True,
                skip_ign=True
            )

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
    # relations = ["NO-REL"] + relations
    vocab_relation = {rel: rel_id for rel_id, rel in enumerate(relations)}
    return vocab_relation


def get_vocab_relation_for_cdr():
    # relations = ["NO-REL", "CID"] # CDR contains only one relation type: CID
    relations = ["CID"]
    vocab_relation = {rel: rel_id for rel_id, rel in enumerate(relations)}
    return vocab_relation


def get_vocab_relation_for_gda():
    # relations = ["NO-REL", "GDA"] # GDA contains only one relation type: GDA
    relations = ["GDA"]
    vocab_relation = {rel: rel_id for rel_id, rel in enumerate(relations)}
    return vocab_relation


def get_vocab_relation_for_docred(path):
    original_rel2id = utils.read_json(path)
    relations = [(rel, rel_i) for rel, rel_i in original_rel2id.items()]
    relations = sorted(relations, key=lambda tpl: tpl[1])
    assert relations[0] == ("Na", 0)
    # relations = ["NO-REL"] + [rel for rel, rel_i in relations[1:]]
    relations = [rel for rel, rel_i in relations[1:]]
    vocab_relation = {rel: rel_id for rel_id, rel in enumerate(relations)}
    return vocab_relation


def get_vocab_relation_for_redocred(path):
    original_rel2id = utils.read_json(path)
    relations = [(rel, rel_i) for rel, rel_i in original_rel2id.items()]
    relations = sorted(relations, key=lambda tpl: tpl[1])
    assert relations[0] == ("Na", 0)
    # relations = ["NO-REL"] + [rel for rel, rel_i in relations[1:]]
    relations = [rel for rel, rel_i in relations[1:]]
    vocab_relation = {rel: rel_id for rel_id, rel in enumerate(relations)}
    return vocab_relation


def get_vocab_relation_for_hoip():
    # relations = [
    #     "has result", "has part", "has molecular reaction", "part of"
    # ]
    relations = ["has result", "has part"]
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
    parser.add_argument("--train_demonstrations", type=str, required=True)
    parser.add_argument("--dev_demonstrations", type=str, required=True)
    parser.add_argument("--test_demonstrations", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--actiontype", type=str, required=True)
    args = parser.parse_args()

    main(args=args)

