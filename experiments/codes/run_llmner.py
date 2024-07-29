import argparse
import logging
import os

import torch
import transformers

import sys
sys.path.insert(0, "../..")
from kapipe.systems import LLMNERSystem
from kapipe.trainers import LLMNERTrainer
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
    # path_train_demonstrations = args.train_demonstrations
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
    # assert actiontype in ["train", "evaluate"]
    assert actiontype in ["evaluate", "prompt_check"]

    ##################
    # Set logger
    ##################

    base_output_path = os.path.join(
        path_results_dir,
        "llmner",
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
    # Get documents and vocabulary
    ##################

    # Get documents
    train_documents = utils.read_json(path_train_documents)
    dev_documents = utils.read_json(path_dev_documents)
    test_documents = utils.read_json(path_test_documents)

    # Get demonstration information
    # train_demonstrations = utils.read_json(path_train_demonstrations)
    dev_demonstrations = utils.read_json(path_dev_demonstrations)
    test_demonstrations = utils.read_json(path_test_demonstrations)

    # Get vocabulary
    if path_train_documents.endswith("cdr/train.json"):
        vocab_etype = get_vocab_etype_for_cdr()
    elif path_train_documents.endswith("conll2003/train.json"):
        vocab_etype = get_vocab_etype_for_conll2003(
            os.path.join(
                os.path.dirname(path_train_documents),
                "meta/entity_type_to_id.json"
            )
        )
    else:
        vocab_etype = get_vocab_etype(
            documents_list=[train_documents, dev_documents, test_documents]
        )

    ##################
    # Get system
    ##################

    trainer = LLMNERTrainer(base_output_path=base_output_path)

    config = utils.get_hocon_config(
        config_path=config_path,
        config_name=config_name
    )
    system = LLMNERSystem(
        device=device,
        config=config,
        vocab_etype=vocab_etype,
        path_demonstration_pool=path_train_documents
    )

    ##################
    # Train or evaluate
    ##################

    if actiontype == "prompt_check":
        with torch.no_grad():
            path_out = base_output_path + "/output.txt"
            with open(path_out, "w") as f:
                for i, (document, demos) in enumerate(
                    zip(dev_documents, dev_demonstrations)
                ):
                    doc_key = document["doc_key"]
                    logging.info(f"Processing {doc_key}")
                    document = system.extract(
                        document=document,
                        demonstrations_for_doc=demos
                    )
                    f.write(f"--- DOC_KEY ({doc_key}) ---\n\n")
                    f.write("Prompt:\n")
                    f.write(document["ner_prompt"] + "\n\n")
                    f.write("-----\n")
                    f.write("Generated Text:\n")
                    f.write(document["ner_generated_text"] + "\n\n")
                    f.write("-----\n")
                    f.write("Parsed mentions:\n")
                    for m in document["mentions"]:
                        f.write(f"{m}\n")
                    f.write("-----\n")
                    f.flush()
                    if i > 5:
                        break
            return

    trainer.setup_dataset(
        system=system,
        documents=dev_documents,
        demonstrations=dev_demonstrations,
        split="dev"
    )
    trainer.setup_dataset(
        system=system,
        documents=test_documents,
        demonstrations=test_demonstrations,
        split="test"
    )

    trainer.save_system(system=system)

    if actiontype == "evaluate":
        trainer.evaluate(
            system=system,
            documents=dev_documents,
            demonstrations=dev_demonstrations,
            split="dev"
        )
        trainer.evaluate(
            system=system,
            documents=test_documents,
            demonstrations=test_demonstrations,
            split="test"
        )

    ##################
    # Closing
    ##################

    logging.info("Done.")
    sw.stop("main")
    logging.info("Time: %f min." % sw.get_time("main", minute=True))

    return prefix


def get_vocab_etype(documents_list):
    entity_types = set()
    for documents in documents_list:
        for document in documents:
            for mention in document["mentions"]:
                entity_types.add(mention["entity_type"])
    entity_types = sorted(list(entity_types))
    # entity_types = ["NO-ENT"] + entity_types
    vocab_etype = {e_type: e_id for e_id, e_type in enumerate(entity_types)}
    return vocab_etype


def get_vocab_etype_for_cdr():
    # entity_types = ["NO-ENT"] + ["Chemical", "Disease"]
    entity_types = ["Chemical", "Disease"]
    vocab_etype = {e_type: e_id for e_id, e_type in enumerate(entity_types)}
    return vocab_etype


def get_vocab_etype_for_conll2003(path):
    original_etype2id = utils.read_json(path)
    entity_types = [
        (etype, etype_i) for etype, etype_i in original_etype2id.items()
    ]
    entity_types = sorted(entity_types, key=lambda tpl: tpl[1])
    # entity_types = ["NO-ENT"] + [etype for etype, etype_i in entity_types]
    entity_types = [etype for etype, etype_i in entity_types]
    vocab_etype = {e_type: e_id for e_id, e_type in enumerate(entity_types)}
    return vocab_etype


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--dev", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
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


