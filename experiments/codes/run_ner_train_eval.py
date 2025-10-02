import argparse
import logging
import os

import pandas as pd
import torch
import transformers
from tqdm import tqdm

import sys
sys.path.insert(0, "../..")
from kapipe.ner import (
    BiaffineNER, BiaffineNERTrainer,
    LLMNER, LLMNERTrainer
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
    dataset_name = args.dataset_name
    path_train_documents = args.train_documents
    path_dev_documents = args.dev_documents
    path_test_documents = args.test_documents
    # path_train_demonstrations = args.train_demonstrations
    path_dev_demonstrations = args.dev_demonstrations
    path_test_demonstrations = args.test_demonstrations

    # Output Path
    path_results_dir = args.results_dir
    prefix = args.prefix
    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()
        args.prefix = prefix

    # Action
    actiontype = args.actiontype

    assert method_name in ["biaffine_ner", "llm_ner"]
    assert actiontype in ["train", "evaluate", "check_preprocessing", "check_prompt"]

    ##################
    # Logging Setup
    ##################

    # Set base output path
    base_output_path = os.path.join(
        path_results_dir,
        "ner",
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

    # Load demonstrations (for LLM and in-context learning)
    if method_name == "llm_ner":
        # train_demonstrations = utils.read_json(path_train_demonstrations)
        dev_demonstrations = utils.read_json(path_dev_demonstrations)
        test_demonstrations = utils.read_json(path_test_demonstrations)

    # Create vocabulary of entity types
    if dataset_name == "cdr":
        vocab_etype = get_vocab_etype_for_cdr(method_name=method_name)
    elif dataset_name == "conll2003":
        vocab_etype = get_vocab_etype_for_conll2003(
            path=os.path.join(
                os.path.dirname(path_train_documents),
                "meta/entity_type_to_id.json"
            ),
            method_name=method_name
        )
    elif dataset_name == "linked_docred":
        vocab_etype = get_vocab_etype_for_linked_docred(
            path=os.path.join(
                os.path.dirname(path_train_documents),
                "meta/ner2id.json"
            ),
            method_name=method_name
        )
    elif dataset_name == "medmentions":
        vocab_etype = get_vocab_etype_for_medmentions(
            path=os.path.join(
                os.path.dirname(path_train_documents),
                "meta/st21pv_semantic_types.json"
            ),
            method_name=method_name
        )
    else:
        vocab_etype = get_vocab_etype(
            documents_list=[
                train_documents,
                dev_documents,
                test_documents
            ],
            method_name=method_name
        )

    # Load meta information (e.g., pretty labels and definitions) for entity types
    if method_name == "llm_ner":
        etype_meta_info = {
            row["Entity Type"]: {
                "Pretty Name": row["Pretty Name"],
                "Definition": row["Definition"]
            }
            for _, row in pd.read_csv(f"./dataset-meta-information/{dataset_name}_entity_types.csv").iterrows()
        }

    # Show statistics
    shared_functions.show_ner_documents_statistics(
        documents=train_documents,
        title="Training"
    )
    shared_functions.show_ner_documents_statistics(
        documents=dev_documents,
        title="Development"
    )
    shared_functions.show_ner_documents_statistics(
        documents=test_documents,
        title="Test"
    )

    ##################
    # Method
    ##################

    if method_name == "biaffine_ner":
        # Initialize the trainer (evaluator)
        trainer = BiaffineNERTrainer(base_output_path=base_output_path)

        if actiontype == "train" or actiontype == "check_preprocessing":
            # Initialize the extractor
            config = utils.get_hocon_config(
                config_path=config_path,
                config_name=config_name
            )
            extractor = BiaffineNER(
                device=device,
                config=config,
                vocab_etype=vocab_etype
            )
        else: 
            # Load the extractor
            extractor = BiaffineNER(
                device=device,
                path_snapshot=trainer.paths["path_snapshot"]
            )

    elif method_name == "llm_ner":
        assert actiontype != "train"

        # Initialize the trainer (evaluator)
        trainer = LLMNERTrainer(base_output_path=base_output_path)

        # Load the configuration
        config = utils.get_hocon_config(config_path=config_path, config_name=config_name)

        # Initialize the extractor
        extractor = LLMNER(
            device=device,
            config=config,
            vocab_etype=vocab_etype,
            etype_meta_info=etype_meta_info,
            path_demonstration_pool=path_train_documents
        )

    ##################
    # Training, Evaluation
    ##################

    if method_name == "biaffine_ner":

        # Set up the datasets for evaluation
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
            # Train the extractor
            trainer.train(
                extractor=extractor,
                train_documents=train_documents,
                dev_documents=dev_documents
            )

        if actiontype == "evaluate":
            # Evaluate the extractor on the datasets
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

        if actiontype == "check_preprocessing":
            # Save preprocessed data
            results = []
            for document in tqdm(dev_documents):
                preprocessed_data = extractor.model.preprocessor.preprocess(document)
                for key in ["matrix_valid_span_mask", "matrix_gold_entity_type_labels"]:
                    preprocessed_data[key] = preprocessed_data[key].tolist()
                results.append(preprocessed_data)
            utils.write_json(os.path.join(base_output_path, "dev.check_preprocessing.json"), results)
        
    elif method_name == "llm_ner":

        if actiontype == "check_prompt":
            # Show prompts
            with torch.no_grad():
                path_out = os.path.join(base_output_path, "output.txt")
                with open(path_out, "w") as f:
                    for i, (document, demos) in enumerate(
                        zip(dev_documents, dev_demonstrations)
                    ):
                        doc_key = document["doc_key"]
                        logging.info(f"Processing {doc_key}")
                        result_document = extractor.extract(
                            document=document,
                            demonstrations_for_doc=demos
                        )
                        f.write(f"--- DOC_KEY ({doc_key}) ---\n\n")
                        f.write("Prompt:\n")
                        f.write(result_document["ner_prompt"] + "\n\n")
                        f.write("-----\n")
                        f.write("Generated Text:\n")
                        f.write(result_document["ner_generated_text"] + "\n\n")
                        f.write("-----\n")
                        f.write("Parsed mentions:\n")
                        for m in result_document["mentions"]:
                            f.write(f"{m}\n")
                        f.write("-----\n")
                        f.write("Gold mentions:\n")
                        for m in document["mentions"]:
                            f.write(f"{m}\n")
                        f.write("-----\n")
                        f.flush()
                        if i > 5:
                            break
                return

        # Set up the datasets for evaluation
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

        # Save the configurations of the extractor
        trainer.save_extractor(extractor=extractor)

        if actiontype == "evaluate":
            # Evaluate the extractor on the datasets
            trainer.evaluate(
                extractor=extractor,
                documents=dev_documents,
                demonstrations=dev_demonstrations,
                contexts=None,
                split="dev"
            )
            trainer.evaluate(
                extractor=extractor,
                documents=test_documents,
                demonstrations=test_demonstrations,
                contexts=None,
                split="test"
            )

    ##################
    # Closing
    ##################

    logging.info("Done.")
    sw.stop("main")
    logging.info("Time: %f min." % sw.get_time("main", minute=True))

    return prefix


def get_vocab_etype(documents_list, method_name):
    entity_types = set()
    for documents in documents_list:
        for document in documents:
            for mention in document["mentions"]:
                entity_types.add(mention["entity_type"])
    entity_types = sorted(list(entity_types))
    if method_name == "biaffine_ner":
        entity_types = ["NO-ENT"] + entity_types
    vocab_etype = {e_type: e_id for e_id, e_type in enumerate(entity_types)}
    return vocab_etype


def get_vocab_etype_for_cdr(method_name):
    entity_types = ["Chemical", "Disease"]
    if method_name == "biaffine_ner":
        entity_types = ["NO-ENT"] + entity_types
    vocab_etype = {e_type: e_id for e_id, e_type in enumerate(entity_types)}
    return vocab_etype


def get_vocab_etype_for_conll2003(path, method_name):
    original_etype2id = utils.read_json(path)
    entity_types = [(etype, etype_i) for etype, etype_i in original_etype2id.items()]
    entity_types = sorted(entity_types, key=lambda tpl: tpl[1])
    entity_types = [etype for etype, etype_i in entity_types]
    if method_name == "biaffine_ner":
        entity_types = ["NO-ENT"] + entity_types
    vocab_etype = {e_type: e_id for e_id, e_type in enumerate(entity_types)}
    return vocab_etype


def get_vocab_etype_for_linked_docred(path, method_name):
    original_etype2id = utils.read_json(path)
    entity_types = [(etype, etype_i) for etype, etype_i in original_etype2id.items()]
    entity_types = sorted(entity_types, key=lambda tpl: tpl[1])
    entity_types = [etype for etype, etype_i in entity_types]
    if method_name == "biaffine_ner":
        entity_types = ["NO-ENT"] + entity_types
    vocab_etype = {e_type: e_id for e_id, e_type in enumerate(entity_types)}
    return vocab_etype


def get_vocab_etype_for_medmentions(path, method_name):
    entity_types = utils.read_json(path)
    entity_types = [x["name"] for x in entity_types]
    if method_name == "biaffine_ner":
        entity_types = ["NO-ENT"] + entity_types
    vocab_etype = {e_type: e_id for e_id, e_type in enumerate(entity_types)}
    return vocab_etype


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
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--train_documents", type=str, required=True)
    parser.add_argument("--dev_documents", type=str, required=True)
    parser.add_argument("--test_documents", type=str, required=True)
    parser.add_argument("--train_demonstrations", type=str, default=None)
    parser.add_argument("--dev_demonstrations", type=str, default=None)
    parser.add_argument("--test_demonstrations", type=str, default=None)

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
        shared_functions.pop_logger_handler()
        main(args=args)
    else:
        # Training or Evaluation
        main(args=args)

