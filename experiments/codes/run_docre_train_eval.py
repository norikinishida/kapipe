import argparse
import logging
import os

import pandas as pd
import torch
import transformers

import sys
sys.path.insert(0, "../..")
from kapipe.docre import (
    ATLOP, ATLOPTrainer,
    MAATLOP, MAATLOPTrainer,
    MAQA, MAQATrainer,
    LLMDocRE, LLMDocRETrainer
)
from kapipe import utils
from kapipe.utils import StopWatch

import shared_functions


def main(args):
    torch.autograd.set_detect_anomaly(True)
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
    path_entity_dict = args.entity_dict

    # Output Path
    path_results_dir = args.results_dir
    prefix = args.prefix
    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()
        args.prefix = prefix

    # Action
    actiontype = args.actiontype

    assert method_name in ["atlop", "ma_atlop", "ma_qa", "llm_docre"]
    assert actiontype in ["train", "evaluate", "check_preprocessing", "check_prompt"]

    ##################
    # Logging Setup
    ##################

    # Set base output path
    base_output_path = os.path.join(
        path_results_dir,
        "docre",
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
    if method_name == "llm_docre":
        # train_demonstrations = utils.read_json(path_train_demonstrations)
        dev_demonstrations = utils.read_json(path_dev_demonstrations)
        test_demonstrations = utils.read_json(path_test_demonstrations)

    # Create vocabulary of relation labels
    if method_name == "ma_qa":
        vocab_answer = get_vocab_answer()
    elif dataset_name == "cdr":
        vocab_relation = get_vocab_relation_for_cdr(method_name=method_name)
    elif dataset_name == "docred":
        vocab_relation = get_vocab_relation_for_docred(
            path=os.path.join(
                os.path.dirname(path_train_documents),
                "meta", "rel2id.json"
            ),
            method_name=method_name
        )
    elif dataset_name == "gda":
        vocab_relation = get_vocab_relation_for_gda(method_name=method_name)
    elif dataset_name == "hoip":
        vocab_relation = get_vocab_relation_for_hoip(method_name=method_name)
    elif dataset_name == "linked_docred":
        vocab_relation = get_vocab_relation_for_linked_docred(
            path=os.path.join(
                os.path.dirname(path_train_documents),
                "meta", "rel2id.json"
            ),
            method_name=method_name
        )
    elif dataset_name == "medmentions_dsrel":
        vocab_relation = get_vocab_relation_for_medmentions_dsrel(
            path="./dataset-meta-information/medmentions_dsrel_relations.csv",
            method_name=method_name
        )
    elif dataset_name == "redocred":
        vocab_relation = get_vocab_relation_for_redocred(
            path=os.path.join(
                os.path.dirname(path_train_documents),
                "meta", "rel2id.json"
            ),
            method_name=method_name
        )
    else:
        vocab_relation = get_vocab_relation(
            documents_list=[
                train_documents,
                dev_documents,
                test_documents
            ],
            method_name=method_name
        )

    # Load meta information (e.g., pretty labels and definitions) for relation labels
    if method_name == "llm_docre":
        rel_meta_info = {
            row["Relation"]: {
                "Pretty Name": row["Pretty Name"],
                "Definition": row["Definition"]
            }
            for _, row in pd.read_csv(f"./dataset-meta-information/{dataset_name}_relations.csv").iterrows()
        }
 
    # Create supplemental information for DocRED/Re-DocRED official evaluation
    if dataset_name == "docred":
        train_file_name, dev_file_name, test_file_name = get_docred_info(dataset_name="docred")
    elif dataset_name == "redocred":
        train_file_name, dev_file_name, test_file_name = get_docred_info(dataset_name="redocred")
    else:
        train_file_name = dev_file_name = test_file_name = None
    supplemental_info = {
        "original_data_dir": os.path.join(path_train_documents, "..", "original"),
        "train_file_name": train_file_name,
        "dev_file_name": dev_file_name,
        "test_file_name": test_file_name,
    }

    # Show statistics
    shared_functions.show_docre_documents_statistics(
        documents=train_documents,
        vocab_relation=vocab_relation,
        with_supervision=True,
        title="Training"
    )
    shared_functions.show_docre_documents_statistics(
        documents=dev_documents,
        vocab_relation=vocab_relation,
        with_supervision=True,
        title="Development"
    )
    shared_functions.show_docre_documents_statistics(
        documents=test_documents,
        vocab_relation=vocab_relation,
        with_supervision=False if dataset_name in ["docred", "linked_docred"] else True,
        title="Test"
    )

    ##################
    # Method
    ##################

    if method_name == "atlop":
        # Initialize the trainer (evaluator)
        trainer = ATLOPTrainer(base_output_path=base_output_path)

        if actiontype == "train" or actiontype == "check_preprocessing":
            # Initialize the extractor
            config = utils.get_hocon_config(
                config_path=config_path,
                config_name=config_name
            )
            extractor = ATLOP(
                device=device,
                config=config,
                vocab_relation=vocab_relation
            )
        else:
            # Load the extractor
            extractor = ATLOP(
                device=device,
                path_snapshot=trainer.paths["path_snapshot"]
            )

    elif method_name == "ma_atlop":
        # Initialize the extractor
        trainer = MAATLOPTrainer(base_output_path=base_output_path)

        if actiontype == "train":
            # Initialize the extractor
            config = utils.get_hocon_config(
                config_path=config_path,
                config_name=config_name
            )
            extractor = MAATLOP(
                device=device,
                config=config,
                vocab_relation=vocab_relation,
                path_entity_dict=path_entity_dict
            )
        else:
            # Load the extractor
            extractor = MAATLOP(
                device=device,
                path_snapshot=trainer.paths["path_snapshot"]
            )

    elif method_name == "ma_qa":
        # Initialize the trainer (evaluator)
        trainer = MAQATrainer(base_output_path=base_output_path)

        if actiontype == "train":
            # Initialize the extractor
            config = utils.get_hocon_config(
                config_path=config_path,
                config_name=config_name
            )
            extractor = MAQA(
                device=device,
                config=config,
                vocab_answer=vocab_answer,
                path_entity_dict=path_entity_dict
            )
        else:
            # Load the extractor
            extractor = MAQA(
                device=device,
                path_snapshot=trainer.paths["path_snapshot"]
            )

    elif method_name == "llm_docre":
        # Initialize the trainer (evaluator)
        trainer = LLMDocRETrainer(base_output_path=base_output_path)

        # Initialize the extractor
        config = utils.get_hocon_config(
            config_path=config_path,
            config_name=config_name
        )
        extractor = LLMDocRE(
            device=device,
            config=config,
            vocab_relation=vocab_relation,
            rel_meta_info=rel_meta_info,
            path_entity_dict=path_entity_dict,
            path_demonstration_pool=path_train_documents
        )

    ##################
    # Training, Evaluation
    ##################

    if method_name in ["atlop", "ma_atlop"]:

        # train_documents = remove_documents_without_relations(
        #     train_documents=train_documents,
        # )
        # shared_functions.show_docre_documents_statistics(
        #     documents=train_documents,
        #     vocab_relation=vocab_relation,
        #     with_supervision=True,
        #     title="Training after Removal"
        # )

        # Set up the datasets for evaluation
        trainer.setup_dataset(
            extractor=extractor,
            documents=train_documents,
            split="train",
            with_gold_annotations=True
        )
        trainer.setup_dataset(
            extractor=extractor,
            documents=dev_documents,
            split="dev",
            with_gold_annotations=True
        )
        trainer.setup_dataset(
            extractor=extractor,
            documents=test_documents,
            split="test",
            with_gold_annotations=(
                False if extractor.config["dataset_name"] in [
                    "docred", "linked_docred"
                ] else True
            )
        )

        if actiontype == "train":
            # Train the extractor
            trainer.train(
                extractor=extractor,
                train_documents=train_documents,
                dev_documents=dev_documents,
                supplemental_info=supplemental_info
            )

        if actiontype == "evaluate":
            # Evaluate the extractor on the datasets
            if extractor.config["dataset_name"] == "docred":
                trainer.official_evaluate(
                    extractor=extractor,
                    documents=dev_documents,
                    split="dev",
                    supplemental_info=supplemental_info
                )
                trainer.official_evaluate(
                    extractor=extractor,
                    documents=test_documents,
                    split="test",
                    supplemental_info=supplemental_info,
                    #
                    prediction_only=True,
                )
            elif extractor.config["dataset_name"] == "redocred":
                trainer.official_evaluate(
                    extractor=extractor,
                    documents=dev_documents,
                    split="dev",
                    supplemental_info=supplemental_info
                )
                trainer.official_evaluate(
                    extractor=extractor,
                    documents=test_documents,
                    split="test",
                    supplemental_info=supplemental_info
                )
            elif extractor.config["dataset_name"] == "linked_docred":
                trainer.evaluate(
                    extractor=extractor,
                    documents=dev_documents,
                    split="dev",
                    supplemental_info=supplemental_info
                )
                trainer.evaluate(
                    extractor=extractor,
                    documents=test_documents,
                    split="test",
                    supplemental_info=supplemental_info,
                    #
                    prediction_only=True
                )
            else:
                trainer.evaluate(
                    extractor=extractor,
                    documents=dev_documents,
                    split="dev",
                    supplemental_info=supplemental_info
                )
                trainer.evaluate(
                    extractor=extractor,
                    documents=test_documents,
                    split="test",
                    supplemental_info=supplemental_info
                )

        if actiontype == "check_preprocessing":
            # Save preprocessed data
            results = []
            for document in dev_documents:
                if method_name == "ma_atlop":
                    if extractor.config["do_negative_entity_sampling"]:
                        document = extractor.sample_negative_entities_randomly(
                            document=document,
                            sample_size=round(
                                len(document["entities"])
                                * extractor.config["negative_entity_ratio"]
                            )
                        )
                preprocessed_data = extractor.model.preprocessor.preprocess(document)
                preprocessed_data["pair_head_entity_indices"] = preprocessed_data["pair_head_entity_indices"].tolist()
                preprocessed_data["pair_tail_entity_indices"] = preprocessed_data["pair_tail_entity_indices"].tolist()
                preprocessed_data["pair_gold_relation_labels"] = preprocessed_data["pair_gold_relation_labels"].tolist()
                results.append(preprocessed_data)
            utils.write_json(base_output_path + "/dev.check_preprocessing.json", results)

    elif method_name == "ma_qa":
        # Set up the datasets
        trainer.setup_dataset(
            extractor=extractor,
            documents=train_documents,
            split="train"
        )
        trainer.setup_dataset(
            extractor=extractor,
            documents=dev_documents,
            split="dev"
        )
        trainer.setup_dataset(
            extractor=extractor,
            documents=test_documents,
            split="test",
            with_gold_annotations=(
                False if extractor.config["dataset_name"] in [
                    "docred", "linked_docred"
                ] else True
            )
        )
 
        if actiontype == "train":
            # Train the extractor
            trainer.train(
                extractor=extractor,
                train_documents=train_documents,
                dev_documents=dev_documents,
                supplemental_info=supplemental_info
            )

        if actiontype == "evaluate":
            # Evaluate the extractor on the datasets
            if extractor.config["dataset_name"] == "docred":
                trainer.official_evaluate(
                    extractor=extractor,
                    documents=dev_documents,
                    split="dev",
                    supplemental_info=supplemental_info
                )
                trainer.official_evaluate(
                    extractor=extractor,
                    documents=test_documents,
                    split="test",
                    supplemental_info=supplemental_info,
                    #
                    prediction_only=True
                )
            elif extractor.config["dataset_name"] == "redocred":
                trainer.official_evaluate(
                    extractor=extractor,
                    documents=dev_documents,
                    split="dev",
                    supplemental_info=supplemental_info
                )
                trainer.official_evaluate(
                    extractor=extractor,
                    documents=test_documents,
                    split="test",
                    supplemental_info=supplemental_info
                )
            else:
                trainer.evaluate(
                    extractor=extractor,
                    documents=dev_documents,
                    split="dev",
                    supplemental_info=supplemental_info,
                    #
                    skip_intra_inter=True
                )
                trainer.evaluate(
                    extractor=extractor,
                    documents=test_documents,
                    split="test",
                    supplemental_info=supplemental_info,
                    #
                    skip_intra_inter=True
                )

        elif actiontype == "check_preprocessing":
            # Save preprocessed data
            results = []
            for document in dev_documents:
                preprocessed_data = extractor.model.preprocessor.preprocess(document)
                results.append(preprocessed_data)
            utils.write_json(base_output_path + "/dev.check_preprocessing.json", results)

    elif method_name == "llm_docre":

        if actiontype == "check_prompt":
            # Show prompts
            with torch.no_grad():
                path_out = base_output_path + "/output.txt"
                with open(path_out, "w") as f:
                    for i, (document, demos) in enumerate(zip(
                        dev_documents, dev_demonstrations
                    )):
                        doc_key = document["doc_key"]
                        logging.info(f"Processing {doc_key}")
                        result_document = extractor.extract(
                            document=document,
                            demonstrations_for_doc=demos
                        )
                        f.write(f"--- DOC_KEY ({doc_key}) ---\n\n")
                        f.write("Prompt:\n")
                        f.write(result_document["docre_prompt"] + "\n\n")
                        f.write("-----\n")
                        f.write("Generated Text:\n")
                        f.write(result_document["docre_generated_text"] + "\n\n")
                        f.write("-----\n")
                        f.write("Parsed Triples:\n")
                        for t in result_document["relations"]:
                            f.write(f"{t}\n")
                        f.write("-----\n")
                        f.write("Gold Triples:\n")
                        for t in document["relations"]:
                            f.write(f"{t}\n")
                        f.write("-----\n")
                        f.flush()
                        if i > 5:
                            break
                return

        # Set up the datasets for evaluation
        trainer.setup_dataset(
            extractor=extractor,
            documents=train_documents,
            split="train",
            with_gold_annotations=True
        )
        trainer.setup_dataset(
            extractor=extractor,
            documents=dev_documents,
            split="dev",
            with_gold_annotations=True
        )
        trainer.setup_dataset(
            extractor=extractor,
            documents=test_documents,
            split="test",
            with_gold_annotations=(
                False if extractor.config["dataset_name"] in [
                    "docred", "linked_docred"
                ] else True
            )
        )

        # Save the configurations of the extractor
        trainer.save_extractor(extractor=extractor)

        if actiontype == "evaluate":
            # Evaluate the extractor on the datasets
            if extractor.config["dataset_name"] == "docred":
                trainer.official_evaluate(
                    extractor=extractor,
                    documents=dev_documents,
                    demonstrations=dev_demonstrations,
                    contexts=None,
                    split="dev",
                    supplemental_info=supplemental_info
                )
                trainer.official_evaluate(
                    extractor=extractor,
                    documents=test_documents,
                    demonstrations=test_demonstrations,
                    contexts=None,
                    split="test",
                    supplemental_info=supplemental_info,
                    #
                    prediction_only=True
                )
            elif extractor.config["dataset_name"] == "redocred":
                trainer.official_evaluate(
                    extractor=extractor,
                    documents=dev_documents,
                    demonstrations=dev_demonstrations,
                    contexts=None,
                    split="dev",
                    supplemental_info=supplemental_info
                )
                trainer.official_evaluate(
                    extractor=extractor,
                    documents=test_documents,
                    demonstrations=test_demonstrations,
                    contexts=None,
                    split="test",
                    supplemental_info=supplemental_info
                )
            elif extractor.config["dataset_name"] == "linked_docred":
                trainer.evaluate(
                    extractor=extractor,
                    documents=dev_documents,
                    demonstrations=dev_demonstrations,
                    contexts=None,
                    split="dev",
                    supplemental_info=supplemental_info,
                    #
                    skip_intra_inter=True,
                    skip_ign=True,
                )
                trainer.evaluate(
                    extractor=extractor,
                    documents=test_documents,
                    demonstrations=test_demonstrations,
                    contexts=None,
                    split="test",
                    supplemental_info=supplemental_info,
                    #
                    skip_intra_inter=True,
                    skip_ign=True,
                    #
                    prediction_only=True
                )
            else:
                trainer.evaluate(
                    extractor=extractor,
                    documents=dev_documents,
                    demonstrations=dev_demonstrations,
                    contexts=None,
                    split="dev",
                    supplemental_info=supplemental_info,
                    #
                    skip_intra_inter=True,
                    skip_ign=True
                )
                trainer.evaluate(
                    extractor=extractor,
                    documents=test_documents,
                    demonstrations=test_demonstrations,
                    contexts=None,
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


def get_vocab_relation(documents_list, method_name):
    relations = set()
    for documents in documents_list:
        for document in documents:
            for triple in document["relations"]:
                relations.add(triple["relation"])
    relations = sorted(list(relations))
    if method_name in ["atlop", "ma_atlop"]:
        relations = ["NO-REL"] + relations
    vocab_relation = {rel: rel_id for rel_id, rel in enumerate(relations)}
    return vocab_relation


def get_vocab_relation_for_cdr(method_name):
    relations = ["CID"] # CDR contains only one relationship
    if method_name in ["atlop", "ma_atlop"]:
        relations = ["NO-REL"] + relations
    vocab_relation = {rel: rel_id for rel_id, rel in enumerate(relations)}
    return vocab_relation


def get_vocab_relation_for_docred(path, method_name):
    original_rel2id = utils.read_json(path)
    relations = [(rel, rel_i) for rel, rel_i in original_rel2id.items()]
    relations = sorted(relations, key=lambda tpl: tpl[1])
    assert relations[0] == ("Na", 0)
    relations = [rel for rel, rel_i in relations[1:]]
    if method_name in ["atlop", "ma_atlop"]:
        relations = ["NO-REL"] + relations
    vocab_relation = {rel: rel_id for rel_id, rel in enumerate(relations)}
    return vocab_relation


def get_vocab_relation_for_gda(method_name):
    relations = ["GDA"] # GDA contains only one relationship
    if method_name in ["atlop", "maatop"]:
        relations = ["NO-REL"] + relations
    vocab_relation = {rel: rel_id for rel_id, rel in enumerate(relations)}
    return vocab_relation


def get_vocab_relation_for_hoip(method_name):
    # relations = [
    #     "has result", "has part", "has molecular reaction", "part of"
    # ]
    relations = ["has result", "has part"]
    if method_name in ["atlop", "ma_atlop"]:
        relations = ["NO-REL"] + relations
    vocab_relation = {rel: rel_id for rel_id, rel in enumerate(relations)}
    return vocab_relation

   
def get_vocab_relation_for_linked_docred(path, method_name):
    original_rel2id = utils.read_json(path)
    relations = [(rel, rel_i) for rel, rel_i in original_rel2id.items()]
    relations = sorted(relations, key=lambda tpl: tpl[1])
    assert relations[0] == ("Na", 0)
    relations = [rel for rel, rel_i in relations[1:]]
    if method_name in ["atlop", "ma_atlop"]:
        relations = ["NO-REL"] + relations
    vocab_relation = {rel: rel_id for rel_id, rel in enumerate(relations)}
    return vocab_relation

    
def get_vocab_relation_for_medmentions_dsrel(path, method_name):
    rel_meta_info = {
        row["Relation"]: {
            "Pretty Name": row["Pretty Name"],
            "Definition": row["Definition"]
        }
        for _, row in pd.read_csv(path).iterrows()
    }
    rel_meta_info = {
        k:v for k,v in rel_meta_info.items() 
        if k not in [
            "is_a",
            "is_sibling_with",
            "is_equivalent_to",
            "is_classified_as",
            "has_variant",
            "misc."
        ]
    }
    relations = list(rel_meta_info.keys())
    if method_name in ["atlop", "ma_atlop"]:
        relations = ["NO-REL"] + relations
    vocab_relation = {rel: rel_id for rel_id, rel in enumerate(relations)}
    return vocab_relation


def get_vocab_relation_for_redocred(path, method_name):
    original_rel2id = utils.read_json(path)
    relations = [(rel, rel_i) for rel, rel_i in original_rel2id.items()]
    relations = sorted(relations, key=lambda tpl: tpl[1])
    assert relations[0] == ("Na", 0)
    relations = [rel for rel, rel_i in relations[1:]]
    if method_name in ["atlop", "ma_atlop"]:
        relations = ["NO-REL"] + relations
    vocab_relation = {rel: rel_id for rel_id, rel in enumerate(relations)}
    return vocab_relation


def get_vocab_answer():
    answers = ["No", "Yes"]
    vocab_answer = {ans: ans_id for ans_id, ans in enumerate(answers)}
    return vocab_answer

    
def remove_documents_without_relations(train_documents):
    new_train_documents = [d for d in train_documents if len(d["relations"]) > 0]
    logging.info(f"Removed {len(train_documents) - len(new_train_documents)} documents without triples in the training dataset: {len(train_documents)} -> {len(new_train_documents)}")
    return new_train_documents


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
    parser.add_argument("--entity_dict", type=str, default=None)

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

