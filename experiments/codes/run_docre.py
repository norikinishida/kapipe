import argparse
import logging
import os

import pandas as pd
import torch
import transformers

import sys
sys.path.insert(0, "../..")
from kapipe.triple_extraction import ATLOP, ATLOPTrainer
from kapipe.triple_extraction import MAATLOP, MAATLOPTrainer
from kapipe.triple_extraction import MAQA, MAQATrainer
from kapipe.triple_extraction import LLMDocRE, LLMDocRETrainer
from kapipe import utils
from kapipe.utils import StopWatch

import shared_functions


def main(args):
    torch.autograd.set_detect_anomaly(True)
    transformers.logging.set_verbosity_error()

    sw = StopWatch()
    sw.start("main")

    device = torch.device(f"cuda:{args.gpu}")

    method_name = args.method

    path_train_documents = args.train
    path_dev_documents = args.dev
    path_test_documents = args.test

    # path_train_demonstrations = args.train_demonstrations
    path_dev_demonstrations = args.dev_demonstrations
    path_test_demonstrations = args.test_demonstrations

    path_entity_dict = args.entity_dict

    dataset_name = args.dataset_name

    path_results_dir = args.results_dir

    config_path = args.config_path
    config_name = args.config_name
    prefix = args.prefix

    actiontype = args.actiontype

    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()
        args.prefix = prefix

    assert method_name in ["atlop", "maatlop", "maqa", "llmdocre"]
    assert actiontype in ["train", "evaluate", "check_preprocessing", "check_prompt"]

    ##################
    # Set logger
    ##################

    base_output_path = os.path.join(
        path_results_dir,
        "docre",
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
    # Get documents, vocabulary, and supplemental_info
    ##################

    # Get documents
    train_documents = utils.read_json(path_train_documents)
    dev_documents = utils.read_json(path_dev_documents)
    test_documents = utils.read_json(path_test_documents)

    # Get demonstration information
    if method_name == "llmdocre":
        # train_demonstrations = utils.read_json(path_train_demonstrations)
        dev_demonstrations = utils.read_json(path_dev_demonstrations)
        test_demonstrations = utils.read_json(path_test_demonstrations)

    # Get vocabulary
    if method_name == "maqa":
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

    if method_name == "llmdocre":
        rel_meta_info = {
            row["Relation"]: {
                "Pretty Name": row["Pretty Name"],
                "Definition": row["Definition"]
            }
            for _, row in pd.read_csv(f"./dataset-meta-information/{dataset_name}_relations.csv").iterrows()
        }
 
    # Get supplemental information
    if dataset_name == "docred":
        train_file_name, dev_file_name, test_file_name = get_docred_info(dataset_name="docred")
    elif dataset_name == "redocred":
        train_file_name, dev_file_name, test_file_name = get_docred_info(dataset_name="redocred")
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

    # Show documents statistics
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
    # Get extractor
    ##################

    if method_name == "atlop":

        trainer = ATLOPTrainer(base_output_path=base_output_path)

        config = utils.get_hocon_config(
            config_path=config_path,
            config_name=config_name
        )

        if actiontype == "train" or actiontype == "check_preprocessing":
            extractor = ATLOP(
                device=device,
                config=config,
                vocab_relation=vocab_relation,
                path_model=None
            )
        else:
            extractor = ATLOP(
                device=device,
                config=config,
                vocab_relation=trainer.paths["path_vocab_relation"],
                path_model=trainer.paths["path_snapshot"]
            )

    elif method_name == "maatlop":

        trainer = MAATLOPTrainer(base_output_path=base_output_path)

        config = utils.get_hocon_config(
            config_path=config_path,
            config_name=config_name
        )

        if actiontype == "train":
            extractor = MAATLOP(
                device=device,
                config=config,
                path_entity_dict=path_entity_dict,
                vocab_relation=vocab_relation,
                path_model=None
            )
        else:
            extractor = MAATLOP(
                device=device,
                config=config,
                path_entity_dict=path_entity_dict,
                vocab_relation=trainer.paths["path_vocab_relation"],
                path_model=trainer.paths["path_snapshot"]
            )

    elif method_name == "maqa":

        trainer = MAQATrainer(base_output_path=base_output_path)

        config = utils.get_hocon_config(
            config_path=config_path,
            config_name=config_name
        )

        if actiontype == "train":
            extractor = MAQA(
                device=device,
                config=config,
                path_entity_dict=path_entity_dict,
                vocab_answer=vocab_answer,
                path_model=None
            )
        else:
            extractor = MAQA(
                device=device,
                config=config,
                path_entity_dict=path_entity_dict,
                vocab_answer=trainer.paths["path_vocab_answer"],
                path_model=trainer.paths["path_snapshot"]
            )


    elif method_name == "llmdocre":

        trainer = LLMDocRETrainer(base_output_path=base_output_path)

        config = utils.get_hocon_config(
            config_path=config_path,
            config_name=config_name
        )

        extractor = LLMDocRE(
            device=device,
            config=config,
            #
            vocab_relation=vocab_relation,
            rel_meta_info=rel_meta_info,
            #
            path_entity_dict=path_entity_dict,
            #
            path_demonstration_pool=path_train_documents
        )

    ##################
    # Train or evaluate
    ##################

    if method_name in ["atlop", "maatlop"]:

        # train_documents = remove_documents_without_relations(
        #     train_documents=train_documents,
        # )
        # shared_functions.show_docre_documents_statistics(
        #     documents=train_documents,
        #     vocab_relation=vocab_relation,
        #     with_supervision=True,
        #     title="Training after Removal"
        # )

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
            with_gold_annotations=False if config["dataset_name"] in ["docred", "linked_docred"] else True
        )

        if actiontype == "train":
            trainer.train(
                extractor=extractor,
                train_documents=train_documents,
                dev_documents=dev_documents,
                supplemental_info=supplemental_info
            )

        elif actiontype == "evaluate":
            if config["dataset_name"] == "docred":
                # Dev
                trainer.official_evaluate(
                    extractor=extractor,
                    documents=dev_documents,
                    split="dev",
                    supplemental_info=supplemental_info
                )
                # Test
                trainer.official_evaluate(
                    extractor=extractor,
                    documents=test_documents,
                    split="test",
                    supplemental_info=supplemental_info,
                    #
                    prediction_only=True,
                )
            elif config["dataset_name"] == "redocred":
                # Dev
                trainer.official_evaluate(
                    extractor=extractor,
                    documents=dev_documents,
                    split="dev",
                    supplemental_info=supplemental_info
                )
                # Test
                trainer.official_evaluate(
                    extractor=extractor,
                    documents=test_documents,
                    split="test",
                    supplemental_info=supplemental_info
                )
            elif config["dataset_name"] == "linked_docred":
                # Dev
                trainer.evaluate(
                    extractor=extractor,
                    documents=dev_documents,
                    split="dev",
                    supplemental_info=supplemental_info
                )
                # Test
                trainer.evaluate(
                    extractor=extractor,
                    documents=test_documents,
                    split="test",
                    supplemental_info=supplemental_info,
                    #
                    prediction_only=True
                )
            else:
                # Dev
                trainer.evaluate(
                    extractor=extractor,
                    documents=dev_documents,
                    split="dev",
                    supplemental_info=supplemental_info
                )
                # Test
                trainer.evaluate(
                    extractor=extractor,
                    documents=test_documents,
                    split="test",
                    supplemental_info=supplemental_info
                )

        elif actiontype == "check_preprocessing":
            results = []
            for document in dev_documents:
                if method_name == "maatlop":
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

    elif method_name == "maqa":

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
            with_gold_annotations=False if config["dataset_name"] in ["docred", "linked_docred"] else True
        )
 
        if actiontype == "train":
            trainer.train(
                extractor=extractor,
                train_documents=train_documents,
                dev_documents=dev_documents,
                supplemental_info=supplemental_info
            )

        elif actiontype == "evaluate":
            if config["dataset_name"] == "docred":
                # Dev
                trainer.official_evaluate(
                    extractor=extractor,
                    documents=dev_documents,
                    split="dev",
                    supplemental_info=supplemental_info
                )
                # Test
                trainer.official_evaluate(
                    extractor=extractor,
                    documents=test_documents,
                    split="test",
                    supplemental_info=supplemental_info,
                    #
                    prediction_only=True
                )
            elif config["dataset_name"] == "redocred":
                # Dev
                trainer.official_evaluate(
                    extractor=extractor,
                    documents=dev_documents,
                    split="dev",
                    supplemental_info=supplemental_info
                )
                # Test
                trainer.official_evaluate(
                    extractor=extractor,
                    documents=test_documents,
                    split="test",
                    supplemental_info=supplemental_info
                )
            else:
                # Dev
                trainer.evaluate(
                    extractor=extractor,
                    documents=dev_documents,
                    split="dev",
                    supplemental_info=supplemental_info,
                    #
                    skip_intra_inter=True
                )
                # Test
                trainer.evaluate(
                    extractor=extractor,
                    documents=test_documents,
                    split="test",
                    supplemental_info=supplemental_info,
                    #
                    skip_intra_inter=True
                )

        elif actiontype == "check_preprocessing":
            results = []
            for document in dev_documents:
                preprocessed_data = extractor.model.preprocessor.preprocess(document)
                results.append(preprocessed_data)
            utils.write_json(base_output_path + "/dev.check_preprocessing.json", results)

    elif method_name == "llmdocre":

        if actiontype == "check_prompt":
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
            with_gold_annotations=False if config["dataset_name"] in ["docred", "linked_docred"] else True
        )

        trainer.save_extractor(extractor=extractor)

        if actiontype == "evaluate":
            if config["dataset_name"] == "docred":
                # Dev
                trainer.official_evaluate(
                    extractor=extractor,
                    documents=dev_documents,
                    demonstrations=dev_demonstrations,
                    contexts=None,
                    split="dev",
                    supplemental_info=supplemental_info
                )
                # Test
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
            elif config["dataset_name"] == "redocred":
                # Dev
                trainer.official_evaluate(
                    extractor=extractor,
                    documents=dev_documents,
                    demonstrations=dev_demonstrations,
                    contexts=None,
                    split="dev",
                    supplemental_info=supplemental_info
                )
                # Test
                trainer.official_evaluate(
                    extractor=extractor,
                    documents=test_documents,
                    demonstrations=test_demonstrations,
                    contexts=None,
                    split="test",
                    supplemental_info=supplemental_info
                )
            elif config["dataset_name"] == "linked_docred":
                # Dev
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
                # Test
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
                # Dev
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
                # Test
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
    if method_name in ["atlop", "maatlop"]:
        relations = ["NO-REL"] + relations
    vocab_relation = {rel: rel_id for rel_id, rel in enumerate(relations)}
    return vocab_relation


def get_vocab_relation_for_cdr(method_name):
    relations = ["CID"] # CDR contains only one relationship
    if method_name in ["atlop", "maatlop"]:
        relations = ["NO-REL"] + relations
    vocab_relation = {rel: rel_id for rel_id, rel in enumerate(relations)}
    return vocab_relation


def get_vocab_relation_for_docred(path, method_name):
    original_rel2id = utils.read_json(path)
    relations = [(rel, rel_i) for rel, rel_i in original_rel2id.items()]
    relations = sorted(relations, key=lambda tpl: tpl[1])
    assert relations[0] == ("Na", 0)
    relations = [rel for rel, rel_i in relations[1:]]
    if method_name in ["atlop", "maatlop"]:
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
    if method_name in ["atlop", "maatlop"]:
        relations = ["NO-REL"] + relations
    vocab_relation = {rel: rel_id for rel_id, rel in enumerate(relations)}
    return vocab_relation

   
def get_vocab_relation_for_linked_docred(path, method_name):
    original_rel2id = utils.read_json(path)
    relations = [(rel, rel_i) for rel, rel_i in original_rel2id.items()]
    relations = sorted(relations, key=lambda tpl: tpl[1])
    assert relations[0] == ("Na", 0)
    relations = [rel for rel, rel_i in relations[1:]]
    if method_name in ["atlop", "maatlop"]:
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
    if method_name in ["atlop", "maatlop"]:
        relations = ["NO-REL"] + relations
    vocab_relation = {rel: rel_id for rel_id, rel in enumerate(relations)}
    return vocab_relation


def get_vocab_relation_for_redocred(path, method_name):
    original_rel2id = utils.read_json(path)
    relations = [(rel, rel_i) for rel, rel_i in original_rel2id.items()]
    relations = sorted(relations, key=lambda tpl: tpl[1])
    assert relations[0] == ("Na", 0)
    relations = [rel for rel, rel_i in relations[1:]]
    if method_name in ["atlop", "maatlop"]:
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

    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--method", type=str, required=True)

    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--dev", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)

    parser.add_argument("--train_demonstrations", type=str, default=None)
    parser.add_argument("--dev_demonstrations", type=str, default=None)
    parser.add_argument("--test_demonstrations", type=str, default=None)

    parser.add_argument("--entity_dict", type=str, default=None)

    parser.add_argument("--dataset_name", type=str, default=None)

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

