import argparse
import logging
import os

# import pandas as pd
import torch
import transformers
# import tabulate
# from tqdm import tqdm

import sys
sys.path.insert(0, "../..")
from kapipe.systems import MAQASystem
from kapipe.trainers import MAQATrainer
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
    path_results_dir = args.results_dir
    device = torch.device(f"cuda:{args.gpu}")
    config_path = args.config_path
    config_name = args.config_name
    prefix = args.prefix
    actiontype = args.actiontype
    path_pretrained_model = args.pretrained_model
    path_pretrained_model_vocab_answer = args.pretrained_model_vocab_answer
    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()
        args.prefix = prefix
    assert actiontype in ["train", "evaluate", "check_preprocessing"]

    ##################
    # Set logger
    ##################

    base_output_path = os.path.join(
        path_results_dir,
        "maqa",
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
    vocab_answer = get_vocab_answer()

    # Get supplemental information
    if "redocred" in path_train_documents:
        train_file_name, dev_file_name, test_file_name \
            = shared_functions.get_docred_info(dataset_name="redocred")
    elif "docred" in path_train_documents:
        train_file_name, dev_file_name, test_file_name \
            = shared_functions.get_docred_info(dataset_name="docred")
    else:
        train_file_name = dev_file_name = test_file_name = None
    supplemental_info = {
        # Information for DocRED/Re-DocRED official evaluation
        "original_data_dir": os.path.join(path_train_documents, "..", "original"),
        "train_file_name": train_file_name,
        "dev_file_name": dev_file_name,
        "test_file_name": test_file_name,
    }

    ##################
    # Get system
    ##################

    trainer = MAQATrainer(base_output_path=base_output_path)

    config = utils.get_hocon_config(
        config_path=config_path,
        config_name=config_name
    )
    if actiontype == "train":
        if path_pretrained_model is not None:
            assert path_pretrained_model_vocab_answer is not None
            system = MAQASystem(
                device=device,
                config=config,
                path_entity_dict=path_entity_dict,
                vocab_answer=path_pretrained_model_vocab_answer,
                path_model=path_pretrained_model
            )
        else:
            system = MAQASystem(
                device=device,
                config=config,
                path_entity_dict=path_entity_dict,
                vocab_answer=vocab_answer,
                path_model=None
            )
    else:
        system = MAQASystem(
            device=device,
            config=config,
            path_entity_dict=path_entity_dict,
            vocab_answer=trainer.paths["path_vocab_answer"],
            path_model=trainer.paths["path_snapshot"]
        )

    ##################
    # Train or evaluate
    ##################

    trainer.setup_dataset(
        system=system,
        documents=train_documents,
        split="train"
    )
    trainer.setup_dataset(
        system=system,
        documents=dev_documents,
        split="dev"
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
            split="test"
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
                prediction_only=True
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
            preprocessed_data = system.preprocessor.preprocess(document)
            results.append(preprocessed_data)
        utils.write_json(base_output_path + "/dev.check_preprocessing.json", results)

    ##################
    # Closing
    ##################

    logging.info("Done.")
    sw.stop("main")
    logging.info("Time: %f min." % sw.get_time("main", minute=True))

    return prefix


def get_vocab_answer():
    answers = ["No", "Yes"]
    vocab_answer = {ans: ans_id for ans_id, ans in enumerate(answers)}
    return vocab_answer


# def show_dataset_statistics(dataset, title):
#     n_documents = len(dataset)
#     n_sentences_list = []
#     n_words_list = []
#     n_subtokens_list = []
#     n_segments_list = []
#     n_entities_list = []
#     n_qas_list = []
#     n_positive_qas_list = []
#     n_negative_qas_list = []

#     for data in tqdm(dataset):
#         n_sentences_list.append(len(data.sentences))
#         n_words_list.append(len(utils.flatten_lists(data.sentences)))
#         # n_subtokens_list.append(len(utils.flatten_lists(data.segments)))
#         # n_segments_list.append(len(data.segments))
#         for qa_i in range(len(data.qa_index_to_bert_index)):
#             n_subtokens_list.append(len(utils.flatten_lists(data.qa_index_to_bert_index[qa_i]["segments"])))
#             n_segments_list.append(len(data.qa_index_to_bert_index[qa_i]["segments"]))
#         n_entities_list.append(len(data.entities))
#         n_qas_list.append(len(data.qas))
#         if hasattr(data, "gold_answer_labels"):
#             n_positive_qas = 0
#             n_negative_qas = 0
#             for label in data.gold_answer_labels:
#                 if label == 0:
#                     n_negative_qas += 1
#                 else:
#                     n_positive_qas += 1
#             n_positive_qas_list.append(n_positive_qas)
#             n_negative_qas_list.append(n_negative_qas)
#         else:
#             n_positive_qas_list.append(0)
#             n_negative_qas_list.append(0)

#     results = {}
#     results["Number of documents"] = n_documents
#     results["Number of sentences"] = shared_functions.get_statistics_text(n_sentences_list)
#     results["Number of words"] = shared_functions.get_statistics_text(n_words_list)
#     results["Number of subtokens"] = shared_functions.get_statistics_text(n_subtokens_list)
#     results["Number of segments"] = shared_functions.get_statistics_text(n_segments_list)
#     results["Number of entities"] = shared_functions.get_statistics_text(n_entities_list)
#     results["Number of QAs"] = shared_functions.get_statistics_text(n_qas_list)
#     results["Number of positive QAs"] = shared_functions.get_statistics_text(n_positive_qas_list)
#     results["Number of negative QAs"] = shared_functions.get_statistics_text(n_negative_qas_list)

#     table = {}
#     table[title] = results.keys()
#     table["Statistics"] = results.values()
#     df = pd.DataFrame.from_dict(table)
#     logging.info("\n" + tabulate.tabulate(df, headers="keys", tablefmt="psql", floatfmt=".1f"))


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
    parser.add_argument("--pretrained_model_vocab_answer", type=str, default=None)
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



