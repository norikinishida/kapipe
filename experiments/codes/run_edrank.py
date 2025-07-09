import argparse
import copy
import logging
import os
import random

# import numpy as np
import torch
import transformers
from tqdm import tqdm

import sys
sys.path.insert(0, "../..")
from kapipe.triple_extraction import (
    BlinkCrossEncoder, BlinkCrossEncoderTrainer,
    LLMED, LLMEDTrainer
)
from kapipe import evaluation
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
    path_train_candidate_entities = args.train_candidate_entities
    path_dev_candidate_entities = args.dev_candidate_entities
    path_test_candidate_entities = args.test_candidate_entities
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

    assert method_name in ["blinkcrossencoder", "llmed"]
    assert actiontype in ["train", "evaluate", "check_preprocessing", "check_prompt"]

    ##################
    # Logging Setup
    ##################

    # Set base output path
    base_output_path = os.path.join(
        path_results_dir,
        "edrank",
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

    # Load candidate entities
    train_candidate_entities = utils.read_json(path_train_candidate_entities)
    dev_candidate_entities = utils.read_json(path_dev_candidate_entities)
    test_candidate_entities = utils.read_json(path_test_candidate_entities)

    # Load demonstrations (for LLM and in-context learning)
    if method_name == "llmed":
        # train_demonstrations = utils.read_json(path_train_demonstrations)
        dev_demonstrations = utils.read_json(path_dev_demonstrations)
        test_demonstrations = utils.read_json(path_test_demonstrations)

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

    if method_name == "blinkcrossencoder":
        # Initialize the trainer (evaluator)
        trainer = BlinkCrossEncoderTrainer(base_output_path=base_output_path)

        if actiontype == "train" or actiontype == "check_preprocessing":
            # Initialize the extractor
            config = utils.get_hocon_config(
                config_path=config_path,
                config_name=config_name
            )
            extractor = BlinkCrossEncoder(
                device=device,
                config=config,
                path_entity_dict=path_entity_dict
            )
        else:
            # Load the extractor
            extractor = BlinkCrossEncoder(
                device=device,
                path_snapshot=trainer.paths["path_snapshot"]
            )

    elif method_name == "llmed":
        # Initialize the extractor
        trainer = LLMEDTrainer(base_output_path=base_output_path)

        # Initialize the extractor
        config = utils.get_hocon_config(
            config_path=config_path,
            config_name=config_name
        )
        extractor = LLMED(
            device=device,
            config=config,
            path_entity_dict=path_entity_dict,
            path_demonstration_pool=path_train_documents,
            path_candidate_entities_pool=path_train_candidate_entities
        )

    ##################
    # Training, Evaluation
    ##################

    if method_name == "blinkcrossencoder":

        # Remove out-of-kb mentions in the training dataset
        (
            processed_train_documents,
            processed_train_candidate_entities
        ) = remove_out_of_kb_mentions(
            train_documents=train_documents,
            train_candidate_entities=train_candidate_entities,
            entity_dict=extractor.entity_dict
        )
        shared_functions.show_ed_documents_statistics(
            documents=processed_train_documents,
            title="Training after Out-of-Kb Removal"
        )

        # Evaluate the candidate entities for the training dataset
        logging.info(utils.pretty_format_dict(
            evaluation.ed.recall_at_k(
                pred_path=processed_train_candidate_entities,
                gold_path=processed_train_documents,
                inkb=False
            )
        ))

        # Add/move gold entities in the candidate entities for the training dataset
        processed_train_candidate_entities = add_or_move_gold_entity_in_candidates(
            documents=processed_train_documents,
            candidate_entities=processed_train_candidate_entities
        )

        # Re-evaluate the candidate entities for the training dataset
        logging.info(utils.pretty_format_dict(
            evaluation.ed.recall_at_k(
                pred_path=processed_train_candidate_entities,
                gold_path=processed_train_documents,
                inkb=False
            )
        ))

        # Set up the datasets for evaluation
        trainer.setup_dataset(
            extractor=extractor,
            documents=dev_documents,
            candidate_entities=dev_candidate_entities,
            split="dev"
        )
        trainer.setup_dataset(
            extractor=extractor,
            documents=test_documents,
            candidate_entities=test_candidate_entities,
            split="test"
        )

        # Evaluate the candidate entities for the development dataset
        logging.info(utils.pretty_format_dict(
            evaluation.ed.recall_at_k(
                pred_path=dev_candidate_entities,
                gold_path=trainer.paths["path_dev_gold"],
                inkb=True
            ) | evaluation.ed.accuracy(
                pred_path=path_dev_candidate_entities.replace(
                    ".pred_candidate_entities.", ".pred."
                ),
                gold_path=trainer.paths["path_dev_gold"],
                inkb=True
            ) | evaluation.ed.fscore(
                pred_path=path_dev_candidate_entities.replace(
                    ".pred_candidate_entities.", ".pred."
                ),
                gold_path=trainer.paths["path_dev_gold"],
                inkb=True
            )
        ))

        # Evaluate the candidate entities for the test dataset
        logging.info(utils.pretty_format_dict(
            evaluation.ed.recall_at_k(
                pred_path=test_candidate_entities,
                gold_path=trainer.paths["path_test_gold"],
                inkb=True
            ) | evaluation.ed.accuracy(
                pred_path=path_test_candidate_entities.replace(
                    ".pred_candidate_entities.", ".pred."
                ),
                gold_path=trainer.paths["path_test_gold"],
                inkb=True
            ) | evaluation.ed.fscore(
                pred_path=path_test_candidate_entities.replace(
                    ".pred_candidate_entities.", ".pred."
                ),
                gold_path=trainer.paths["path_test_gold"],
                inkb=True
            )
        )) 

        if actiontype == "train":
            # Train the extractor
            trainer.train(
                extractor=extractor,
                train_documents=processed_train_documents,
                train_candidate_entities=processed_train_candidate_entities,
                dev_documents=dev_documents,
                dev_candidate_entities=dev_candidate_entities
            )

        elif actiontype == "evaluate":
            # Evaluate the extractor on the datasets
            trainer.evaluate(
                extractor=extractor,
                documents=dev_documents,
                candidate_entities=dev_candidate_entities,
                split="dev"
            )
            trainer.evaluate(
                extractor=extractor,
                documents=test_documents,
                candidate_entities=test_candidate_entities,
                split="test"
            )

        elif actiontype == "check_preprocessing":
            # Save preprocessed data
            results = []
            for document, candidate_entities_for_doc in zip(dev_documents, dev_candidate_entities):
                preprocessed_data = extractor.model.preprocessor.preprocess(
                    document=document,
                    candidate_entities_for_doc=candidate_entities_for_doc,
                    max_n_candidates=3
                )
                results.append(preprocessed_data)
            utils.write_json(base_output_path + "/dev.check_preprocessing.json", results)

    elif method_name == "llmed":

        if actiontype == "check_prompt":
            # Show prompts
            with torch.no_grad():
                path_out = os.path.join(base_output_path, "output.txt")
                with open(path_out, "w") as f:
                    for i, (document, demos,cands) in enumerate(zip(
                        dev_documents, dev_demonstrations, dev_candidate_entities
                    )):
                        doc_key = document["doc_key"]
                        logging.info(f"Processing {doc_key}")
                        document = extractor.extract(
                            document=document,
                            demonstrations_for_doc=demos,
                            candidate_entities_for_doc=cands
                        )
                        f.write(f"--- DOC_KEY ({doc_key}) ---\n\n")
                        f.write("Prompt:\n")
                        f.write(document["ed_prompt"] + "\n\n")
                        f.write("-----\n")
                        f.write("Generated Text:\n")
                        f.write(document["ed_generated_text"] + "\n\n")
                        f.write("-----\n")
                        f.write("Parsed mention-entity pairs:\n")
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
            candidate_entities=dev_candidate_entities,
            split="dev"
        )
        trainer.setup_dataset(
            extractor=extractor,
            documents=test_documents,
            candidate_entities=test_candidate_entities,
            split="test"
        )

        # Evaluate the candidate entities for the development dataset
        logging.info(utils.pretty_format_dict(
            evaluation.ed.recall_at_k(
                pred_path=dev_candidate_entities,
                gold_path=trainer.paths["path_dev_gold"],
                inkb=True
            ) | evaluation.ed.accuracy(
                pred_path=path_dev_candidate_entities.replace(
                    ".pred_candidate_entities.", ".pred."
                ),
                gold_path=trainer.paths["path_dev_gold"],
                inkb=True
            ) | evaluation.ed.fscore(
                pred_path=path_dev_candidate_entities.replace(
                    ".pred_candidate_entities.", ".pred."
                ),
                gold_path=trainer.paths["path_dev_gold"],
                inkb=True
            )
        ))

        # Evaluate the candidate entities for the test dataset
        logging.info(utils.pretty_format_dict(
            evaluation.ed.recall_at_k(
                pred_path=test_candidate_entities,
                gold_path=trainer.paths["path_test_gold"],
                inkb=True
            ) | evaluation.ed.accuracy(
                pred_path=path_test_candidate_entities.replace(
                    ".pred_candidate_entities.", ".pred."
                ),
                gold_path=trainer.paths["path_test_gold"],
                inkb=True
            ) | evaluation.ed.fscore(
                pred_path=path_test_candidate_entities.replace(
                    ".pred_candidate_entities.", ".pred."
                ),
                gold_path=trainer.paths["path_test_gold"],
                inkb=True
            )
        )) 

        # Save the configurations of the extractor
        trainer.save_extractor(extractor=extractor)

        if actiontype == "evaluate":
            # Evaluate the extractor on the datasets
            trainer.evaluate(
                extractor=extractor,
                documents=dev_documents,
                candidate_entities=dev_candidate_entities,
                demonstrations=dev_demonstrations,
                contexts=None,
                split="dev"
            )
            trainer.evaluate(
                extractor=extractor,
                documents=test_documents,
                candidate_entities=test_candidate_entities,
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


def remove_out_of_kb_mentions(
    train_documents,
    train_candidate_entities,
    entity_dict
):
    new_train_documents = []
    new_train_candidate_entities = []

    kb_entity_ids = set(list(entity_dict.keys()))

    n_prev_mentions = 0
    n_new_mentions = 0
    n_prev_entities = 0
    n_new_entities = 0

    for doc_i in tqdm(
        range(len(train_documents)),
        desc="removing out-of-kb mentions"
    ):
        # Copy data
        doc = copy.deepcopy(train_documents[doc_i])
        candidate_entities_for_doc = copy.deepcopy(train_candidate_entities[doc_i])

        assert doc["doc_key"] == candidate_entities_for_doc["doc_key"]

        # Remove out-of-kb mentions and their candidate entities
        n_prev_mentions += len(doc["mentions"])
        candidate_entities_for_mentions = [
            cs for cs, m in zip(
                candidate_entities_for_doc["candidate_entities"], doc["mentions"]
            )
            if m["entity_id"] in kb_entity_ids
        ]
        mentions = [m for m in doc["mentions"] if m["entity_id"] in kb_entity_ids]
        n_new_mentions += len(mentions)

        # Reset mentions and candidate entities
        doc["mentions"] = mentions
        candidate_entities_for_doc["candidate_entities"] = candidate_entities_for_mentions

        assert len(mentions) == len(candidate_entities_for_mentions)
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
 
        # new_train_documents[doc_i] = doc
        # new_train_candidate_entities[doc_i] = candidate_entities
        new_train_documents.append(doc)
        new_train_candidate_entities.append(candidate_entities_for_doc)

    logging.info(f"Removed {n_prev_mentions - n_new_mentions}({n_prev_entities - n_new_entities}) out-of-kb mentions (entities) in the training dataset: {n_prev_mentions} ({n_prev_entities}) -> {n_new_mentions} ({n_new_entities})")
    logging.info(f"Removed {len(train_documents) - len(new_train_documents)} documents with only out-of-kb mentions/entities in the training dataset: {len(train_documents)} -> {len(new_train_documents)}")

    return new_train_documents, new_train_candidate_entities


def add_or_move_gold_entity_in_candidates(
    documents,
    candidate_entities
):
    result_candidate_entities = []
    count_add = 0
    count_move = 0
    count_mentions = 0
    top_k = 0
    for document, cands_for_doc in zip(documents, candidate_entities):
        result_cands_for_mentions = []

        mentions = document["mentions"]
        cands_for_mentions = cands_for_doc["candidate_entities"]
        assert len(mentions) == len(cands_for_mentions)

        original_gold_entity_rank_list = [] # list[float]

        for mention, cands_for_mention in zip(mentions, cands_for_mentions):
            top_k = len(cands_for_mention)

            # Remove the gold entity in the candidates
            gold_entity_id = mention["entity_id"]
            cand_entity_ids = [c["entity_id"] for c in cands_for_mention]
            if gold_entity_id in cand_entity_ids:
                if gold_entity_id != cand_entity_ids[0]:
                    count_move += 1
                index = cand_entity_ids.index(gold_entity_id)
                gold_canonical_name = cands_for_mention[index]["canonical_name"]
                result_cands_for_mention = cands_for_mention[:index] + cands_for_mention[index+1:]
                original_gold_entity_rank_list.append(index + random.random())
            else:
                count_add += 1
                gold_canonical_name = None # Not used
                result_cands_for_mention = cands_for_mention[:-1]
                original_gold_entity_rank_list.append(top_k+1000 + random.random()) # Should be larger than top_k
            count_mentions += 1

            # Append the gold entity to the top of the candidate list
            result_cands_for_mention = [{
                "entity_id": gold_entity_id,
                "canonical_name": gold_canonical_name,
                "score": 1000000.0
            }] + result_cands_for_mention
            result_cands_for_mentions.append(result_cands_for_mention)

        assert len(document["mentions"]) == len(result_cands_for_mentions) == len(original_gold_entity_rank_list)

        result_cands_for_doc = {
            "doc_key": document["doc_key"],
            "candidate_entities": result_cands_for_mentions,
            "original_gold_entity_rank_list": original_gold_entity_rank_list
        }
        result_candidate_entities.append(result_cands_for_doc)

    logging.info(f"Added (or changed the position of) gold entities to the list of top-{top_k} candidate entities for {count_add} ({count_move}) / {count_mentions} mentions")
    return result_candidate_entities

 

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
    parser.add_argument("--train_candidate_entities", type=str, required=True)
    parser.add_argument("--dev_candidate_entities", type=str, required=True)
    parser.add_argument("--test_candidate_entities", type=str, required=True)
    parser.add_argument("--train_demonstrations", type=str, default=None)
    parser.add_argument("--dev_demonstrations", type=str, default=None)
    parser.add_argument("--test_demonstrations", type=str, default=None)
    parser.add_argument("--entity_dict", type=str, required=True)

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
        

