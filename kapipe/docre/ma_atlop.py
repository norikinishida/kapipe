from __future__ import annotations

from collections import OrderedDict
import copy
import logging
import os
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers.modeling_outputs import ModelOutput
from opt_einsum import contract
from tqdm import tqdm
import jsonlines

from ..nn_utils import (
    AdaptiveThresholdingLoss,
    get_optimizer2,
    get_scheduler2
)
from .. import evaluation
from .. import utils
from ..utils import BestScoreHolder
from ..datatypes import Config, Document, Triple


logger = logging.getLogger(__name__)


class MAATLOP:
    """
    Mention-Agnostic ATLOP (Oumaima and Nishida et al., 2024)
    """

    def __init__(
        self,
        device: str,
        # Initialization
        config: Config | str | None = None,
        vocab_relation: dict[str, int] | str | None = None,
        path_entity_dict: str | None = None,
        # Loading
        path_snapshot: str | None = None
    ):
        logger.info("########## MAATLOP Initialization Starts ##########")

        self.device = device
        self.path_snapshot = path_snapshot

        if path_snapshot is not None:
            assert config is None
            assert vocab_relation is None
            assert path_entity_dict is None
            config = path_snapshot + "/config"
            vocab_relation = path_snapshot + "/relations.vocab.txt"
            path_entity_dict = path_snapshot + "/entity_dict.json"
            path_model = path_snapshot + "/model"

        # Load the configuration
        if isinstance(config, str):
            config_path = config
            config = utils.get_hocon_config(config_path=config_path)
            logger.info(f"Loaded configuration from {config_path}")
        self.config = config
        logger.info(utils.pretty_format_dict(self.config))

        # Load the relation vocabulary
        if isinstance(vocab_relation, str):
            vocab_path = vocab_relation
            vocab_relation = utils.read_vocab(vocab_path)
            logger.info(f"Loaded relation type vocabulary from {vocab_path}")
        self.vocab_relation = vocab_relation
        self.ivocab_relation = {i:l for l, i in self.vocab_relation.items()}

        # Load the entity dictionary
        logger.info(f"Loading entity dictionary from {path_entity_dict}")
        self.entity_dict = {
            epage["entity_id"]: epage
            for epage in utils.read_json(path_entity_dict)
        }
        logger.info(f"Completed loading of entity dictionary with {len(self.entity_dict)} entities from {path_entity_dict}")
        self.kb_entity_ids = list(self.entity_dict.keys())

        # Initialize the model
        self.model_name = config["model_name"]
        self.top_k_labels = config["top_k_labels"]
        if self.model_name == "ma_atlop_model":
            self.model = MAATLOPModel(
                device=device,
                bert_pretrained_name_or_path=config["bert_pretrained_name_or_path"],
                max_seg_len=config["max_seg_len"],
                entity_dict=self.entity_dict,
                entity_seq_length=config["entity_seq_length"],
                use_localized_context_pooling=config["use_localized_context_pooling"],
                bilinear_block_size=config["bilinear_block_size"],
                use_entity_loss=self.config["do_negative_entity_sampling"],
                vocab_relation=self.vocab_relation,
                possible_head_entity_types=config["possible_head_entity_types"],
                possible_tail_entity_types=config["possible_tail_entity_types"],
                use_mention_as_canonical_name=config["use_mention_as_canonical_name"]
            )
        else:
            raise Exception(f"Invalid model_name: {self.model_name}")

        # Show parameter shapes
        # logger.info("Model parameters:")
        # for name, param in self.model.named_parameters():
        #     logger.info(f"{name}: {tuple(param.shape)}")

        # Load trained model parameters
        if path_snapshot is not None:
            self.model.load_state_dict(
                torch.load(path_model, map_location=torch.device("cpu")),
                strict=False
            )
            logger.info(f"Loaded model parameters from {path_model}")

        self.model.to(self.model.device)

        logger.info("########## MAATLOP Initialization Ends ##########")

    # def load_model(self, path_model: str) -> None:
    #     if ignored_names is None:
    #         self.model.load_state_dict(
    #             torch.load(path, map_location=torch.device("cpu")),
    #             strict=False
    #         )
    #     else:
    #         checkpoint = torch.load(path, map_location=torch.device("cpu"))
    #         for name in ignored_names:
    #             logger.info(f"Ignored {name} component in loading")
    #             checkpoint = {
    #                 k:v for k,v in checkpoint.items() if not name in k
    #             }
    #         self.model.load_state_dict(checkpoint, strict=False)

    def save(self, path_snapshot: str, model_only: bool = False) -> None:
        path_config = path_snapshot + "/config"
        path_vocab = path_snapshot + "/relations.vocab.txt"
        path_entity_dict = path_snapshot + "/entity_dict.json"
        path_model = path_snapshot + "/model"
        if not model_only:
            utils.write_json(path_config, self.config)
            utils.write_vocab(path_vocab, self.vocab_relation, write_frequency=False)
            utils.write_json(path_entity_dict, self.entity_dict)
        torch.save(self.model.state_dict(), path_model)

    def compute_loss(self, document: Document) -> (
        tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor, int]
        | tuple[torch.Tensor, torch.Tensor, int, int]
    ):
        # Switch to training mode
        self.model.train()

        # Negative Entity Sampling
        if self.config["do_negative_entity_sampling"]:
            document = self.sample_negative_entities_randomly(
                document=document,
                sample_size=round(
                    len(document["entities"]) * self.config["negative_entity_ratio"]
                )
            )

        # Preprocess
        preprocessed_data = self.model.preprocess(document=document)

        # Tensorize
        model_input = self.model.tensorize(
            preprocessed_data=preprocessed_data,
            compute_loss=True
        )

        # Forward
        model_output = self.model.forward(**model_input)

        if self.config["do_negative_entity_sampling"]:
            return (
                model_output.pair_loss,
                model_output.pair_acc,
                model_output.n_valid_pairs,
                model_output.n_valid_triples,
                #
                model_output.entity_loss,
                model_output.n_entities,
            )
        else:
            return (
                model_output.pair_loss,
                model_output.pair_acc,
                model_output.n_valid_pairs,
                model_output.n_valid_triples
            )

    def sample_negative_entities_randomly(
        self,
        document: Document,
        sample_size: int
    ) -> Document:
        result_document = copy.deepcopy(document)

        n_entities = len(result_document["entities"])

        gold_entity_ids = [e["entity_id"] for e in result_document["entities"]]

        # Sample candidate entity ids from the entire KB
        sampled_entity_ids = np.random.choice(
            self.kb_entity_ids,
            sample_size + len(gold_entity_ids),
            replace=False
        )

        # Remove gold entities from the sampled list
        sampled_entity_ids = [
            eid for eid in sampled_entity_ids if not eid in gold_entity_ids
        ]
        sampled_entity_ids = sampled_entity_ids[:sample_size]

        # Retrieve the names and types for the sampled entity ids from the KB
        sampled_entity_names = []
        sampled_entity_types = []
        for eid in sampled_entity_ids:
            epage = self.entity_dict[eid]
            name = epage["canonical_name"]
            etype = epage["entity_type"]
            sampled_entity_names.append(name)
            sampled_entity_types.append(etype)

        # Integrate the sampled entities to the document
        sampled_entity_mention_index = []
        for name, etype, eid in zip(
            sampled_entity_names,
            sampled_entity_types,
            sampled_entity_ids
        ):
            mention = {
                "span": None,
                "name": name,
                "entity_type": etype,
                "entity_id": eid,
            }
            result_document["mentions"].append(mention)
            sampled_entity_mention_index.append(len(result_document["mentions"]) - 1)

        for e_i in range(len(result_document["entities"])):
            result_document["entities"][e_i]["is_dummy"] = True

        for m_i, etype, eid in zip(
            sampled_entity_mention_index,
            sampled_entity_types,
            sampled_entity_ids
        ):
            entity = {
                "mention_indices": [m_i],
                "entity_type": etype,
                "entity_id": eid,
                "is_dummy": False,
            }
            result_document["entities"].append(entity)

        assert len(result_document["entities"]) == n_entities + sample_size
        return result_document

    def extract(self, document: Document) -> Document:
        with torch.no_grad():
            # Switch to inference mode
            self.model.eval()

            # Preprocess
            preprocessed_data = self.model.preprocess(document=document)

            # Return no triple if head or tail entity is missing
            if (
                len(preprocessed_data["pair_head_entity_indices"]) == 0
                or
                len(preprocessed_data["pair_tail_entity_indices"]) == 0
            ):
                result_document = copy.deepcopy(document)
                result_document["relations"] = []
                return result_document

            # Tensorize
            model_input = self.model.tensorize(
                preprocessed_data=preprocessed_data,
                compute_loss=False
            )

            # Forward
            model_output = self.model.forward(**model_input)
            logits = model_output.pair_logits # (n_entity_pairs, n_relations)

            # Structurize
            triples = self.structurize(
                pair_head_entity_indices=preprocessed_data["pair_head_entity_indices"],
                pair_tail_entity_indices=preprocessed_data["pair_tail_entity_indices"],
                logits=logits
            )

            # Integrate
            result_document = copy.deepcopy(document)
            result_document["relations"] = triples
            return result_document

    def structurize(
        self,
        pair_head_entity_indices: np.ndarray,
        pair_tail_entity_indices: np.ndarray,
        logits: torch.Tensor
    ) -> list[Triple]:
        triples: list[Triple] = []

        # Get predicted relation labels (indices)
        # (n_entity_pairs, n_relations)
        pair_pred_relation_labels = self.model.pair_loss_function.get_labels(
            logits=logits,
            top_k=self.top_k_labels
        ).cpu().numpy()

        for head_entity_i, tail_entity_i, rel_indicators in zip(
            pair_head_entity_indices,
            pair_tail_entity_indices,
            pair_pred_relation_labels
        ):
            if head_entity_i == tail_entity_i:
                continue
            # Find positive (i.e., non-zero) relation labels (indices)
            rel_indices = np.nonzero(rel_indicators)[0].tolist()
            for rel_i in rel_indices:
                if rel_i != 0:
                    # Convert relation index to relation name
                    rel = self.ivocab_relation[rel_i]
                    # Add a new triple
                    triples.append({
                        "arg1": int(head_entity_i),
                        "relation": rel,
                        "arg2": int(tail_entity_i),
                        })

        return triples

    def batch_extract(self, documents: list[Document]) -> list[Document]:
        result_documents = []
        for document in tqdm(documents, desc="extraction steps"):
            result_document = self.extract(document=document)
            result_documents.append(result_document)
        return result_documents


class MAATLOPTrainer:
    """
    Trainer class for MA-ATLOP extractor.
    Handles training loop, evaluation, model saving, and early stopping.
    """

    def __init__(self, base_output_path: str):
        self.base_output_path = base_output_path
        self.paths = self.get_paths()

    def get_paths(self) -> dict[str, str]:
        paths = {}

        # configurations
        paths["path_snapshot"] = self.base_output_path

        # training outputs
        paths["path_train_losses"] = self.base_output_path + "/train.losses.jsonl"
        paths["path_dev_evals"] = self.base_output_path + "/dev.eval.jsonl"

        # evaluation outputs
        paths["path_dev_gold"] = self.base_output_path + "/dev.gold.json"
        paths["path_dev_pred"] = self.base_output_path + "/dev.pred.json"
        paths["path_dev_eval"] = self.base_output_path + "/dev.eval.json"
        paths["path_test_gold"] = self.base_output_path + "/test.gold.json"
        paths["path_test_pred"] = self.base_output_path + "/test.pred.json"
        paths["path_test_eval"] = self.base_output_path + "/test.eval.json"

        # required for Ign evaluation
        paths["path_gold_train_triples"] = self.base_output_path + "/gold_train_triples.json"

        return paths

    def setup_dataset(
        self,
        extractor: MAATLOP,
        documents: list[Document],
        split: str,
        with_gold_annotations: bool = True
    ) -> None:
        if split == "train":
            # Cache the gold training triples for Ign evaluation
            if not os.path.exists(self.paths["path_gold_train_triples"]):
                gold_train_triples = []
                for document in tqdm(documents, desc="dataset setup"):
                    mentions = document["mentions"]
                    entity_index_to_mention_names = {
                        e_i: [
                            mentions[m_i]["name"]
                            for m_i in e["mention_indices"]
                        ]
                        for e_i, e in enumerate(document["entities"])
                    }
                    for triple in document["relations"]:
                        arg1_entity_i = triple["arg1"]
                        rel = triple["relation"]
                        arg2_entity_i = triple["arg2"]
                        arg1_mention_names = entity_index_to_mention_names[
                            arg1_entity_i
                        ]
                        arg2_mention_names = entity_index_to_mention_names[
                            arg2_entity_i
                        ]
                        for arg1_mention_name in arg1_mention_names:
                            for arg2_mention_name in arg2_mention_names:
                                gold_train_triples.append((
                                    arg1_mention_name,
                                    rel,
                                    arg2_mention_name
                                ))
                gold_train_triples = list(set(gold_train_triples))
                gold_train_triples = {"root": gold_train_triples}
                utils.write_json(
                    self.paths["path_gold_train_triples"],
                    gold_train_triples
                )
                logger.info(f"Saved the gold training triples for Ign evaluation in {self.paths['path_gold_train_triples']}")

        # Cache the gold annotations for evaluation
        if split != "train" and with_gold_annotations:
            path_gold = self.paths[f"path_{split}_gold"]
            if not os.path.exists(path_gold):
                gold_documents = []
                for document in tqdm(documents, desc="dataset setup"):
                    gold_doc = copy.deepcopy(document)
                    gold_documents.append(gold_doc)
                utils.write_json(path_gold, gold_documents)
                logger.info(f"Saved the gold annotations for evaluation in {path_gold}")

    def train(
        self,
        extractor: MAATLOP,
        train_documents: list[Document],
        dev_documents: list[Document],
        supplemental_info: dict[str, Any]
    ) -> None:
        ##################
        # Setup
        ##################

        train_doc_indices = np.arange(len(train_documents))

        n_train = len(train_doc_indices)
        max_epoch = extractor.config["max_epoch"]
        batch_size = extractor.config["batch_size"]
        gradient_accumulation_steps = extractor.config["gradient_accumulation_steps"]
        total_update_steps = n_train * max_epoch // (batch_size * gradient_accumulation_steps)
        warmup_steps = int(total_update_steps * extractor.config["warmup_ratio"])

        logger.info("Number of training documents: %d" % n_train)
        logger.info("Number of epochs: %d" % max_epoch)
        logger.info("Batch size: %d" % batch_size)
        logger.info("Gradient accumulation steps: %d" % gradient_accumulation_steps)
        logger.info("Total update steps: %d" % total_update_steps)
        logger.info("Warmup steps: %d" % warmup_steps)

        optimizer = get_optimizer2(
            model=extractor.model,
            config=extractor.config
        )
        scheduler = get_scheduler2(
            optimizer=optimizer,
            total_update_steps=total_update_steps,
            warmup_steps=warmup_steps
        )

        writer_train = jsonlines.Writer(
            open(self.paths["path_train_losses"], "w"),
            flush=True
        )
        writer_dev = jsonlines.Writer(
            open(self.paths["path_dev_evals"], "w"),
            flush=True
        )

        bestscore_holder = BestScoreHolder(scale=1.0)
        bestscore_holder.init()

        ##################
        # Initial Validation
        ##################

        # Evaluate the extractor
        if extractor.config["use_official_evaluation"]:
            scores = self.official_evaluate(
                extractor=extractor,
                documents=dev_documents,
                split="dev",
                supplemental_info=supplemental_info,
                #
                get_scores_only=True
            )
        else:
            scores = self.evaluate(
                extractor=extractor,
                documents=dev_documents,
                split="dev",
                supplemental_info=supplemental_info,
                #
                skip_intra_inter=True,
                skip_ign=True,
                get_scores_only=True
            )
        scores.update({"epoch": 0, "step": 0})
        writer_dev.write(scores)
        logger.info(utils.pretty_format_dict(scores))

        # Set the best validation score
        bestscore_holder.compare_scores(scores["standard"]["f1"], 0)

        # Save
        extractor.save(path_snapshot=self.paths["path_snapshot"])
        logger.info(f"Saved config, relation vocabulary, entity dictionary, and model to {self.paths['path_snapshot']}")

        ##################
        # Training Loop
        ##################

        bert_param, task_param = extractor.model.get_params()
        extractor.model.zero_grad()
        step = 0
        batch_i = 0

        # Variables for reporting
        loss_accum = 0.0
        acc_accum = 0.0
        accum_count = 0

        progress_bar = tqdm(total=total_update_steps, desc="training steps")

        for epoch in range(1, max_epoch + 1):

            perm = np.random.permutation(n_train)

            for instance_i in range(0, n_train, batch_size):

                ##################
                # Forward
                ##################

                batch_i += 1

                # Initialize loss
                batch_loss = 0.0
                batch_acc = 0.0
                actual_batchsize = 0
                actual_total_pairs = 0
                actual_total_triples = 0

                batch_entity_loss = 0.0
                actual_total_entities = 0

                for doc_i in train_doc_indices[
                    perm[instance_i : instance_i + batch_size]
                ]:

                    # Forward and compute loss
                    extractor_output = extractor.compute_loss(
                        document=train_documents[doc_i]
                    )
                    if extractor.config["do_negative_entity_sampling"]:
                        (
                            one_loss,
                            one_acc,
                            n_valid_pairs,
                            n_valid_triples,
                            one_entity_loss,
                            n_entities
                        ) = extractor_output
                    else:
                        (
                            one_loss,
                            one_acc,
                            n_valid_pairs,
                            n_valid_triples,
                        ) = extractor_output

                    # Accumulate the loss
                    batch_loss = batch_loss + one_loss
                    batch_acc += one_acc
                    actual_batchsize += 1
                    actual_total_pairs += n_valid_pairs
                    actual_total_triples += n_valid_triples
                    if extractor.config["do_negative_entity_sampling"]:
                        batch_entity_loss = batch_entity_loss + one_entity_loss
                        actual_total_entities += n_entities

                # Average the loss
                actual_batchsize = float(actual_batchsize)
                actual_total_pairs = float(actual_total_pairs)
                actual_total_triples = float(actual_total_triples)
                batch_loss = batch_loss / actual_total_pairs # loss per pair
                batch_acc = batch_acc / actual_total_triples
                if extractor.config["do_negative_entity_sampling"]:
                    actual_total_entities = float(actual_total_entities)
                    batch_entity_loss = batch_entity_loss / actual_total_entities # loss per entity

                ##################
                # Backward
                ##################

                batch_loss = batch_loss + batch_entity_loss

                batch_loss = batch_loss / gradient_accumulation_steps
                batch_loss.backward()

                # Accumulate for reporting
                loss_accum += float(batch_loss.cpu())
                acc_accum += batch_acc
                accum_count += 1

                if batch_i % gradient_accumulation_steps == 0:

                    ##################
                    # Update
                    ##################

                    if extractor.config["max_grad_norm"] > 0:
                        torch.nn.utils.clip_grad_norm_(
                            bert_param,
                            extractor.config["max_grad_norm"]
                        )
                        torch.nn.utils.clip_grad_norm_(
                            task_param,
                            extractor.config["max_grad_norm"]
                        )

                    optimizer.step()
                    scheduler.step()

                    extractor.model.zero_grad()

                    step += 1
                    progress_bar.update()
                    progress_bar.refresh()

                if (
                    (instance_i + batch_size >= n_train)
                    or
                    (
                        (batch_i % gradient_accumulation_steps == 0)
                        and
                        (step % extractor.config["n_steps_for_monitoring"] == 0)
                    )
                ):

                    ##################
                    # Rerpot
                    ##################

                    report = {
                        "step": step,
                        "epoch": epoch,
                        "step_progress": "%d/%d" % (step, total_update_steps),
                        "step_progress(ratio)": float(step) / total_update_steps * 100.0,
                        "one_epoch_progress": "%d/%d" % (instance_i + actual_batchsize, n_train),
                        "one_epoch_progress(ratio)": float(instance_i + actual_batchsize) / n_train * 100.0,
                        "loss": loss_accum / accum_count,
                        "accuracy": 100.0 * acc_accum / accum_count,
                        "max_valid_f1": bestscore_holder.best_score,
                        "patience": bestscore_holder.patience
                    }
                    writer_train.write(report)
                    logger.info(utils.pretty_format_dict(report))
                    loss_accum = 0.0
                    acc_accum = 0.0
                    accum_count = 0

                if (
                    (instance_i + batch_size >= n_train)
                    or
                    (
                        (batch_i % gradient_accumulation_steps == 0)
                        and
                        (extractor.config["n_steps_for_validation"] > 0)
                        and
                        (step % extractor.config["n_steps_for_validation"] == 0)
                    )
                ):

                    ##################
                    # Validation
                    ##################

                    # Evaluate the extractor
                    if extractor.config["use_official_evaluation"]:
                        scores = self.official_evaluate(
                            extractor=extractor,
                            documents=dev_documents,
                            split="dev",
                            supplemental_info=supplemental_info,
                            #
                            get_scores_only=True
                        )
                    else:
                        scores = self.evaluate(
                            extractor=extractor,
                            documents=dev_documents,
                            split="dev",
                            supplemental_info=supplemental_info,
                            #
                            skip_intra_inter=True,
                            skip_ign=True,
                            get_scores_only=True
                        )
                    scores.update({"epoch": epoch, "step": step})
                    writer_dev.write(scores)
                    logger.info(utils.pretty_format_dict(scores))

                    # Update the best validation score
                    did_update = bestscore_holder.compare_scores(
                        scores["standard"]["f1"],
                        epoch
                    )
                    logger.info("[Step %d] Max validation F1: %f" % (step, bestscore_holder.best_score))

                    # Save the model
                    if did_update:
                        extractor.save(
                            path_snapshot=self.paths["path_snapshot"],
                            model_only=True
                        )
                        logger.info(f"Saved model to {self.paths['path_snapshot']}")

                    ##################
                    # Termination Check
                    ##################

                    if (
                        bestscore_holder.patience
                        >= extractor.config["max_patience"]
                    ):
                        writer_train.close()
                        writer_dev.close()
                        progress_bar.close()
                        return

        writer_train.close()
        writer_dev.close()
        progress_bar.close()

    def evaluate(
        self,
        extractor: MAATLOP,
        documents: list[Document],
        split: str,
        supplemental_info: dict[str, Any],
        #
        prediction_only: bool = False,
        skip_intra_inter: bool = False,
        skip_ign: bool = False,
        get_scores_only: bool = False
    ) -> dict[str, Any] | None:
        # Apply the extractor
        result_documents = extractor.batch_extract(documents=documents)
        utils.write_json(self.paths[f"path_{split}_pred"], result_documents)
    
        if prediction_only:
            return

        # Calculate the evaluation scores
        # path_gold_documents = supplemental_info["path_gold_documents"][split]
        scores = evaluation.docre.fscore(
            pred_path=self.paths[f"path_{split}_pred"],
            gold_path=self.paths[f"path_{split}_gold"],
            skip_intra_inter=skip_intra_inter,
            skip_ign=skip_ign,
            # gold_documents_path=path_gold_documents,
            gold_train_triples_path=self.paths["path_gold_train_triples"]
        )

        if get_scores_only:
            return scores

        # Save the evalution scores
        utils.write_json(self.paths[f"path_{split}_eval"], scores)
        logger.info(utils.pretty_format_dict(scores))
        return scores

    def official_evaluate(
        self,
        extractor: MAATLOP,
        documents: list[Document],
        split: str,
        supplemental_info: dict[str, Any],
        #
        prediction_only: bool = False,
        get_scores_only: bool = False
    ) -> dict[str, Any] | None:
        # Apply the extractor
        result_documents = extractor.batch_extract(documents=documents)
        utils.write_json(self.paths[f"path_{split}_pred"], result_documents)
        triples = evaluation.docre.to_official(
            path_input=self.paths[f"path_{split}_pred"],
            path_output=self.paths[
                f"path_{split}_pred"
            ].replace(".json", ".official.json")
        )

        if prediction_only:
            return

        # Calculate the evaluation scores
        original_data_dir = supplemental_info["original_data_dir"]
        train_file_name = supplemental_info["train_file_name"]
        dev_file_name = supplemental_info[f"{split}_file_name"]
        scores = evaluation.docre.official_evaluate(
            triples=triples,
            original_data_dir=original_data_dir,
            train_file_name=train_file_name,
            dev_file_name=dev_file_name
        )

        if get_scores_only:
            return scores

        # Save the evaluation scores
        utils.write_json(self.paths[f"path_{split}_eval"], scores)
        logger.info(utils.pretty_format_dict(scores))
        return scores


class MentionTuple(NamedTuple):
    span: tuple[int, int] | None
    name: str
    entity_type: str
    entity_id: str


class EntityTuple(NamedTuple):
    mention_indices: list[int]
    entity_type: str
    entity_id: str
    is_dummy: bool


class TripleTuple(NamedTuple):
    arg1: int
    relation: str
    arg2: int


class MAATLOPModel(nn.Module):

    def __init__(
        self,
        device,
        bert_pretrained_name_or_path,
        max_seg_len,
        entity_dict,
        entity_seq_length,
        use_localized_context_pooling,
        bilinear_block_size,
        use_entity_loss,
        vocab_relation,
        possible_head_entity_types=None,
        possible_tail_entity_types=None,
        use_mention_as_canonical_name=False
    ):
        """
        Parameters
        ----------
        device : str
        bert_pretrained_name_or_path : str
        max_seg_len : int
        entity_dict : dict[str, EntityPage]
        entity_seq_length : int
        use_localized_context_pooling : bool
        bilinear_block_size : int
        use_entity_loss : bool
        vocab_relation : dict[str, int]
        possible_head_entity_types: list[str] | None
            by default None
        possible_tail_entity_types: list[str] | None
            by default None
        use_mention_as_canonical_name : bool
            by default False
        """
        super().__init__()

        ########################
        # Hyper parameters
        ########################

        self.device = device
        self.bert_pretrained_name_or_path = bert_pretrained_name_or_path
        self.max_seg_len = max_seg_len
        self.entity_dict = entity_dict
        self.entity_seq_length = entity_seq_length
        self.use_localized_context_pooling = use_localized_context_pooling
        self.bilinear_block_size = bilinear_block_size
        self.use_entity_loss = use_entity_loss
        self.vocab_relation = vocab_relation
        self.possible_head_entity_types = possible_head_entity_types
        self.possible_tail_entity_types = possible_tail_entity_types
        self.use_mention_as_canonical_name = use_mention_as_canonical_name

        self.n_relations = len(self.vocab_relation)

        ########################
        # Components
        ########################

        # BERT, tokenizer
        self.bert, self.tokenizer = self._initialize_bert_and_tokenizer(
            pretrained_model_name_or_path=self.bert_pretrained_name_or_path
        )

        # Dimensionality
        self.hidden_dim = self.bert.config.hidden_size

        # Entity-level classification
        if self.use_entity_loss:
            self.entity_linear = nn.Linear(self.hidden_dim, 2)

        # DocRE classification
        if self.use_localized_context_pooling:
            self.linear_head = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
            self.linear_tail = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        else:
            self.linear_head = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.linear_tail = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.block_bilinear = nn.Linear(
            self.hidden_dim * self.bilinear_block_size,
            self.n_relations
        )

        ########################
        # Preprocessor
        ########################

        self.preprocessor = MAATLOPPreprocessor(
            tokenizer=self.tokenizer,
            max_seg_len=self.max_seg_len,
            entity_dict=self.entity_dict,
            entity_seq_length=self.entity_seq_length,
            vocab_relation=self.vocab_relation,
            possible_head_entity_types=self.possible_head_entity_types,
            possible_tail_entity_types=self.possible_tail_entity_types,
            use_mention_as_canonical_name=self.use_mention_as_canonical_name
        )

        ########################
        # Loss function
        ########################

        self.entity_loss_function = nn.CrossEntropyLoss(reduction="none")
        self.pair_loss_function = AdaptiveThresholdingLoss()

    def _initialize_bert_and_tokenizer(self, pretrained_model_name_or_path):
        """
        Parameters
        ----------
        pretrained_model_name_or_path : str

        Returns
        -------
        tuple[AutoModel, AutoTokenizer]
        """
        bert = AutoModel.from_pretrained(
            pretrained_model_name_or_path,
            return_dict=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            additional_special_tokens=None
        )
        return bert, tokenizer

    ########################
    # For optimization
    ########################

    def get_params(self, named=False):
        """
        Parameters
        ----------
        named : bool
            by default False

        Returns
        -------
        tuple[list[tuple[str, Any]], list[tuple[str, Any]]]
        """
        bert_based_param, task_param = [], []
        for name, param in self.named_parameters():
            if name.startswith('bert'):
                to_add = (name, param) if named else param
                bert_based_param.append(to_add)
            else:
                to_add = (name, param) if named else param
                task_param.append(to_add)
        return bert_based_param, task_param

    ################
    # Forward pass
    ################

    def preprocess(
        self,
        document
    ):
        """
        Parameters
        ----------
        document : Document

        Returns
        -------
        dict[str, Any]
        """
        return self.preprocessor.preprocess(document=document)
 
    def tensorize(self, preprocessed_data, compute_loss):
        """
        Parameters
        ----------
        preprocessed_data : dict[str, Any]
        compute_loss : bool

        Returns
        -------
        dict[str, Any]
        """
        model_input = {}

        model_input["compute_loss"] = compute_loss

        # (n_entities, max_seg_len)
        model_input["segments_id"] = torch.tensor(
            preprocessed_data["bert_input"]["segments_id"],
            device=self.device
        )

        # (n_entities, max_seg_len)
        model_input["segments_mask"] = torch.tensor(
            preprocessed_data["bert_input"]["segments_mask"],
            device=self.device
        )

        # (n_entities, max_seg_len)
        model_input["segments_token_type_id"] = torch.tensor(
            preprocessed_data["bert_input"]["segments_token_type_id"],
            device=self.device
        )

        # (n_entity_pairs,)
        model_input["pair_head_entity_indices"] = torch.tensor(
            preprocessed_data["pair_head_entity_indices"],
            device=self.device
        )

        # (n_entity_pairs,)
        model_input["pair_tail_entity_indices"] = torch.tensor(
            preprocessed_data["pair_tail_entity_indices"],
            device=self.device
        )

        if not compute_loss:
            return model_input

        if self.use_entity_loss:
            # (n_entities,)
            model_input["entity_gold_labels"] = torch.tensor(
                preprocessed_data["entity_gold_labels"],
                device=self.device
            ).to(torch.long)

        # (n_entity_pairs, n_relations)
        model_input["pair_gold_relation_labels"] = torch.tensor(
            preprocessed_data["pair_gold_relation_labels"],
            device=self.device
        ).to(torch.float)

        return model_input

    def forward(
        self,
        segments_id,
        segments_mask,
        segments_token_type_id,
        pair_head_entity_indices,
        pair_tail_entity_indices,
        compute_loss,
        entity_gold_labels=None,
        pair_gold_relation_labels=None,
    ):
        """
        Parameters
        ----------
        segments_id : torch.Tensor
            shape of (n_entities, max_seg_len)
        segments_mask : torch.Tensor
            shape of (n_entities, max_seg_len)
        segments_token_type_id : torch.Tensor
            shape of (n_entities, max_seg_len)
        pair_head_entity_indices : torch.Tensor
            shape of (n_entity_pairs,)
        pair_tail_entity_indices : torch.Tensor
            shape of (n_entity_pairs,)
        compute_loss : bool
        entity_gold_labels : torch.Tensor | None
            shape of (n_entities,); by default None
        pair_gold_relation_labels : torch.Tensor | None
            shape of (n_entity_pairs, n_relations); by default None

        Returns
        -------
        ModelOutput
        """
        # Encode tokens by BERT
        # (n_entities, max_seg_len, hidden_dim)
        # (n_entities, n_heads, max_seg_len, max_seg_len)
        segments_vec, segments_att = self.encode_tokens(
            segments_id=segments_id,
            segments_mask=segments_mask,
            segments_token_type_id=segments_token_type_id
        )

        # Get [CLS] vectors
        # (n_entities, hidden_dim)
        entity_vectors = segments_vec[:, 0, :]

        if self.use_entity_loss:
            # Compute entity-level logits by a linear layer
            # (n_entities, 2)
            entity_logits = self.entity_linear(entity_vectors)

        # Expand the entity vectors to entity pair
        # (n_entity_pairs, hidden_dim), (n_entity_pairs, hidden_dim)
        (
            pair_head_entity_vectors,
            pair_tail_entity_vectors
        ) = self.expand_entity_vectors(
            entity_vectors=entity_vectors,
            pair_head_entity_indices=pair_head_entity_indices,
            pair_tail_entity_indices=pair_tail_entity_indices
        )

        # Compute entity-pair context vectors (Localized Context Pooling)
        if self.use_localized_context_pooling:
            # (n_entity_pairs, hidden_dim)
            pair_context_vectors = self.compute_entity_pair_context_vectors(
                segments_vec=segments_vec,
                segments_att=segments_att,
                segments_mask=segments_mask,
                segments_token_type_id=segments_token_type_id,
                pair_head_entity_indices=pair_head_entity_indices,
                pair_tail_entity_indices=pair_tail_entity_indices
            )
        else:
            # (n_entity_pairs, hidden_dim)
            pair_context_vectors = None

        # Compute pair-level logits by block bilinear
        # (n_entity_pairs, n_relations)
        pair_logits = self.compute_logits_by_block_bilinear(
            pair_head_entity_vectors=pair_head_entity_vectors,
            pair_tail_entity_vectors=pair_tail_entity_vectors,
            pair_context_vectors=pair_context_vectors
        )

        if not compute_loss:
            if self.use_entity_loss:
                return ModelOutput(
                    entity_logits=entity_logits,
                    pair_logits=pair_logits
                )
            else:
                return ModelOutput(
                    pair_logits=pair_logits
                )

        if self.use_entity_loss:
            # Compute entity-level loss (summed over entities)
            # (n_entities,)
            entity_loss = self.entity_loss_function(
                entity_logits,
                entity_gold_labels
            )
            entity_loss = entity_loss.sum() # Scalar

            n_entities = len(entity_gold_labels)

        # Compute pair-level loss (summed over valid pairs)
        # (n_entity_pairs,)
        pair_loss = self.pair_loss_function(
            pair_logits,
            pair_gold_relation_labels
        )
        pair_loss = pair_loss.sum() # Scalar

        # Compute pair-level accuracy (summed over valid triples)
        # (n_entity_pairs, n_relations)
        pair_pred_relation_labels \
            = self.pair_loss_function.get_labels(logits=pair_logits)
        # (n_entity_pairs, n_relations)
        pair_acc = (
            pair_pred_relation_labels == pair_gold_relation_labels
        ).to(torch.float)
        pair_acc = pair_acc.sum().item() # float

        n_valid_pairs, n_relations = pair_gold_relation_labels.shape
        n_valid_triples = n_valid_pairs * n_relations

        if self.use_entity_loss:
            return ModelOutput(
                pair_logits=pair_logits,
                pair_loss=pair_loss,
                pair_acc=pair_acc,
                n_valid_pairs=n_valid_pairs,
                n_valid_triples=n_valid_triples,
                #
                entity_logits=entity_logits,
                entity_loss=entity_loss,
                n_entities=n_entities,
            )
        else:
            return ModelOutput(
                pair_logits=pair_logits,
                pair_loss=pair_loss,
                pair_acc=pair_acc,
                n_valid_pairs=n_valid_pairs,
                n_valid_triples=n_valid_triples,
            )

    ################
    # Subfunctions
    ################

    def encode_tokens(self, segments_id, segments_mask, segments_token_type_id):
        """
        Parameters
        ----------
        segments_id : torch.Tensor
            shape of (n_entities, max_seg_len)
        segments_mask : torch.Tensor
            shape of (n_entities, max_seg_len)
        segments_token_type_id : torch.Tensor
            shape of (n_entities, max_seg_len)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            shape of (n_entities, max_seg_len, hidden_dim)
            shape of (n_entities, n_heads, max_seg_len, hidden_dim)
        """
        bert_output = self.bert(
            input_ids=segments_id,
            attention_mask=segments_mask,
            token_type_ids=segments_token_type_id,
            output_attentions=True,
            output_hidden_states=False
        )
        # (n_entities, max_seg_len, hidden_dim)
        segments_vec = bert_output["last_hidden_state"]
        # (n_bert_layers, n_entities, n_heads, max_seg_len, max_seg_len)
        segments_att = bert_output["attentions"]
        # (n_entities, n_heads, max_seg_len, max_seg_len)
        segments_att = segments_att[-1]
        return segments_vec, segments_att

    def expand_entity_vectors(
        self,
        entity_vectors,
        pair_head_entity_indices,
        pair_tail_entity_indices
    ):
        """
        Parameters
        ----------
        entity_vectors : torch.Tensor
            shape of (n_entities, hidden_dim)
        pair_head_entity_indices : torch.Tensor
            shape of (n_entity_pairs,)
        pair_tail_entity_indices : torch.Tensor
            shape of (n_entity_pairs,)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            1. shape of (n_entity_pairs, hidden_dim)
            2. shape of (n_entity_pairs, hidden_dim)
        """
        # Expand the entity vectors
        # (n_entity_pairs, hidden_dim)
        pair_head_entity_vectors = entity_vectors[pair_head_entity_indices]
        # (n_entity_pairs, hidden_dim)
        pair_tail_entity_vectors = entity_vectors[pair_tail_entity_indices]
        return pair_head_entity_vectors, pair_tail_entity_vectors

    def compute_entity_pair_context_vectors(
        self,
        segments_vec,
        segments_att,
        segments_mask,
        segments_token_type_id,
        pair_head_entity_indices,
        pair_tail_entity_indices
    ):
        """
        Parameters
        ----------
        segments_vec : torch.Tensor
            shape of (n_entities, max_seg_len, hidden_dim)
        segments_att : torch.Tensor
            shape of (n_entities, n_heads, max_seg_len, max_seg_len)
        segments_mask : torch.Tensor
            shape of (n_entities, max_seg_len)
        segments_token_type_id : torch.Tensor
            shape of (n_entities, max_seg_len)
        pair_head_entity_indices : torch.Tensor
            shape of (n_entity_pairs,)
        pair_tail_entity_indices : torch.Tensor
            shape of (n_entity_pairs,)

        Returns
        -------
        torch.Tensor
            shape of (n_entity_pairs, hidden_dim)
        """
        n_entities = segments_vec.shape[0]

        # The number of text-side tokens
        # Remove "[SEP]" from the count
        # (n_entities,)
        text_seq_lengths = segments_token_type_id.sum(dim=1) - 1
        text_seq_mask = segments_token_type_id.to(torch.bool)

        # First, for each entity, obtain token embeddings in the text-side sequence
        entity_token_vectors = []
        for e_i in range(n_entities):
            # Remove the "[SEP]" embedding
            # (n_tokens_for_this_entity, hidden_dim)
            entity_tok_vecs = segments_vec[e_i, text_seq_mask[e_i]][:-1]
            # (max_seg_len, hidden_dim)
            entity_tok_vecs = F.pad(
                entity_tok_vecs,
                (0, 0, 0, self.max_seg_len - text_seq_lengths[e_i])
            )
            entity_token_vectors.append(entity_tok_vecs.unsqueeze(0))
        # (n_entities, max_seg_len, hidden_dim)
        entity_token_vectors = torch.cat(entity_token_vectors, dim=0)

        # Second, for each entity, obtain attentions to the text-side sequence
        # Pool the attentions over the heads, c.f., SAIS (Xiao et al., 2022)
        # (n_entities, max_seg_len, max_seg_len)
        segments_att = segments_att.sum(dim=1)
        # Extract attentions from the first token in the entity-side sequence
        #   to the text-side sequence.
        # index=1 is for the first entity-side token (index=0 is for [CLS])
        # (n_entities, max_seg_len)
        segments_att = segments_att[:, 1]
        entity_attentions = []
        for e_i in range(n_entities):
            # (n_tokens_for_this_entity,)
            entity_att = segments_att[e_i, text_seq_mask[e_i]][:-1]
            # (max_seg_len,)
            entity_att = F.pad(
                entity_att,
                (0, self.max_seg_len - text_seq_lengths[e_i])
            )
            entity_attentions.append(entity_att.unsqueeze(0))
        # (n_entities, max_seg_len)
        entity_attentions = torch.cat(entity_attentions, dim=0)
        # Normalize the entity-level attentions
        # entity_attentions = entity_attentions / (
        #     entity_attentions.sum(dim=1, keepdim=True) + 1e-10
        # )

        # Third, compute entity-pair-level attentions
        # (n_entity_pairs, max_seg_len)
        pair_head_attentions = entity_attentions[pair_head_entity_indices]
        # (n_entity_pairs, max_seg_len)
        pair_tail_attentions = entity_attentions[pair_tail_entity_indices]
        # (n_entity_pairs, max_seg_len)
        pair_attentions = pair_head_attentions * pair_tail_attentions
        # Normalize the pair-level attentions
        pair_attentions = pair_attentions / (
            pair_attentions.sum(dim=1, keepdim=True) + 1e-10
        )

        # Fourth, compute entity-pair-level token vecotrs
        # (n_entity_pairs, max_seg_len, hidden_dim)
        pair_head_token_vectors = entity_token_vectors[pair_head_entity_indices]
        # (n_entity_pairs, max_seg_len, hidden_dim)
        pair_tail_token_vectors = entity_token_vectors[pair_tail_entity_indices]
        # (n_entity_pairs, max_seg_len, hidden_dim)
        pair_token_vectors = pair_head_token_vectors + pair_tail_token_vectors

        # Lastly, compute entity-pair-level context vectors
        # (n_entity_pairs, hidden_dim)
        pair_context_vectors = contract(
            "pld,pl->pd",
            pair_token_vectors,
            pair_attentions
        )

        return pair_context_vectors

    def compute_logits_by_block_bilinear(
        self,
        pair_head_entity_vectors,
        pair_tail_entity_vectors,
        pair_context_vectors
    ):
        """
        Parameters
        ----------
        pair_head_entity_vectors : torch.Tensor
            shape of (n_entity_pairs, hidden_dim)
        pair_tail_entity_vectors : torch.Tensor
            shape of (n_entity_pairs, hidden_dim)
        pair_context_vectors : torch.Tensor | None
            shape of (n_entity_pairs, hidden_dim)

        Returns
        -------
        torch.Tensor
            shape of (n_entity_pairs, n_relations)
        """
        n_entity_pairs = len(pair_head_entity_vectors)

        if self.use_localized_context_pooling:
            zh = torch.cat(
                [pair_head_entity_vectors, pair_context_vectors],
                dim=1
            )
            zt = torch.cat(
                [pair_tail_entity_vectors, pair_context_vectors],
                dim=1
            )
        else:
            zh = pair_head_entity_vectors
            zt = pair_tail_entity_vectors

        zh = torch.tanh(self.linear_head(zh))
        zt = torch.tanh(self.linear_tail(zt))

        zh = zh.view(
            n_entity_pairs,
            self.hidden_dim // self.bilinear_block_size,
            self.bilinear_block_size
        )
        zt = zt.view(
            n_entity_pairs,
            self.hidden_dim // self.bilinear_block_size,
            self.bilinear_block_size
        )
        # (n_entity_pairs, hidden_dim * bilinear_block_size)
        input_to_block_bilinear = (
            zh.unsqueeze(3) * zt.unsqueeze(2)
        ).view(n_entity_pairs, self.hidden_dim * self.bilinear_block_size)

        # (n_entity_pairs, n_relations)
        logits = self.block_bilinear(input_to_block_bilinear)

        return logits


class MAATLOPPreprocessor:

    def __init__(
        self,
        tokenizer,
        max_seg_len,
        entity_dict,
        entity_seq_length,
        vocab_relation,
        possible_head_entity_types=None,
        possible_tail_entity_types=None,
        use_mention_as_canonical_name=False,
    ):
        """
        Parameters
        ----------
        tokenizer : PreTrainedTokenizer
        max_seg_len : int
        entity_dict : dict[str, EntityPage]
        entity_seq_length : int
        vocab_relation: dict[str, int]
        possible_head_entity_types: list[str] | None
            by default None
        possible_tail_entity_types: list[str] | None
            by default None
        use_mention_as_canonical_name : bool
            by default False
        """
        self.tokenizer = tokenizer
        self.max_seg_len = max_seg_len
        self.entity_dict = entity_dict
        self.entity_seq_length = entity_seq_length
        self.vocab_relation = vocab_relation
        self.possible_head_entity_types = possible_head_entity_types
        self.possible_tail_entity_types = possible_tail_entity_types
        self.use_mention_as_canonical_name = use_mention_as_canonical_name

        self.special_entity_sep_marker = ":"

    def preprocess(
        self,
        document
    ):
        """
        Parameters
        ----------
        document : Document

        Returns
        -------
        dict[str, Any]
        """
        preprocessed_data = OrderedDict()

        #####
        # doc_key: str
        # sentences: list[list[str]]
        # mentions: list[MentionTuple]
        # entities: list[EntityTuple]
        # relations: list[TripleTuple]
        #####

        preprocessed_data["doc_key"] = document["doc_key"]

        sentences = [s.split() for s in document["sentences"]]
        preprocessed_data["sentences"] = sentences

        mentions = [
            MentionTuple(
                None,
                m["name"],
                m["entity_type"],
                m["entity_id"]
            )
            for m in document["mentions"]
        ] # list[MentionTuple]
        preprocessed_data["mentions"] = mentions

        entities = [
            EntityTuple(
                e["mention_indices"],
                e["entity_type"],
                e["entity_id"],
                e["is_dummy"] if "is_dummy" in e else True
            )
            for e in document["entities"]
        ] # list[EntityTuple]
        preprocessed_data["entities"] = entities

        with_supervision = True if "relations" in document else False
        if with_supervision:
            relations = [
                TripleTuple(r["arg1"], r["relation"], r["arg2"])
                for r in document["relations"]
            ] # list[TripleTuple]
            preprocessed_data["relations"] = relations

        #####
        # mention_index_to_entity_index: list[int]
        #####

        # Mention index to entity index
        # NOTE: Although a single mention may belong to multiple entities,
        #   we assign only one entity index to each mention
        mention_index_to_entity_index = [None] * len(mentions) # list[int]
        for entity_i, entity in enumerate(entities):
            for mention_i in entity.mention_indices:
                mention_index_to_entity_index[mention_i] = entity_i
        preprocessed_data["mention_index_to_entity_index"] \
            = mention_index_to_entity_index

        #####
        # bert_input: dict[str, Any]
        # segments: list[list[str]]
        # segments_id: list[list[int]]
        # segments_mask: list[list[int]]
        # segments_token_type_id: list[list[int]]
        #####

        (
            segments,
            segments_id,
            segments_mask,
            segments_token_type_id
        ) = self.tokenize_and_split(
            sentences=sentences,
            entities=entities,
            mentions=mentions if self.use_mention_as_canonical_name else None
        )
        bert_input = {}
        bert_input["segments"] = segments
        bert_input["segments_id"] = segments_id
        bert_input["segments_mask"] = segments_mask
        bert_input["segments_token_type_id"] = segments_token_type_id
        preprocessed_data["bert_input"] = bert_input

        #####
        # pair_head_entity_indices: list[int]
        # pair_tail_entity_indices: list[int]
        # pair_gold_relation_labels: list[list[int]]
        #####

        not_include_entity_pairs = None
        if "not_include_pairs" in document:
            # list[tuple[EntityIndex, EntityIndex]]
            epairs = [
                (epair["arg1"], epair["arg2"])
                for epair in document["not_include_pairs"]
            ]
            not_include_entity_pairs \
                = [(e1,e2) for e1,e2 in epairs] + [(e2,e1) for e1,e2 in epairs]

        pair_head_entity_indices = [] # list[int]
        pair_tail_entity_indices = [] # list[int]
        if with_supervision:
            pair_gold_relation_labels = [] # list[list[int]]

        for head_entity_i in range(len(entities)):
            for tail_entity_i in range(len(entities)):
                # Skip diagonal
                if head_entity_i == tail_entity_i:
                    continue

                # Skip based on entity types if specified
                # e.g, Skip chemical-chemical, disease-disease,
                #      and disease-chemical pairs for CDR.
                if (
                    (self.possible_head_entity_types is not None)
                    and
                    (self.possible_tail_entity_types is not None)
                ):
                    head_entity_type = entities[head_entity_i].entity_type
                    tail_entity_type = entities[tail_entity_i].entity_type
                    if not (
                        (head_entity_type in self.possible_head_entity_types)
                        and
                        (tail_entity_type in self.possible_tail_entity_types)
                    ):
                        continue

                # Skip "not_include" pairs if specified
                if not_include_entity_pairs is not None:
                    if (head_entity_i, tail_entity_i) \
                            in not_include_entity_pairs:
                        continue

                pair_head_entity_indices.append(head_entity_i)
                pair_tail_entity_indices.append(tail_entity_i)

                if with_supervision:
                    rels = self.find_relations(
                        arg1=head_entity_i,
                        arg2=tail_entity_i,
                        relations=relations
                    )
                    multilabel_positive_indicators \
                        = [0] * len(self.vocab_relation)
                    if len(rels) == 0:
                        # Found no gold relation for this entity pair
                        multilabel_positive_indicators[0] = 1
                    else:
                        for rel in rels:
                            rel_id = self.vocab_relation[rel]
                            multilabel_positive_indicators[rel_id] = 1
                    pair_gold_relation_labels.append(
                        multilabel_positive_indicators
                    )

        pair_head_entity_indices = np.asarray(pair_head_entity_indices)
        pair_tail_entity_indices = np.asarray(pair_tail_entity_indices)
        preprocessed_data["pair_head_entity_indices"] = pair_head_entity_indices
        preprocessed_data["pair_tail_entity_indices"] = pair_tail_entity_indices
        if with_supervision:
            pair_gold_relation_labels = np.asarray(pair_gold_relation_labels)
            preprocessed_data["pair_gold_relation_labels"] \
                = pair_gold_relation_labels

        #####
        # entity_gold_labels: list[int]
        #####
        entity_gold_labels = [int(e.is_dummy) for e in entities]
        preprocessed_data["entity_gold_labels"] = entity_gold_labels

        return preprocessed_data

    #####
    # Subfunctions
    #####

    def tokenize_and_split(self, sentences, entities, mentions):
        """
        Parameters
        ----------
        sentences : list[list[str]]
        entities : list[EntityTuple]
        mentions : list[MentionTuple]

        Returns
        -------
        tuple[list[list[str]], list[list[int]], list[list[int]], list[list[int]]]
        """
        # Make entity-side sequence
        entity_seqs = [] # list[str]
        for ent in entities:
            entity_id = ent.entity_id
            epage = self.entity_dict[entity_id]
            if self.use_mention_as_canonical_name:
                m_i = ent.mention_indices[0]
                canonical_name = mentions[m_i].name
            else:
                canonical_name = epage["canonical_name"]
            # synonyms = epage["synonyms"]
            description = epage["description"]
            # "<canonical name> : <description>"
            entity_seq = " ".join([
                canonical_name,
                self.special_entity_sep_marker,
                description
            ])
            entity_seq = " ".join(entity_seq.split(" ")[:self.entity_seq_length])
            entity_seqs.append(entity_seq)
        # Make text-side sequence
        text_seq = " ".join(utils.flatten_lists(sentences)) # str
        # Combine and tokenize the sequences
        # [CLS] <canonical name> : <description> [SEP] <text> [SEP]
        inputs = self.tokenizer(
            entity_seqs,
            [text_seq] * len(entity_seqs),
            max_length=self.max_seg_len,
            padding=True,
            truncation="only_second",
            return_overflowing_tokens=False
        )
        segments_id = inputs["input_ids"] # list[list[int]]
        segments = [
            self.tokenizer.convert_ids_to_tokens(seg)
            for seg in inputs["input_ids"]
        ] # list[list[str]]
        segments_mask = inputs["attention_mask"] # list[list[int]]
        segments_token_type_id = inputs["token_type_ids"] # list[list[int]]
        return (
            segments,
            segments_id,
            segments_mask,
            segments_token_type_id
        )

    def find_relations(self, arg1, arg2, relations):
        """
        Parameters
        ----------
        arg1 : int
        arg2 : int
        relations : list[TripleTuple]

        Returns
        -------
        list[str]
        """
        rels = [] # list[str]
        for triple in relations:
            if triple.arg1 == arg1 and triple.arg2 == arg2:
                rels.append(triple.relation)
        return rels

