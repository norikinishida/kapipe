from __future__ import annotations

from collections import OrderedDict
import copy
import logging
import os
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers.modeling_outputs import ModelOutput
from tqdm import tqdm
import jsonlines

from ..datatypes import Config, Document, Triple
from .. import utils
from ..utils import BestScoreHolder
from .. import evaluation
from ..nn_utils import (
    make_mlp,
    FocalLoss,
    get_optimizer2,
    get_scheduler2
)


logger = logging.getLogger(__name__)


HEAD_ENTITY_SLOT = "@@@HEAD###"
TAIL_ENTITY_SLOT = "@@@TAIL###"
TRIPLE_TO_QUESTION_TEMPLATES = {
    "cdr": {
        "CID": "Does @@@HEAD### induce @@@TAIL### ?"
    },
    "hoip": {
        "has result": "Does @@@HEAD### result in @@@TAIL### ?",
        "has part": "Does @@@HEAD### involve @@@TAIL### ?",
        # "has molecular reaction": "Does @@@HEAD### have molecular reaction of @@@TAIL### ?",
        # "part of": "Is @@@HEAD### part of @@@TAIL### ?",
    }
}


class MAQA:
    """
    Mention-Agnostic QA-based DocRE Extractor (Oumaima and Nishida et al., 2024)
    """

    def __init__(
        self,
        device: str,
        # Initialization
        config: Config | str | None = None,
        vocab_answer: dict[str, int] | str | None = None,
        path_entity_dict: str | None = None,
        # Loading
        path_snapshot: str | None = None,
    ):
        logger.info("########## MAQA Initialization Starts ##########")

        self.device = device
        self.path_snapshot = path_snapshot

        if path_snapshot is not None:
            assert config is None
            assert vocab_answer is None
            assert path_entity_dict is None
            config = path_snapshot + "/config"
            vocab_answer = path_snapshot + "/answers.vocab.txt"
            path_entity_dict = path_snapshot + "/entity_dict.json"
            path_model = path_snapshot + "/model"

        # Load the configuration
        if isinstance(config, str):
            config_path = config
            config = utils.get_hocon_config(config_path=config_path)
            logger.info(f"Loaded configuration from {config_path}")
        self.config = config
        logger.info(utils.pretty_format_dict(self.config))

        # Load the answer vocabulary
        if isinstance(vocab_answer, str):
            vocab_path = vocab_answer
            vocab_answer = utils.read_vocab(vocab_path)
            logger.info(f"Loaded answer type vocabulary from {vocab_path}")
        self.vocab_answer = vocab_answer
        self.ivocab_answer = {i:l for l, i in self.vocab_answer.items()}

        # Load the entity dictionary
        logger.info(f"Loading entity dictionary from {path_entity_dict}")
        self.entity_dict = {
            epage["entity_id"]: epage
            for epage in utils.read_json(path_entity_dict)
        }
        logger.info(f"Completed loading of entity dictionary with {len(self.entity_dict)} entities from {path_entity_dict}")

        # Load the model
        self.model_name = config["model_name"]
        if self.model_name == "ma_qa_model":
            self.model = MAQAModel(
                device=device,
                bert_pretrained_name_or_path=config["bert_pretrained_name_or_path"],
                max_seg_len=config["max_seg_len"],
                entity_dict=self.entity_dict,
                dataset_name=config["dataset_name"],
                dropout_rate=config["dropout_rate"],
                vocab_answer=self.vocab_answer,
                loss_function_name=config["loss_function"],
                focal_loss_gamma=(
                    config["focal_loss_gamma"] \
                    if config["loss_function"] == "focal_loss" else None
                ),
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

        logger.info("########## MAQA Initialization Ends ##########")

    def save(self, path_snapshot: str, model_only: bool = False) -> None:
        path_config = path_snapshot + "/config"
        path_vocab = path_snapshot + "/answers.vocab.txt"
        path_entity_dict = path_snapshot + "/entity_dict.json"
        path_model = path_snapshot + "/model"
        if not model_only:
            utils.write_json(path_config, self.config)
            utils.write_vocab(path_vocab, self.vocab_answer, write_frequency=False)
            utils.write_json(path_entity_dict, self.entity_dict)
        torch.save(self.model.state_dict(), path_model)

    def compute_loss(
        self,
        document: Document,
        qa_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Switch to training mode
        self.model.train()

        # Preprocess
        preprocessed_data = self.model.preprocess(document=document)

        # Tensorize
        model_input = self.model.tensorize(
            preprocessed_data=preprocessed_data,
            qa_index=qa_index,
            compute_loss=True
        )

        # Forward
        model_output = self.model.forward(**model_input)

        return (
            model_output.loss,
            model_output.acc
        )

    def extract(self, document: Document) -> Document:
        with torch.no_grad():
            # Switch to inference mode
            self.model.eval()

            # Preprocess
            preprocessed_data = self.model.preprocess(document=document)

            # Generate triples iteratively
            triples: list[Triple] = []
            qas = preprocessed_data["qas"]
            for qa_index in range(len(qas)):
                # Tensorize
                model_input = self.model.tensorize(
                    preprocessed_data=preprocessed_data,
                    qa_index=qa_index,
                    compute_loss=False
                )

                # Forward
                model_output = self.model.forward(**model_input)
                logits = model_output.logits # (1, n_answers)

                # Structurize
                pred_answer_label = torch.argmax(logits, dim=1).cpu().item() # int
                pred_answer = self.ivocab_answer[pred_answer_label] # str
                if pred_answer_label != 0:
                    head_entity_i, relation, tail_entity_i = qas[qa_index].triple
                    triples.append({
                        "arg1": int(head_entity_i),
                        "relation": relation,
                        "arg2": int(tail_entity_i),
                        "question": " ".join(qas[qa_index].question),
                        "answer": pred_answer
                    })

            # Integrate
            result_document = copy.deepcopy(document)
            result_document["relations"] = triples
            return result_document

    def batch_extract(self, documents: list[Document]) -> list[Document]:
        result_documents = []
        for document in tqdm(documents, desc="extraction steps"):
            result_document = self.extract(document=document)
            result_documents.append(result_document)
        return result_documents


class MAQATrainer:

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

        # required for Ign evaulation
        paths["path_gold_train_triples"] = self.base_output_path + "/gold_train_triples.json"

        return paths

    def setup_dataset(
        self,
        extractor: MAQA,
        documents: list[Document],
        split: str,
        with_gold_annotations: bool = True
    ) -> None:
        # Cache the gold training triples for Ign evaluation
        if split == "train":
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
        if split !=  "train" and with_gold_annotations:
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
        extractor: MAQA,
        train_documents: list[Document],
        dev_documents: list[Document],
        supplemental_info: dict[str, Any]
    ) -> None:
        ##################
        # Setup
        ##################

        train_doc_indices = np.arange(len(train_documents))

        # We expand the training documents for each QA level,
        # because each document consists of the different number of QAs.
        # First, aggregate (doc_i, qa_index) tuples
        # for positive and negative questions separately.
        pos_train_tuples: list[tuple[int,int]] = []
        neg_train_tuples: list[tuple[int,int]] = []
        for doc_i in train_doc_indices:
            document = train_documents[doc_i]
            preprocessed_data = extractor.preprocessor.preprocess(
                document=document
            )
            qas = preprocessed_data["qas"]
            for qa_i in range(len(qas)):
                answer: str = qas[qa_i].answer
                if answer == "Yes":
                    pos_train_tuples.append((doc_i, qa_i))
                elif answer == "No":
                    neg_train_tuples.append((doc_i, qa_i))
                else:
                    raise Exception(f"Invalid answer: {answer}")
        n_pos_train = len(pos_train_tuples)
        n_neg_train_before_sampling = len(neg_train_tuples)

        # Then, perform negative-question sampling
        if extractor.config["n_negative_samples"] > 0:
            perm = np.random.permutation(len(neg_train_tuples))
            perm = perm[
                0 : len(pos_train_tuples) * extractor.config["n_negative_samples"]
            ]
            neg_train_tuples = [neg_train_tuples[i] for i in perm]
        n_neg_train_after_sampling = len(neg_train_tuples)

        # Finally, concatenate the positive and negative tuples
        train_doc_index_and_qa_index_tuples = pos_train_tuples + neg_train_tuples
        train_doc_index_and_qa_index_tuples = np.asarray(
            train_doc_index_and_qa_index_tuples
        )

        n_train = len(train_doc_index_and_qa_index_tuples)
        max_epoch = extractor.config["max_epoch"]
        batch_size = extractor.config["batch_size"]
        gradient_accumulation_steps = extractor.config["gradient_accumulation_steps"]
        total_update_steps = n_train * max_epoch // (batch_size * gradient_accumulation_steps)
        warmup_steps = int(total_update_steps * extractor.config["warmup_ratio"])

        logger.info(f"Number of training QAs (all): {n_pos_train} (pos) + {n_neg_train_before_sampling} (neg) = {n_pos_train + n_neg_train_before_sampling}")
        logger.info(f"Number of training QAs (after negative sampling): {n_pos_train} (pos) + {n_neg_train_after_sampling} (neg) = {n_pos_train + n_neg_train_after_sampling}")
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
        logger.info(f"Saved config, answer vocabulary, entity dictionary, and model to {self.paths['path_snapshot']}")

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

                for (doc_i, qa_i) in train_doc_index_and_qa_index_tuples[
                    perm[instance_i: instance_i + batch_size]
                ]:
                    doc_i = int(doc_i)
                    qa_i = int(qa_i)

                    # Forward and compute loss
                    one_loss, one_acc = extractor.compute_loss(
                        document=train_documents[doc_i],
                        qa_index=qa_i
                    )

                    # Accumulate the loss
                    batch_loss = batch_loss + one_loss
                    batch_acc = batch_acc + one_acc
                    actual_batchsize += 1

                # Average the loss
                actual_batchsize = float(actual_batchsize)
                batch_loss = batch_loss / actual_batchsize # loss per pair
                batch_acc = batch_acc / actual_batchsize

                ##################
                # Backward
                ##################

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
                    # Report
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
        extractor: MAQA,
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
        scores = evaluation.docre.fscore(
            pred_path=self.paths[f"path_{split}_pred"],
            gold_path=self.paths[f"path_{split}_gold"],
            skip_intra_inter=skip_intra_inter,
            skip_ign=skip_ign,
            gold_train_triples_path=self.paths["path_gold_train_triples"]
        )

        if get_scores_only:
            return scores

        # Save the evaluation scores
        utils.write_json(self.paths[f"path_{split}_eval"], scores)
        logger.info(utils.pretty_format_dict(scores))
        return scores

    def official_evaluate(
        self,
        extractor: MAQA,
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
            path_output=
            self.paths[f"path_{split}_pred"].replace(".json", ".official.json")
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


class TripleTuple(NamedTuple):
    arg1: int
    relation: str
    arg2: int


class QATuple(NamedTuple):
    triple: TripleTuple
    question: list[str]
    answer: str | None


class MAQAModel(nn.Module):

    def __init__(
        self,
        device,
        bert_pretrained_name_or_path,
        max_seg_len,
        entity_dict,
        dataset_name,
        dropout_rate,
        vocab_answer,
        loss_function_name,
        focal_loss_gamma=None,
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
        dataset_name : str
        dropout_rate : float
        vocab_answer : dict[str, int]
        loss_function_name : str
        focal_loss_gamma : float | None
            by default None
        possible_head_entity_types : list[str] | None
            by default None
        possible_tail_entity_types : list[str] | None
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
        self.dataset_name = dataset_name
        self.dropout_rate = dropout_rate
        self.vocab_answer = vocab_answer
        self.loss_function_name = loss_function_name
        self.focal_loss_gamma = focal_loss_gamma
        self.possible_head_entity_types = possible_head_entity_types
        self.possible_tail_entity_types = possible_tail_entity_types
        self.use_mention_as_canonical_name = use_mention_as_canonical_name

        self.n_answers = len(self.vocab_answer)

        ########################
        # Components
        ########################

        # BERT, tokenizer
        self.bert, self.tokenizer = self._initialize_bert_and_tokenizer(
            pretrained_model_name_or_path=self.bert_pretrained_name_or_path
        )

        # Dimensionality
        self.hidden_dim = self.bert.config.hidden_size

        # QA
        self.mlp = make_mlp(
            input_dim=2 * self.hidden_dim,
            hidden_dims=2 * self.hidden_dim,
            output_dim=self.n_answers,
            dropout_rate=self.dropout_rate
        )

        ######
        # Preprocessor
        ######

        self.preprocessor = MAQAPreprocessor(
            tokenizer=self.tokenizer,
            max_seg_len=self.max_seg_len,
            entity_dict=self.entity_dict,
            dataset_name=self.dataset_name,
            vocab_answer=self.vocab_answer,
            possible_head_entity_types=self.possible_head_entity_types,
            possible_tail_entity_types=self.possible_tail_entity_types,
            use_mention_as_canonical_name=self.use_mention_as_canonical_name
        )

        ######
        # Loss Function
        ######

        if self.loss_function_name == "cross_entropy":
            self.loss_function = nn.CrossEntropyLoss(reduction="none")
        elif self.loss_function_name == "focal_loss":
            self.loss_function = FocalLoss(
                gamma=self.focal_loss_gamma,
                reduction="none"
            )
        else:
            raise Exception(f"Invalid loss_function: {self.loss_function_name}")

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

    def preprocess(self, document):
        """
        Parameters
        ----------
        document : Document

        Returns
        -------
        dict[str, Any]
        """
        return self.preprocessor.preprocess(document=document)

    def tensorize(self, preprocessed_data, qa_index, compute_loss):
        """
        Parameters
        ----------
        preprocessed_data : dict[str, Any]
        qa_index : int
        compute_loss : bool

        Returns
        -------
        dict[str, Any]
        """
        model_input = {}

        model_input["compute_loss"] = compute_loss

        # (n_segments, max_seg_len)
        model_input["segments_id"] = torch.tensor(
            preprocessed_data["qa_index_to_bert_input"][qa_index]["segments_id"],
            device=self.device
        )

        # (n_segments, max_seg_len)
        model_input["segments_mask"] = torch.tensor(
            preprocessed_data["qa_index_to_bert_input"][qa_index]["segments_mask"],
            device=self.device
        )

        # (n_segments, max_seg_len)
        model_input["segments_token_type_id"] = torch.tensor(
            preprocessed_data["qa_index_to_bert_input"][qa_index]
            ["segments_token_type_id"],
            device=self.device
        )

        if not compute_loss:
            return model_input

        # (1,)
        model_input["gold_answer_labels"] = torch.tensor(
            preprocessed_data["gold_answer_labels"][qa_index:qa_index+1],
            device=self.device
        ).to(torch.long)

        return model_input

    def forward(
        self,
        segments_id,
        segments_mask,
        segments_token_type_id,
        compute_loss,
        gold_answer_labels=None
    ):
        """
        Parameters
        ----------
        segments_id : torch.Tensor
            shape of (n_segments, max_seg_len)
        segments_mask : torch.Tensor
            shape of (n_segments, max_seg_len)
        segments_token_type_id : torch.Tensor
            shape of (n_segments, max_seg_len)
        compute_loss : bool
        gold_answer_labels : torch.Tensor | None
            shape of (1,); by default None

        Returns
        -------
        ModelOutput
        """
        # Encode tokens by BERT
        # (n_segments, max_seg_len, hidden_dim)
        segments_vec = self.encode_tokens(
            segments_id=segments_id,
            segments_mask=segments_mask,
            segments_token_type_id=segments_token_type_id
        )

        # Compute segment vectors
        # (n_segments, 2 * hidden_dim)
        segment_vectors = self.compute_segment_vectors(
            segments_vec=segments_vec,
            segments_mask=segments_mask
        )

        # Compute a document vector
        # (1, 2 * hidden_dim)
        document_vector = self.compute_document_vector(
            segment_vectors=segment_vectors
        )

        # Compute logits by MLP
        # (1, n_answers)
        logits = self.compute_logits_by_mlp(document_vector=document_vector)

        if not compute_loss:
            return ModelOutput(
                logits=logits
            )

        # Compute loss
        # (1,)
        loss = self.loss_function(logits, gold_answer_labels)
        loss = loss.sum() # Scalar

        # Compute accuracy
        # (1,)
        pred_answer_labels = torch.argmax(logits, dim=1)
        # (1,)
        acc = (pred_answer_labels == gold_answer_labels).to(torch.float)
        acc = acc.sum().item() # Scalar

        return ModelOutput(
            logits=logits,
            loss=loss,
            acc=acc
        )

    ################
    # Subfunctions
    ################

    def encode_tokens(self, segments_id, segments_mask, segments_token_type_id):
        """
        Parameters
        ----------
        segments_id : torch.Tensor
            shape of (n_segments, max_seg_len)
        segments_mask : torch.Tensor
            shape of (n_segments, max_seg_len)
        segments_token_type_id : torch.Tensor
            shape of (n_segments, max_seg_len)

        Returns
        -------
        torch.Tensor
            shape of (n_segments, max_seg_len, hidden_dim)
        """
        # Check
        # assert max_seg_len == self.max_seg_len

        # Encode segments by BERT
        bert_output = self.bert(
            input_ids=segments_id,
            attention_mask=segments_mask,
            token_type_ids=segments_token_type_id,
            output_attentions=False,
            output_hidden_states=False
        )
        # (n_segments, max_seg_len, hidden_dim)
        segments_vec = bert_output["last_hidden_state"]

        return segments_vec

    def compute_segment_vectors(self, segments_vec, segments_mask):
        """
        Parameters
        ----------
        segments_vec : torch.Tensor
            shape of (n_segments, max_seg_len, hidden_dim)
        segments_mask : torch.Tensor
            shape of (n_segments, max_seg_len)

        Returns
        -------
        torch.Tensor
            shape of (n_segments, 2 * hidden_dim)
        """
        # Get [CLS] embeddings
        # (n_segments, hidden_dim)
        cls_vectors = segments_vec[:, 0, :]

        # Get average embeddings
        # (n_segments, max_seg_len, hidden_dim)
        avg_vectors = segments_vec * segments_mask.unsqueeze(-1)
        # (n_segments, hidden_dim)
        avg_vectors = avg_vectors.sum(dim=1)
        # (n_segments, 1)
        n_tokens_for_each_seg = segments_mask.sum(dim=1).unsqueeze(-1)
        # (n_segments, hidden_dim)
        avg_vectors = avg_vectors / n_tokens_for_each_seg

        # (n_segments, 2 * hidden_dim)
        segment_vectors = torch.cat([cls_vectors, avg_vectors], dim=1)
        return segment_vectors

    def compute_document_vector(self, segment_vectors):
        """
        Parameters
        ----------
        segment_vectors : torch.Tensor
            shape of (n_segments, 2 * hidden_dim)

        Returns
        -------
        torch.Tensor
            shape of (1, 2 * hidden_dim)
        """
        # (n_segments, 2 * hidden_dim) -> (1, 2 * hidden_dim)
        document_vector = segment_vectors.logsumexp(dim=0).unsqueeze(0)
        return document_vector

    def compute_logits_by_mlp(self, document_vector):
        """
        Parameters
        ----------
        document_vector : torch.Tensor
            shape of (1, 2 * hidden_dim)

        Returns
        -------
        torch.Tensor
            shape of (1, n_answers)
        """
        logits = self.mlp(document_vector) # (1, n_answers)
        return logits


class MAQAPreprocessor:

    def __init__(
        self,
        tokenizer,
        max_seg_len,
        entity_dict,
        dataset_name,
        vocab_answer,
        possible_head_entity_types=None,
        possible_tail_entity_types=None,
        use_mention_as_canonical_name=False
    ):
        """
        Parameters
        ----------
        tokenizer : PreTrainedTokenizer
        max_seg_len : int
        entity_dict : dict[str, EntityPage]
        dataset_name : str
        vocab_answer: dict[str, int]
        possible_head_entity_types : list[str] | None
            by default None
        possible_tail_entity_types : list[str] | None
            by default None
        use_mention_as_canonical_name : bool
            by default False
        """
        self.tokenizer = tokenizer
        self.max_seg_len = max_seg_len
        self.entity_dict = entity_dict
        self.dataset_name = dataset_name
        self.vocab_answer = vocab_answer
        self.possible_head_entity_types = possible_head_entity_types
        self.possible_tail_entity_types = possible_tail_entity_types
        self.use_mention_as_canonical_name = use_mention_as_canonical_name

        self.qa_generator = QAGenerator(
            dataset_name=self.dataset_name,
            entity_dict=self.entity_dict,
            possible_head_entity_types=self.possible_head_entity_types,
            possible_tail_entity_types=self.possible_tail_entity_types,
            use_mention_as_canonical_name=self.use_mention_as_canonical_name
        )

    def preprocess(self, document):
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
            MentionTuple(None, m["name"], m["entity_type"], m["entity_id"])
            for m in document["mentions"]
        ]
        preprocessed_data["mentions"] = mentions

        entities = [
            EntityTuple(e["mention_indices"], e["entity_type"], e["entity_id"])
            for e in document["entities"]
        ]
        preprocessed_data["entities"] = entities

        with_supervision = True if "relations" in document else False
        if with_supervision:
            relations = [
                TripleTuple(r["arg1"], r["relation"], r["arg2"])
                for r in document["relations"]
            ]
            preprocessed_data["relations"] = relations

        #####
        # qas: list[QATuple]
        #####

        qas = self.qa_generator.generate(document=document) # list[QATuple]
        preprocessed_data["qas"] = qas

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
        # qa_index_to_bert_input: list[dict[str, Any]]
        # segments: list[list[str]]
        # segments_id: list[list[int]]
        # segments_mask: list[list[int]]
        # segments_token_type_id: list[list[int]]
        #####

        qa_index_to_bert_input = [] # list[dict[str, Any]]
        for qa in qas:
            (
                segments,
                segments_id,
                segments_mask,
                segments_token_type_id
            ) = self.tokenize_and_split(
                sentences=sentences,
                question=qa.question
            )
            bert_input = {}
            bert_input["segments"] = segments
            bert_input["segments_id"] = segments_id
            bert_input["segments_mask"] = segments_mask
            bert_input["segments_token_type_id"] = segments_token_type_id
            qa_index_to_bert_input.append(bert_input)
        preprocessed_data["qa_index_to_bert_input"] = qa_index_to_bert_input

        #####
        # gold_answer_labels: list[int]
        #####

        if with_supervision:
            gold_answer_labels = [
                self.vocab_answer[qa.answer]
                for qa in qas
            ] # list[int]
            preprocessed_data["gold_answer_labels"] = gold_answer_labels

        return preprocessed_data

    #####
    # Subfunctions
    #####

    def tokenize_and_split(self, sentences, question):
        """
        Parameters
        ----------
        sentences: list[list[str]]
        question: list[str]

        Returns
        -------
        tuple[list[list[str]], list[list[int]], list[list[int]],
            list[list[int]]]
        """
        first_seq = " ".join(question)
        second_seq = " ".join(utils.flatten_lists(sentences))
        inputs = self.tokenizer(
            first_seq,
            second_seq,
            max_length=self.max_seg_len,
            padding="max_length",
            truncation="only_second",
            return_overflowing_tokens=True
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


class QAGenerator:

    def __init__(
        self,
        dataset_name,
        entity_dict,
        possible_head_entity_types=None,
        possible_tail_entity_types=None,
        use_mention_as_canonical_name=False
    ):
        self.dataset_name = dataset_name
        self.entity_dict = entity_dict
        self.possible_head_entity_types = possible_head_entity_types
        self.possible_tail_entity_types = possible_tail_entity_types
        self.use_mention_as_canonical_name = use_mention_as_canonical_name
        self.templates = TRIPLE_TO_QUESTION_TEMPLATES[self.dataset_name]

    def generate(self, document):
        """
        Parameters
        ----------
        document : Document

        Returns
        -------
        list[QATuple]
        """
        qas = [] # list[QATuple]
        false_negative_triples = []

        entities = document["entities"]
        if "relations" in document:
            with_supervision = True
            relations = document["relations"]
        else:
            with_supervision = False

        # Generate a canonical name list for entities for template-based generation
        # An entity type list will be used for type-based filtering (dataset specific)
        entity_names = [] # list[str]
        entity_types = [] # list[str]
        for e_i in range(len(entities)):
            entity_id = entities[e_i]["entity_id"] # str
            if self.use_mention_as_canonical_name:
                m_i = entities[e_i]["mention_indices"][0]
                canonical_name = document["mentions"][m_i]["name"]
            else:
                epage = self.entity_dict[entity_id] # dict
                canonical_name = epage["canonical_name"] # str
            entity_type = entities[e_i]["entity_type"] # str
            entity_names.append(canonical_name)
            entity_types.append(entity_type)

        # `not_include_entity_pairs` will be used for filtering
        not_include_entity_pairs = None
        if "not_include_pairs" in document:
            # List[(int, int)]
            epairs = [
                (epair["arg1"], epair["arg2"])
                for epair in document["not_include_pairs"]
            ]
            not_include_entity_pairs \
                = [(e1,e2) for e1,e2 in epairs] + [(e2,e1) for e1,e2 in epairs]

        # Create QAs
        for head_entity_i in range(len(entities)):
            for tail_entity_i in range(len(entities)):
                # Skip diagonal
                if head_entity_i == tail_entity_i:
                    continue

                # Skip based on entity types if specified
                # e.g, Skip chemical-chemical, disease-disease,
                #   and disease-chemical pairs for CDR.
                if (
                    (self.possible_head_entity_types is not None)
                    and
                    (self.possible_tail_entity_types is not None)
                ):
                    head_entity_type = entity_types[head_entity_i]
                    tail_entity_type = entity_types[tail_entity_i]
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

                if with_supervision:
                    gold_rels = self.find_relations(
                        arg1=head_entity_i,
                        arg2=tail_entity_i,
                        relations=relations
                    )
                # NOTE:
                # Please note that we generate questions only for relations written in `templates`.
                # This can result in lower recall scores,
                # since gold triples for relations not written in `templates` cannot be predicted (i.e., always false negative).
                for relation in self.templates.keys():
                    # Generate a question
                    question = self.templates[relation].replace(
                        HEAD_ENTITY_SLOT, entity_names[head_entity_i]
                    ).replace(
                        TAIL_ENTITY_SLOT, entity_names[tail_entity_i]
                    )
                    # Generate the answer
                    if with_supervision:
                        if relation in gold_rels:
                            answer = "Yes"
                        else:
                            answer = "No"
                    # QA instance
                    if with_supervision:
                        qa = QATuple(
                                TripleTuple(
                                    int(head_entity_i),
                                    relation,
                                    int(tail_entity_i)
                                ),
                                question.split(),
                                answer
                            )
                    else:
                        qa = QATuple(
                                TripleTuple(
                                    int(head_entity_i),
                                    relation,
                                    int(tail_entity_i)
                                ),
                                question.split(),
                                None
                            )
                    qas.append(qa)
                if with_supervision:
                    # `false_negative_rels` cannot be generated
                    false_negative_rels \
                        = set(gold_rels) - set(self.templates.keys())
                    # We record such false-negative triples
                    if len(false_negative_rels) > 0:
                        false_negative_triples.extend([
                            (
                                entities[head_entity_i]["entity_id"],
                                r,
                                entities[tail_entity_i]["entity_id"]
                            )
                            for r in false_negative_rels
                        ])

        assert len(qas) > 0
        if len(false_negative_rels) > 0:
            logger.warning("Questions for the following triple(s) are not generated, since corresponding templates cannot be found for the relations:")
            for x in false_negative_triples:
                logger.warning(f"{x}")
        return qas

    def find_relations(self, arg1, arg2, relations):
        """
        Parameters
        ----------
        arg1 : int
        arg2 : int
        relations : list[dict[str, int|str]]

        Returns
        -------
        List[str]
        """
        rels = []
        for triple in relations:
            if triple["arg1"] == arg1 and triple["arg2"] == arg2:
                rels.append(triple["relation"])
        return rels

