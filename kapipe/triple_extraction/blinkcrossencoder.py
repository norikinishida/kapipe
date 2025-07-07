import copy
import logging
import os

import numpy as np
import torch
from tqdm import tqdm
import jsonlines

from . import shared_functions
from ..models import BlinkCrossEncoderModel
from .. import evaluation
from .. import utils
from ..utils import BestScoreHolder


logger = logging.getLogger(__name__)


class BlinkCrossEncoder:
    """Cross-Encoder in BLINK (Wu et al., 2020).
    """

    def __init__(
        self,
        # General
        device,
        config,
        # Task specific
        path_entity_dict,
        # Misc.
        path_model=None,
        verbose=True
    ):
        """
        Parameters
        ----------
        device: str
        config: ConfigTree | str
        path_entity_dict: str
        path_model: str | None
            by default None
        verbose: bool
            by default True
        """
        self.verbose = verbose
        if self.verbose:
            logger.info(">>>>>>>>>> BlinkCrossEncoder Initialization >>>>>>>>>>")
        self.device = device

        ######
        # Config
        ######

        if isinstance(config, str):
            tmp = config
            config = utils.get_hocon_config(
                config_path=config,
                config_name=None
            )
            if self.verbose:
                logger.info(f"Loaded configuration from {tmp}")
        self.config = config
        if self.verbose:
            logger.info(utils.pretty_format_dict(self.config))

        ######
        # Model
        ######

        self.model_name = config["model_name"]

        self.entity_dict = {
            epage["entity_id"]: epage
            for epage in utils.read_json(path_entity_dict)
        }
        if self.verbose:
            logger.info(f"Loaded entity dictionary from {path_entity_dict}")

        if self.model_name == "blinkcrossencodermodel":
            self.model = BlinkCrossEncoderModel(
                device=device,
                bert_pretrained_name_or_path=config[
                    "bert_pretrained_name_or_path"
                ],
                max_seg_len=config["max_seg_len"],
                entity_dict=self.entity_dict,
                mention_context_length=self.config["mention_context_length"]
            )
        else:
            raise Exception(f"Invalid model_name: {self.model_name}")

        # Show parameter shapes
        # logger.info("Model parameters:")
        # for name, param in self.model.named_parameters():
        #     logger.infof"{name}: {tuple(param.shape)}")

        # Load trained model parameters
        if path_model is not None:
            self.load_model(path=path_model)
            if self.verbose:
                logger.info(f"Loaded model parameters from {path_model}")

        self.model.to(self.model.device)

        if self.verbose:
            logger.info("<<<<<<<<<< BlinkCrossEncoder Initialization <<<<<<<<<<")

    def load_model(self, path):
        """
        Parameters
        ----------
        path : str
        """
        self.model.load_state_dict(
            torch.load(path, map_location=torch.device("cpu")),
            strict=False
        )

    def save_model(self, path):
        """
        Parameters
        ----------
        path : str
        """
        torch.save(self.model.state_dict(), path)

    def compute_loss(
        self,
        document,
        candidate_entities_for_doc,
        mention_index
    ):
        """
        Parameters
        ----------
        document : Document
        candidate_entities_for_doc : dict[str, str | list[list[CandEntKeyInfo]]]
        mention_index : int

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
        """
        assert document["doc_key"] == candidate_entities_for_doc["doc_key"]

        # Switch to training mode
        self.model.train()

        # Preprocess
        preprocessed_data = self.model.preprocess(
            document=document,
            candidate_entities_for_doc=candidate_entities_for_doc,
            max_n_candidates=self.config["max_n_candidates_in_training"]
        )

        # Tensorize
        model_input = self.model.tensorize(
            preprocessed_data=preprocessed_data,
            mention_index=mention_index,
            compute_loss=True
        )

        # Forward
        model_output = self.model.forward(**model_input)

        return (
            model_output.loss,
            model_output.acc
        )

    def extract(self, document, candidate_entities_for_doc):
        """
        Parameters
        ----------
        document : Document
        candidate_entities_for_doc : dict[str, str | list[list[CandEntKeyInfo]]]

        Returns
        -------
        Document
        """
        assert document["doc_key"] == candidate_entities_for_doc["doc_key"]

        with torch.no_grad():
            # Switch to inference mode
            self.model.eval()

            if len(document["mentions"]) == 0:
                result_document = copy.deepcopy(document)
                result_document["entities"] = []
                return result_document

            # Preprocess
            preprocessed_data = self.model.preprocess(
                document=document,
                candidate_entities_for_doc=candidate_entities_for_doc,
                max_n_candidates=self.config["max_n_candidates_in_inference"]
            )

            # Get outputs (mention-level) by iterating over the mentions
            mentions = [] # list[Mention]
            cands_for_mentions \
                = candidate_entities_for_doc["candidate_entities"]
            for mention_index in range(len(preprocessed_data["mentions"])):
                # Tensorize
                model_input = self.model.tensorize(
                    preprocessed_data=preprocessed_data,
                    mention_index=mention_index,
                    compute_loss=False
                )

                # Forward
                model_output = self.model.forward(**model_input)
                # (1, n_candidates)
                logits = model_output.logits

                # Structurize (1)
                # Transform logits to mention-level entity IDs
                pred_candidate_entity_index \
                    = torch.argmax(logits, dim=1).cpu().item() # int
                pred_candidate_entity_id = (
                    cands_for_mentions
                    [mention_index]
                    [pred_candidate_entity_index]
                    ["entity_id"]
                )
                mentions.append({
                    "entity_id": pred_candidate_entity_id,
                })

            # Structurize (2)
            # Transform to entity-level entity IDs
            # i.e., aggregate mentions based on the entity IDs
            entities = utils.aggregate_mentions_to_entities(
                document=document,
                mentions=mentions
            )

            # Integrate
            result_document = copy.deepcopy(document)
            for m_i in range(len(result_document["mentions"])):
                result_document["mentions"][m_i].update(mentions[m_i])
            result_document["entities"] = entities
            return result_document

    def batch_extract(self, documents, candidate_entities):
        """
        Parameters
        ----------
        documents : list[Document]
        candidate_entities : list[dict[str, str | list[list[CandEntKeyInfo]]]]

        Returns
        -------
        list[Document
        """
        result_documents = []
        for document, candidate_entities_for_doc in tqdm(
            zip(documents, candidate_entities),
            total=len(documents),
            desc="extraction steps"
        ):
            result_document = self.extract(
                document=document,
                candidate_entities_for_doc=candidate_entities_for_doc
            )
            result_documents.append(result_document)
        return result_documents


class BlinkCrossEncoderTrainer:

    def __init__(self, base_output_path):
        """
        Parameters
        ----------
        base_output_path : str
        """
        self.base_output_path = base_output_path
        self.paths = self.get_paths()

    def get_paths(self):
        """
        Returns
        -------
        dict[str, str]
        """
        paths = {}

        # Path to config file
        paths["path_config"] = self.base_output_path + "/config"
        # Path to model snapshot
        paths["path_snapshot"] = self.base_output_path + "/model"

        # Paths to training-losses and validation-scores files
        paths["path_train_losses"] = self.base_output_path + "/train.losses.jsonl"
        paths["path_dev_evals"] = self.base_output_path + "/dev.eval.jsonl"

        # Paths to validation outputs and scores
        paths["path_dev_gold"] = self.base_output_path + "/dev.gold.json"
        paths["path_dev_pred"] = self.base_output_path + "/dev.pred.json"
        paths["path_dev_eval"] = self.base_output_path + "/dev.eval.json"

        # Paths to evaluation outputs and scores
        paths["path_test_gold"] = self.base_output_path + "/test.gold.json"
        paths["path_test_pred"] = self.base_output_path + "/test.pred.json"
        paths["path_test_eval"] = self.base_output_path + "/test.eval.json"

        return paths

    def setup_dataset(self, extractor, documents, candidate_entities, split):
        """
        Parameters
        ----------
        extractor : BlinkCrossEncoder
        documents : list[Document]
        candidate_entities : list[dict[str, list[list[CandEntKeyInfo]]]]
        split : str
        """
        path_gold = self.paths[f"path_{split}_gold"]
        if not os.path.exists(path_gold):
            kb_entity_ids = set(list(extractor.entity_dict.keys()))
            gold_documents = []
            for document, candidate_entities_for_doc in tqdm(
                zip(documents, candidate_entities),
                desc="dataset setup"
            ):
                gold_doc = copy.deepcopy(document)

                cands_for_mentions \
                    = candidate_entities_for_doc["candidate_entities"]
                mentions = document["mentions"]
                assert len(mentions) == len(cands_for_mentions)

                for m_i, (mention, cands_for_mention) in enumerate(zip(
                    mentions,
                    cands_for_mentions
                )):
                    cand_entity_ids = [
                        c["entity_id"] for c in cands_for_mention
                    ]
                    entity_id = mention["entity_id"]
                    in_kb = entity_id in kb_entity_ids
                    in_cand = entity_id in cand_entity_ids
                    gold_doc["mentions"][m_i]["in_kb"] = in_kb
                    gold_doc["mentions"][m_i]["in_cand"] = in_cand
                gold_documents.append(gold_doc)
            utils.write_json(path_gold, gold_documents)
            logger.info(f"Saved the gold annotations for evaluation in {path_gold}")

    def train(
        self,
        extractor,
        train_documents,
        train_candidate_entities,
        dev_documents,
        dev_candidate_entities
    ):
        """
        Parameters
        ----------
        extractor : BlinkCrossEncoder
        train_documents : list[Document]
        train_candidate_entities : list[dict[str, list[list[CandEntKeyInfo]]]]
        dev_documents : list[Document]
        dev_candidate_entities : list[dict[str, list[list[CandEntKeyInfo]]]]
        """
        # Collect tuples of (document index, mention index, gold entity rank in candidates)
        train_doc_index_and_mention_index_tuples = [] # list[tuple[int,int,int]]
        train_doc_indices = np.arange(len(train_documents))
        for doc_i in train_doc_indices:
            ranks = train_candidate_entities[doc_i]["original_gold_entity_rank_list"] # list[int]
            for m_i in range(len(train_documents[doc_i]["mentions"])):
                rank = ranks[m_i]
                train_doc_index_and_mention_index_tuples.append((doc_i, m_i, rank))
        # Sort the tuples based on their ranks in descending order
        train_doc_index_and_mention_index_tuples = sorted(train_doc_index_and_mention_index_tuples, key=lambda tpl: -tpl[-1])
        train_doc_index_and_mention_index_tuples = np.asarray(train_doc_index_and_mention_index_tuples)
        # Limit the training instances to MAX_TRAINING_INSTANCES
        if extractor.config["max_training_instances"] is not None:
            n_prev_instances = len(train_doc_index_and_mention_index_tuples)
            train_doc_index_and_mention_index_tuples = train_doc_index_and_mention_index_tuples[:extractor.config["max_training_instances"]]
            n_new_instances = len(train_doc_index_and_mention_index_tuples)
            if n_prev_instances != n_new_instances:
                logger.info("Removed training mentions where the gold entity appears at a lower rank among the candidates")
                logger.info(f"{n_prev_instances} -> {n_new_instances} mentions")

        ##################
        # Get optimizer and scheduler
        ##################

        n_train = len(train_doc_index_and_mention_index_tuples)
        max_epoch = extractor.config["max_epoch"]
        batch_size = extractor.config["batch_size"]
        gradient_accumulation_steps = extractor.config["gradient_accumulation_steps"]
        total_update_steps = n_train * max_epoch // (batch_size * gradient_accumulation_steps)
        warmup_steps = int(total_update_steps * extractor.config["warmup_ratio"])

        logger.info("Number of training mentions: %d" % n_train)
        logger.info("Number of epochs: %d" % max_epoch)
        logger.info("Batch size: %d" % batch_size)
        logger.info("Gradient accumulation steps: %d" % gradient_accumulation_steps)
        logger.info("Total update steps: %d" % total_update_steps)
        logger.info("Warmup steps: %d" % warmup_steps)

        optimizer = shared_functions.get_optimizer2(
            model=extractor.model,
            config=extractor.config
        )
        # extractor.model, optimizer = amp.initialize(
        #     extractor.model,
        #     optimizer,
        #     opt_level="O1",
        #     verbosity=0
        # )
        scheduler = shared_functions.get_scheduler2(
            optimizer=optimizer,
            total_update_steps=total_update_steps,
            warmup_steps=warmup_steps
        )

        ##################
        # Get reporter and best score holder
        ##################

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
        # Evaluate
        ##################

        scores = self.evaluate(
            extractor=extractor,
            documents=dev_documents,
            candidate_entities=dev_candidate_entities,
            split="dev",
            #
            get_scores_only=True
        )
        scores["epoch"] = 0
        scores["step"] = 0
        writer_dev.write(scores)
        logger.info(utils.pretty_format_dict(scores))

        bestscore_holder.compare_scores(
            scores["inkb_normalized_accuracy"]["accuracy"],
            0
        )

        ##################
        # Save
        ##################

        # Save the model
        extractor.save_model(path=self.paths["path_snapshot"])
        logger.info("Saved model to %s" % self.paths["path_snapshot"])

        # Save the config (only once)
        # utils.dump_hocon_config(self.paths["path_config"], extractor.config)
        utils.write_json(self.paths["path_config"], extractor.config)
        logger.info("Saved config file to %s" % self.paths["path_config"])

        ##################
        # Training-and-validation loops
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

                for (doc_i, mention_index, _) in (
                    train_doc_index_and_mention_index_tuples[
                        perm[instance_i : instance_i + batch_size]
                    ]
                ):
                    doc_i = int(doc_i)
                    mention_index = int(mention_index)

                    # Forward and compute loss
                    one_loss, one_acc = extractor.compute_loss(
                        document=train_documents[doc_i],
                        candidate_entities_for_doc=\
                            train_candidate_entities[doc_i],
                        mention_index=mention_index
                    )

                    # Accumulate the loss
                    batch_loss = batch_loss + one_loss
                    batch_acc = batch_acc + one_acc
                    actual_batchsize += 1

                # Average the loss
                actual_batchsize = float(actual_batchsize)
                batch_loss = batch_loss / actual_batchsize # loss per mention
                batch_acc = batch_acc / actual_batchsize # accuracy per mention

                ##################
                # Backward
                ##################

                batch_loss = batch_loss / gradient_accumulation_steps
                batch_loss.backward()
                # with amp.scale_loss(batch_loss, optimizer) as scaled_loss:
                #     scaled_loss.backward()

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
                        # torch.nn.utils.clip_grad_norm_(
                        #     amp.master_params(optimizer),
                        #     extractor.config["max_grad_norm"]
                        # )
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

                    out = {
                        "step": step,
                        "epoch": epoch,
                        "step_progress": "%d/%d" % (step, total_update_steps),
                        "step_progress(ratio)": \
                            float(step) / total_update_steps * 100.0,
                        "one_epoch_progress": \
                            "%d/%d" % (instance_i + actual_batchsize, n_train),
                        "one_epoch_progress(ratio)": (
                            float(instance_i + actual_batchsize)
                            / n_train
                            * 100.0
                        ),
                        "loss": loss_accum / accum_count,
                        "accuracy": 100.0 * acc_accum / accum_count,
                        "max_valid_inkb_acc": \
                            bestscore_holder.best_score,
                        "patience": bestscore_holder.patience
                    }
                    writer_train.write(out)
                    logger.info(utils.pretty_format_dict(out))
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
                    # Evaluate
                    ##################

                    scores = self.evaluate(
                        extractor=extractor,
                        documents=dev_documents,
                        candidate_entities=dev_candidate_entities,
                        split="dev",
                        #
                        get_scores_only=True
                    )
                    scores["epoch"] = epoch
                    scores["step"] = step
                    writer_dev.write(scores)
                    logger.info(utils.pretty_format_dict(scores))

                    did_update = bestscore_holder.compare_scores(
                        scores["inkb_accuracy"]["accuracy"],
                        epoch
                    )
                    logger.info("[Step %d] Max validation InKB normalized accuracy: %f" % (step, bestscore_holder.best_score))

                    ##################
                    # Save
                    ##################

                    if did_update:
                        extractor.save_model(path=self.paths["path_snapshot"])
                        logger.info("Saved model to %s" % self.paths["path_snapshot"])

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
        extractor,
        documents,
        candidate_entities,
        split,
        #
        get_scores_only=False,
    ):
        """
        Parameters
        ----------
        extractor : BlinkCrossEncoder
        documents : list[Document]
        candidate_entities : list[dict[str, list[list[CandEntKeyInfo]]]]
        split : str
        get_scores_only : bool
            by default False

        Returns
        -------
        dict[str, Any]
        """
        # (documents, candidate_entities) -> path_pred
        result_documents = extractor.batch_extract(
            documents=documents,
            candidate_entities=candidate_entities
        )
        utils.write_json(self.paths[f"path_{split}_pred"], result_documents)
        # (path_pred, path_gold) -> scores
        scores = evaluation.ed.accuracy(
            pred_path=self.paths[f"path_{split}_pred"],
            gold_path=self.paths[f"path_{split}_gold"],
            inkb=True
        )
        scores.update(evaluation.ed.fscore(
            pred_path=self.paths[f"path_{split}_pred"],
            gold_path=self.paths[f"path_{split}_gold"],
            inkb=True
        ))
        if get_scores_only:
            return scores
        # scores -> path_eval
        utils.write_json(self.paths[f"path_{split}_eval"], scores)
        logger.info(utils.pretty_format_dict(scores))
        return scores

