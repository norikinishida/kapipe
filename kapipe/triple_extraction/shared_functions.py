from __future__ import annotations

from typing import Any

from torch.optim import Optimizer, Adam, AdamW
# from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR

from ..datatypes import Config, Document


###################################
# Optimizers
###################################


def get_optimizer(model: Any, config: Config) -> list[Optimizer]:
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param, task_param = model.get_params(named=True)
    grouped_bert_param = [
        {
            "params": [
                p for n, p in bert_param
                if not any(nd in n for nd in no_decay)
            ],
            "lr": config["bert_learning_rate"],
            "weight_decay": config["adam_weight_decay"],
        },
        {
            "params": [
                p for n, p in bert_param
                if any(nd in n for nd in no_decay)
            ],
            "lr": config["bert_learning_rate"],
            "weight_decay": 0.0,
        }
    ]
    optimizers = [
        AdamW(
            grouped_bert_param,
            lr=config["bert_learning_rate"],
            eps=config["adam_eps"]
        ),
        Adam(
            model.get_params()[1],
            lr=config["task_learning_rate"],
            eps=config["adam_eps"],
            weight_decay=0
        )
    ]
    return optimizers


def get_optimizer2(model: Any, config: Config) -> Optimizer:
    bert_param, task_param = model.get_params()
    grouped_param = [
        {
            "params": bert_param,
        },
        {
            "params": task_param,
            "lr": config["task_learning_rate"]
        },
    ]
    optimizer = AdamW(
        grouped_param,
        lr=config["bert_learning_rate"],
        eps=config["adam_eps"]
    )
    return optimizer


###################################
# Schedulers
###################################


def get_scheduler(
    optimizers: list[Optimizer],
    total_update_steps: int,
    warmup_steps: int
) -> list[LambdaLR]:
    def lr_lambda_bert(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_update_steps - current_step) / float(max(
                1,
                total_update_steps - warmup_steps
            ))
        )

    def lr_lambda_task(current_step):
        return max(
            0.0,
            float(total_update_steps - current_step) / float(max(
                1,
                total_update_steps
            ))
        )

    schedulers = [
        LambdaLR(optimizers[0], lr_lambda_bert),
        LambdaLR(optimizers[1], lr_lambda_task)
    ]
    return schedulers


def get_scheduler2(
    optimizer: Optimizer,
    total_update_steps: int,
    warmup_steps: int
) -> LambdaLR:
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps
    )


###################################
# Data processing
###################################


def create_intra_inter_map(document: Document) -> dict[str, str]:
    intra_inter_map = {}

    # We first create token-index-to-sentence-index mapping
    token_index_to_sent_index = [] # dict[int, int], i.e., list[int]
    for sent_i, sent in enumerate(document["sentences"]):
        sent_words = sent.split()
        token_index_to_sent_index.extend(
            [sent_i for _ in range(len(sent_words))]
        )
    # We then create mention-index-to-sentence-index mapping
    mention_index_to_sentence_index = [] # list[int]
    for mention in document["mentions"]:
        begin_token_index, end_token_index = mention["span"]
        sentence_index = token_index_to_sent_index[begin_token_index]
        assert token_index_to_sent_index[end_token_index] == sentence_index
        mention_index_to_sentence_index.append(sentence_index)

    entities = document["entities"]
    for u_entity_i in range(len(entities)):
        u_entity = entities[u_entity_i]
        u_mention_indices = u_entity["mention_indices"]
        u_sent_indices = [
            mention_index_to_sentence_index[i] for i in u_mention_indices
        ]
        u_sent_indices = set(u_sent_indices)
        for v_entity_i in range(u_entity_i, len(entities)):
            v_entity = entities[v_entity_i]
            v_mention_indices = v_entity["mention_indices"]
            v_sent_indices = [
                mention_index_to_sentence_index[i] for i in v_mention_indices
            ]
            v_sent_indices = set(v_sent_indices)
            if len(u_sent_indices & v_sent_indices) == 0:
                # No co-occurent mention pairs
                intra_inter_map[f"{u_entity_i}-{v_entity_i}"] = "inter"
                intra_inter_map[f"{v_entity_i}-{u_entity_i}"] = "inter"
            else:
                # There is at least one co-occurent mention pairs
                intra_inter_map[f"{u_entity_i}-{v_entity_i}"] = "intra"
                intra_inter_map[f"{v_entity_i}-{u_entity_i}"] = "intra"
    return intra_inter_map


def create_seen_unseen_map(
    document: Document,
    seen_pairs: set[tuple[str, str]]
) -> dict[str, str]:
    seen_unseen_map = {}
    entities = document["entities"]
    for u_entity_i in range(len(entities)):
        u_entity = entities[u_entity_i]
        u_entity_id = u_entity["entity_id"]
        for v_entity_i in range(u_entity_i, len(entities)):
            v_entity = entities[v_entity_i]
            v_entity_id = v_entity["entity_id"]
            if (
                ((u_entity_id, v_entity_id) in seen_pairs)
                or
                ((v_entity_id, u_entity_id) in seen_pairs)
            ):
                seen_unseen_map[f"{u_entity_id}-{v_entity_id}"] = "seen"
                seen_unseen_map[f"{v_entity_id}-{u_entity_id}"] = "seen"
            else:
                seen_unseen_map[f"{u_entity_id}-{v_entity_id}"] = "unseen"
                seen_unseen_map[f"{v_entity_id}-{u_entity_id}"] = "unseen"
    return seen_unseen_map

