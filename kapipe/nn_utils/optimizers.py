from __future__ import annotations

from torch.optim import Optimizer, Adam, AdamW
# from transformers import AdamW


def get_optimizer(model, config) -> list[Optimizer]:
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


def get_optimizer2(model, config) -> Optimizer:
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

