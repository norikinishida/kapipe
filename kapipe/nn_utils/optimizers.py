from __future__ import annotations

from typing import Any

from torch.optim import Optimizer, Adam, AdamW
# from transformers import AdamW


def get_optimizer(
    model: Any,
    config: dict[str, Any],
) -> list[Optimizer]:
    """Return a list of optimizers for the model parameters, separating BERT and task-specific parameters."""

    # Define parameter names excluded from weight decay
    no_decay = ["bias", "LayerNorm.weight"]

    # Retrieve named BERT parameters and task parameters
    bert_param, task_param = model.get_params(named=True)

    # Group BERT parameters by whether weight decay is applied
    grouped_bert_param = [
        # Parameters subject to weight decay
        {
            "params": [
                p
                for n, p in bert_param
                if not any(nd in n for nd in no_decay)
            ],
            "lr": config["bert_learning_rate"],
            "weight_decay": config["adam_weight_decay"],
        },
        # Parameters excluded from weight decay
        {
            "params": [
                p
                for n, p in bert_param
                if any(nd in n for nd in no_decay)
            ],
            "lr": config["bert_learning_rate"],
            "weight_decay": 0.0,
        }
    ]

    # Create separate optimizers for BERT and task parameters
    optimizers = [
        # AdamW optimizer for BERT parameters
        AdamW(
            grouped_bert_param,
            lr=config["bert_learning_rate"],
            eps=config["adam_eps"]
        ),
        # Adam optimizer for task parameters
        Adam(
            model.get_params()[1],
            lr=config["task_learning_rate"],
            eps=config["adam_eps"],
            weight_decay=0
        )
    ]

    return optimizers


def get_optimizer2(
    model: Any,
    config: dict[str, Any],
) -> Optimizer:
    """Return a single optimizer for the model parameters, with separate learning rates for BERT and task-specific parameters."""

    # Retrieve BERT parameters and task parameters
    bert_param, task_param = model.get_params()

    # Group parameters by their learning-rate settings
    grouped_param = [
        # Use the default optimizer learning rate for BERT parameters
        {
            "params": bert_param,
        },
        # Use the task-specific learning rate for task parameters
        {
            "params": task_param,
            "lr": config["task_learning_rate"]
        },
    ]

    # Create one AdamW optimizer for both parameter groups
    optimizer = AdamW(
        grouped_param,
        lr=config["bert_learning_rate"],
        eps=config["adam_eps"]
    )

    return optimizer

