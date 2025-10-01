from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.optim import Optimizer, Adam, AdamW
# from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR

from ..datatypes import Config, Document


def make_embedding(dict_size, dim, std=0.02):
    """
    Parameters
    ----------
    dict_size: int
    dim: int
    std: float
        by default 0.02

    Returns
    -------
    nn.Embedding
    """
    emb = nn.Embedding(dict_size, dim)
    init.normal_(emb.weight, std=std)
    return emb


def make_linear(
    input_dim,
    output_dim,
    bias=True,
    std=0.02
):
    """
    Parameters
    ----------
    input_dim: int
    output_dim: int
    bias: bool
        by default True
    std: float
        by default 0.02

    Returns
    -------
    nn.Linear
    """
    linear = nn.Linear(input_dim, output_dim, bias)
    # init.normal_(linear.weight, std=std)
    # if bias:
    #     init.zeros_(linear.bias)
    return linear


def make_mlp(
    input_dim,
    hidden_dims,
    output_dim,
    dropout_rate
):
    """
    Parameters
    ----------
    input_dim: int
    hidden_dims: list[int] | int | None
    output_dim: int
    dropout_rate: float

    Returns
    -------
    nn.Sequential | nn.Linear
    """
    if (
        (hidden_dims is None)
        or (hidden_dims == 0)
        or (hidden_dims == [])
        or (hidden_dims == [0])
    ):
        return nn.Linear(input_dim, output_dim)

    if not isinstance(hidden_dims, Iterable):
        hidden_dims = [hidden_dims]

    mlp = [
        nn.Linear(input_dim, hidden_dims[0]),
        nn.ReLU(),
        nn.Dropout(p=dropout_rate)
    ]
    for i in range(1, len(hidden_dims)):
        mlp += [
            nn.Linear(hidden_dims[i-1], hidden_dims[i]),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        ]
    mlp.append(nn.Linear(hidden_dims[-1], output_dim))
    return nn.Sequential(*mlp)


def make_mlp_hidden(input_dim, hidden_dim, dropout_rate):
    """
    Parameters
    ----------
    input_dim: int
    hidden_dim: int
    dropout_rate: float

    Returns
    -------
    nn.Sequential
    """
    mlp = [
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=dropout_rate)
    ]
    return nn.Sequential(*mlp)


class Biaffine(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim=1,
        bias_x=True,
        bias_y=True
    ):
        """
        Parameters
        ----------
        input_dim: int
        output_dim: int
            by default 1
        bias_x: bool
            by default True
        bias_y: bool
            by default True
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(
            torch.Tensor(output_dim, input_dim+bias_x, input_dim+bias_y)
        )

        self.reset_parameters()

    def __repr__(self):
        s = f"input_dim={self.input_dim}, output_dim={self.output_dim}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        # nn.init.zeros_(self.weight)
        init.normal_(self.weight, std=0.02)

    def forward(self, x, y):
        """
        Parameters
        ----------
        x: torch.Tensor
            shape of (batch_size, seq_len, input_dim)
        y: torch.Tensor
            shape of (batch_size, seq_len, input_dim)

        Returns
        -------
        torch.Tensor
            A scoring tensor of shape
                ``[batch_size, output_dim, seq_len, seq_len]``.
            If ``output_dim=1``, the dimension for ``output_dim`` will be
                squeezed automatically.
        """
        if self.bias_x:
            # (batch_size, seq_len, input_dim+1)
            x = torch.cat(
                (x, torch.ones_like(x[..., :1])),
                -1
            )
        if self.bias_y:
            # (batch_size, seq_len, input_dim+1)
            y = torch.cat(
                (y, torch.ones_like(y[..., :1])),
                -1
            )
        # (batch_size, output_dim, seq_len, seq_len)
        s = torch.einsum(
            'bxi,oij,byj->boxy',
            x,
            self.weight,
            y
        )
        return s


def make_transformer_encoder(
    input_dim,
    n_heads,
    ffnn_dim,
    dropout_rate,
    n_layers
):
    """
    Parameters
    ----------
    input_dim : int
    n_heads : int
    ffnn_dim : int
    dropout_rate : float
    n_layers : int

    Returns
    -------
    nn.TransformerEncoder
    """
    transformer_encoder_layer = nn.TransformerEncoderLayer(
        d_model=input_dim,
        nhead=n_heads,
        dim_feedforward=ffnn_dim,
        dropout=dropout_rate
    )
    transformer_encoder = nn.TransformerEncoder(
        transformer_encoder_layer,
        num_layers=n_layers
    )
    return transformer_encoder


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

