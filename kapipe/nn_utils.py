from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.optim import Optimizer, Adam, AdamW
# from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR


###################################
# Layers
###################################


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
# Loss Functions
###################################


class MarginalizedCrossEntropyLoss(nn.Module):
    """A marginalized cross entropy loss, which can be used in multi-positive classification setup.
    """
    def __init__(self, reduction="none"):
        super().__init__()
        self.reduction = reduction

    def forward(self, output, target):
        """
        Parameters
        ----------
        output : torch.Tensor
            shape of (batch_size, n_labels)
        target : torch.Tensor
            shape of (batch_size, n_labels); binary

        Returns
        -------
        torch.Tensor
            shape of (batch_size,), or scalar
        """
        # output: (batch_size, n_labels)
        # target: (batch_size, n_labels); binary

        # Loss = sum_{i} L_{i}
        # L_{i}
        #   = -log[ sum_{k} exp(y_{i,k} + m_{i,k}) / sum_{k} exp(y_{i,k}) ]
        #   = -(
        #       log[ sum_{k} exp(y_{i,k} + m_{i,k}) ]
        #       - log[ sum_{k} exp(y_{i,k}) ]
        #       )
        #   = log[sum_{k} exp(y_{i,k})] - log[sum_{k} exp(y_{i,k} + m_{i,k})]
        # (batch_size,)
        logsumexp_all = torch.logsumexp(output, dim=1)
        mask = torch.log(target.to(torch.float)) # 1 -> 0; 0 -> -inf
        logsumexp_pos = torch.logsumexp(output + mask, dim=1)
        # (batch_size,)
        loss = logsumexp_all - logsumexp_pos

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss


class FocalLoss(nn.CrossEntropyLoss):
    """Focal loss.
    """
    def __init__(
        self,
        gamma,
        alpha=None,
        ignore_index=-100,
        reduction="none"
    ):
        """
        Parameters
        ----------
        gamma : float
        alpha : float | None, optional
            by default None
        ignore_index : int, optional
            by default -100
        reduction : str, optional
            by default "none"
        """
        super().__init__(
            weight=alpha,
            ignore_index=ignore_index,
            reduction="none"
        )
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, output, target):
        """
        Parameters
        ----------
        output : torch.Tensor
            shape of (N, C, H, W)
        target : torch.Tensor
            shape of (N, H, W)

        Returns
        -------
        torch.Tensor
            shape of (N, H, W), or scalar
        """
        # (N, H, W)
        target = target * (target != self.ignore_index).long()
        # (N, H, W)
        ce_loss = super().forward(output, target)

        # (N, C, H, W)
        prob = F.softmax(output, dim=1)
        # (N, H, W)
        prob = torch.gather(prob, dim=1, index=target.unsqueeze(1)).squeeze(1)
        # (N, H, W)
        weight = torch.pow(1 - prob, self.gamma)

        # (N, H, W)
        focal_loss = weight * ce_loss

        if self.reduction == "mean":
            return torch.mean(focal_loss)
        elif self.reduction == "sum":
            return torch.sum(focal_loss)
        else:
            return focal_loss


class AdaptiveThresholdingLoss(nn.Module):
    """Adaptive Thresholding loss function in ATLOP
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target, pos_weight=1.0, neg_weight=1.0):
        """
        Parameters
        ----------
        output : torch.Tensor
            shape of (batch_size, n_labels)
        target : torch.Tensor
            shape of (batch_size, n_labels); binary
        pos_weight : float
            by default 1.0
        neg_weight : float
            by default 1.0

        Returns
        -------
        torch.Tensor
            shape of (batch_size,)
        """
        # output: (batch_size, n_labels)
        # target: (batch_size, n_labels); binary

        # Mask only for the threshold label
        # (batch_size, n_labels)
        th_target = torch.zeros_like(target, dtype=torch.float).to(target)
        th_target[:, 0] = 1.0
        # Mask for the positive labels
        target[:, 0] = 0.0
        # Mask for the positive and threshold labels
        p_and_th_mask = target + th_target
        # Mask for the negative and threshold labels
        n_and_th_mask = 1 - target

        # Rank positive labels to the threshold label
        # (batch_size, n_labels)
        p_and_th_output = output - (1 - p_and_th_mask) * 1e30
        # (batch_size,)
        loss1 = -(F.log_softmax(p_and_th_output, dim=-1) * target).sum(dim=1)

        # Rank negative labels to the threshold label
        # (batch_size, n_labels)
        n_and_th_output = output - (1 - n_and_th_mask) * 1e30
        # (batch_size,)
        loss2 = -(F.log_softmax(n_and_th_output, dim=-1) * th_target).sum(dim=1)

        # Sum two parts
        loss = pos_weight * loss1 + neg_weight * loss2
        return loss

    def get_labels(self, logits, top_k=-1):
        """
        Parameters
        ----------
        logits : torch.Tensor
            shape of (batch_size, n_labels)
        top_k : int, optional
            by default -1

        Returns
        -------
        torch.Tensor
            shape of (batch_size, n_labels)
        """
        # (batch_size, n_labels)
        labels = torch.zeros_like(logits).to(logits)
        # Identify labels l whose logits, Score(l|x),
        #   are higher than the threshold logit, Score(l=0|x)
        # (batch_size, 1)
        th_logits = logits[:, 0].unsqueeze(1)
        # (batch_size, n_labels)
        mask = (logits > th_logits)
        # Identify labels whose logits are higher
        #   than the minimum logit of the top-k labels
        if top_k > 0:
            # (batch_size, top_k)
            topk_logits, _ = torch.topk(logits, top_k, dim=1)
            # (batch_size, 1)
            topk_min_logits = topk_logits[:, -1].unsqueeze(1)
            # (batch_size, n_labels)
            mask = (logits >= topk_min_logits) & mask
        # Set 1 to the labels that meet the above conditions
        # (batch_size, n_labels)
        labels[mask] = 1.0
        # Set 1 to the thresholding labels if no relation holds
        # (batch_size, n_labels)
        labels[:, 0] = (labels.sum(dim=1) == 0.0).to(logits)
        return labels


###################################
# Optimizers
###################################


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
