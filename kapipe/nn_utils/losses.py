from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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

