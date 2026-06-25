from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginalizedCrossEntropyLoss(nn.Module):
    """Marginalized cross-entropy loss for multi-positive classification."""

    def __init__(self, reduction: str = "none") -> None:
        """Initializer for marginalized cross-entropy loss."""

        super().__init__()

        self.reduction = reduction

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Function to compute loss values for classification scores and binary targets.

        Loss = sum_{i} L_{i}
        L_{i}
          = -log[ sum_{k} exp(y_{i,k} + m_{i,k}) / sum_{k} exp(y_{i,k}) ]
          = -(
              log[ sum_{k} exp(y_{i,k} + m_{i,k}) ]
              - log[ sum_{k} exp(y_{i,k}) ]
              )
          = log[sum_{k} exp(y_{i,k})] - log[sum_{k} exp(y_{i,k} + m_{i,k})]
        (batch_size,)
        """

        # output: (batch_size, n_labels)
        # target: (batch_size, n_labels); binary

        # Compute the log-sum-exp over all labels
        logsumexp_all = torch.logsumexp(output, dim=1)

        # Convert positive labels to zero and negative labels to negative infinity
        positive_mask = torch.log(target.to(dtype=torch.float)) # 1 -> 0; 0 -> -inf

        # Compute the log-sum-exp over positive labels
        logsumexp_positive = torch.logsumexp(
            output + positive_mask,
            dim=1,
        )

        # Compute the negative log marginal probability
        # (batch_size,)
        loss = logsumexp_all - logsumexp_positive

        # Pool the loss values according to the specified reduction method
        if self.reduction == "mean":
            return torch.mean(loss)
        if self.reduction == "sum":
            return torch.sum(loss)

        return loss


class FocalLoss(nn.CrossEntropyLoss):
    """Focal loss for classification."""

    def __init__(
        self,
        gamma: float,
        alpha: torch.Tensor | None = None,
        ignore_index: int = -100,
        reduction: str = "none",
    ) -> None:
        """Initializer for focal loss."""

        super().__init__(
            weight=alpha,
            ignore_index=ignore_index,
            reduction="none",
        )

        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Focal loss values for classification scores and targets."""

        # Replace ignored indices with zero
        # (N, H, W)
        target = target * (target != self.ignore_index).long()

        # Compute cross-entropy loss for each position
        # (N, H, W)
        cross_entropy_loss = super().forward(output, target)

        # Compute class probabilities
        # (N, C, H, W)
        probabilities = F.softmax(output, dim=1)

        # Select the probability assigned to each target class
        # (N, H, W)
        target_probabilities = torch.gather(
            probabilities,
            dim=1,
            index=target.unsqueeze(1),
        ).squeeze(1)

        # Down-weight positions that are already classified confidently
        # (N, H, W)
        focal_weight = torch.pow(1 - target_probabilities, self.gamma)

        # Apply the focal weight to the cross-entropy loss
        # (N, H, W)
        focal_loss = focal_weight * cross_entropy_loss

        # Pool the loss values according to the specified reduction method
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)

        return focal_loss


class AdaptiveThresholdingLoss(nn.Module):
    """Adaptive thresholding loss used by ATLOP."""

    def __init__(self) -> None:
        """Initializer for adaptive thresholding loss."""

        super().__init__()

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        pos_weight: float = 1.0,
        neg_weight: float = 1.0,
    ) -> torch.Tensor:
        """Loss values for adaptive multi-label classification."""

        # output: (batch_size, n_labels)
        # target: (batch_size, n_labels); binary

        # Create a mask for the threshold label
        # (batch_size, n_labels)
        th_target = torch.zeros_like(target, dtype=torch.float).to(target)

        # Activate the threshold label
        th_target[:, 0] = 1.0

        # Remove the threshold label from the positive-label mask
        target[:, 0] = 0.0

        # Create a mask for the positive and threshold labels
        p_and_th_mask = target + th_target

        # Create a mask for the negative and threshold labels
        n_and_th_mask = 1 - target

        # Suppress labels outside the positive-threshold comparison
        # (batch_size, n_labels)
        p_and_th_output = output - (1 - p_and_th_mask) * 1e30

        # Compute the loss for ranking positive labels above the threshold label
        # (batch_size,)
        loss1 = -(F.log_softmax(p_and_th_output, dim=-1) * target).sum(dim=1)

        # Suppress labels outside the negative-threshold comparison
        # (batch_size, n_labels)
        n_and_th_output = output - (1 - n_and_th_mask) * 1e30

        # Compute the loss for ranking the threshold label above negative labels
        # (batch_size,)
        loss2 = -(F.log_softmax(n_and_th_output, dim=-1) * th_target).sum(dim=1)

        # Combine the positive and negative losses
        loss = pos_weight * loss1 + neg_weight * loss2

        return loss

    def get_labels(
        self,
        logits: torch.Tensor,
        top_k: int = -1,
    ) -> torch.Tensor:
        """Convert logits into labels using an adaptive threshold."""

        # Create an empty label tensor
        # (batch_size, n_labels)
        labels = torch.zeros_like(logits).to(logits)

        # Extract the threshold logit.
        # (batch_size, 1)
        th_logits = logits[:, 0].unsqueeze(1)

        # Identify labels whose logits are higher than the threshold logit
        # (batch_size, n_labels)
        mask = (logits > th_logits)

        # Restrict predictions to the top-k labels when requested
        if top_k > 0:
            # Extract the top-k logits
            # (batch_size, top_k)
            topk_logits, _ = torch.topk(logits, top_k, dim=1)

            # Extract the minimum logit among the top-k logits
            # (batch_size, 1)
            topk_min_logits = topk_logits[:, -1].unsqueeze(1)

            # Retain labels satifying both thresholding conditions
            # (batch_size, n_labels)
            mask = (logits >= topk_min_logits) & mask

        # Activate labels satisfying the thresholding conditions
        # (batch_size, n_labels)
        labels[mask] = 1.0

        # Activate the threshold label when no other label is active
        # (batch_size, n_labels)
        labels[:, 0] = (labels.sum(dim=1) == 0.0).to(logits)

        return labels

