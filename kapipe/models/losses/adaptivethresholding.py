import torch
import torch.nn as nn
import torch.nn.functional as F


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
