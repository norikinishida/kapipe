import torch
import torch.nn as nn
import torch.nn.functional as F


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



