import torch
import torch.nn as nn


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


