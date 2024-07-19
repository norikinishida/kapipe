from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.init as init


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


