from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.init as init


def make_embedding(
    dict_size: int,
    dim: int,
    std: float = 0.02
) -> nn.Embedding:
    """Factory for a normally initialized embedding layer."""

    # Create an embedding layer
    emb = nn.Embedding(dict_size, dim)

    # Initialize the embedding weights with a normal distribution
    init.normal_(emb.weight, std=std)

    return emb


def make_linear(
    input_dim: int,
    output_dim: int,
    bias: bool = True,
    std: float = 0.02,
) -> nn.Linear:
    """Factory for a linear layer."""
    # Create a linear layer using PyTorch's default initialization
    linear = nn.Linear(
        in_features=input_dim,
        out_features=output_dim,
        bias=bias,
    )

    # init.normal_(linear.weight, std=std)
    # if bias:
    #     init.zeros_(linear.bias)

    return linear


def make_mlp(
    input_dim: int,
    hidden_dims: int | Iterable[int] | None,
    output_dim: int,
    dropout_rate: float,
) -> nn.Linear | nn.Sequential:
    """Factory for a multilayer perceptron."""

    # Return a single linear layer when no hidden layer is requested
    if (
        (hidden_dims is None)
        or (hidden_dims == 0)
        or (hidden_dims == [])
        or (hidden_dims == [0])
    ):
        return nn.Linear(input_dim, output_dim)

    # Convert a single hidden dimension into a list
    if not isinstance(hidden_dims, Iterable):
        hidden_dims = [hidden_dims]

    # Create the first hidden layer
    layers: list[nn.Module] = [
        nn.Linear(input_dim, hidden_dims[0]),
        nn.ReLU(),
        nn.Dropout(p=dropout_rate),
    ]

    # Create the remaining hidden layers
    for i in range(1, len(hidden_dims)):
        # Add the linear transformation, activation, and dropout
        layers.extend(
            [
                nn.Linear(hidden_dims[i - 1], hidden_dims[i]),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
            ]
        )

    # Add the output layer without an activation function
    layers.append(
        nn.Linear(hidden_dims[-1], output_dim)
    )

    # Combine the layers into one module
    return nn.Sequential(*layers)


def make_mlp_hidden(
    input_dim: int,
    hidden_dim: int,
    dropout_rate: float,
) -> nn.Sequential:
    """Factory for one hidden MLP layer."""

    # Create the linear transformation, activation, and dropout
    layers: list[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=dropout_rate),
    ]

    # Combine the layers into one module
    return nn.Sequential(*layers)


class Biaffine(nn.Module):
    """Biaffine scoring layer."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        bias_x: bool = True,
        bias_y: bool = True,
    ) -> None:
        """Initializer for a biaffine scoring layer."""

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias_x = bias_x
        self.bias_y = bias_y

        # Compute the dimensions after adding optional bias terms
        input_dim_x = input_dim + int(bias_x)
        input_dim_y = input_dim + int(bias_y)

        # Create the trainable biaffine weight tensor
        self.weight = nn.Parameter(
            torch.Tensor(
                output_dim,
                input_dim_x,
                input_dim_y,
            )
        )

        # Initialize the trainable parameters
        self.reset_parameters()

    def __repr__(self) -> str:
        """String representation of the layer configuration."""

        fields = [
            f"input_dim={self.input_dim}",
            f"output_dim={self.output_dim}",
        ]

        if self.bias_x:
            fields.append(f"bias_x={self.bias_x}")
        if self.bias_y:
            fields.append(f"bias_y={self.bias_y}")
        
        configuration = ", ".join(fields)

        return f"{self.__class__.__name__}({configuration})"

    def reset_parameters(self) -> None:
        """Parameter initialization with a normal distribution."""

        # Initialize the biaffine weights
        init.normal_(self.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Biaffine scores for every pair of input vectors."""

        # Append a constant feature to the first input when requested
        if self.bias_x:
            # (batch_size, seq_len, input_dim+1)
            x = torch.cat(
                tensors=(
                    x,
                    torch.ones_like(x[..., :1]),
                ),
                dim=-1,
            )

        # Append a constant feature to the second input when requested
        if self.bias_y:
            # (batch_size, seq_len, input_dim+1)
            y = torch.cat(
                tensors=(
                    y,
                    torch.ones_like(y[..., :1]),
                ),
                dim=-1,
            )


        # Compute scores for every pair of sequence positions
        # (batch_size, output_dim, seq_len, seq_len)
        scores = torch.einsum(
            "bxi,oij,byj->boxy",
            x,
            self.weight,
            y,
        )

        return scores


def make_transformer_encoder(
    input_dim: int,
    n_heads: int,
    ffnn_dim: int,
    dropout_rate: float,
    n_layers: int,
) -> nn.TransformerEncoder:
    """Factory for a Transformer encoder."""

    # Create the shared Transformer encoder layer configuration
    transformer_encoder_layer = nn.TransformerEncoderLayer(
        d_model=input_dim,
        nhead=n_heads,
        dim_feedforward=ffnn_dim,
        dropout=dropout_rate,
    )

    # Stack independent copies of the encoder layer
    transformer_encoder = nn.TransformerEncoder(
        encoder_layer=transformer_encoder_layer,
        num_layers=n_layers,
    )

    return transformer_encoder

