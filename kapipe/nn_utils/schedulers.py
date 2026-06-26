from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import get_linear_schedule_with_warmup


def get_scheduler(
    optimizers: list[Optimizer],
    total_update_steps: int,
    warmup_steps: int,
) -> list[LambdaLR]:
    """Return a list of schedulers for the optimizers, with separate learning-rate schedules for BERT and task-specific parameters."""

    def lr_lambda_bert(current_step: int) -> float:
        """Return the learning-rate multiplier for the BERT optimizer."""

        # Increase the learning rate linearly during warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # Decrease the learning rate linearly after warmup
        return max(
            0.0,
            float(total_update_steps - current_step)
            / float(
                max(
                    1,
                    total_update_steps - warmup_steps
                )
            ),
        )

    def lr_lambda_task(current_step: int) -> float:
        """Return the learning-rate multiplier for the task optimizer."""

        # Decrease the learning rate linearly from the initial value
        return max(
            0.0,
            float(total_update_steps - current_step)
            / float(
                max(
                    1,
                    total_update_steps
                )
            ),
        )

    # Create separate schedulers for the BERT and task optimizers
    schedulers = [
        # Apply warmup and linear decay to the BERT optimizer
        LambdaLR(optimizers[0], lr_lambda_bert),
        # Apply linear decay to the task optimizer
        LambdaLR(optimizers[1], lr_lambda_task),
    ]

    return schedulers


def get_scheduler2(
    optimizer: Optimizer,
    total_update_steps: int,
    warmup_steps: int,
) -> LambdaLR:
    """Return a linear scheduler with warmup for the given optimizer."""

    # Create a scheduler with linear warmup and linear decay
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps
    )
