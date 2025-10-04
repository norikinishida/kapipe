from __future__ import annotations

from torch.optim import Optimizer
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR


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
