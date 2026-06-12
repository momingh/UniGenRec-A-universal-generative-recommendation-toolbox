import logging
import math
from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_lr_scheduler(
    optimizer: Optimizer,
    scheduler_name: Optional[str] = "cosine",
    warmup_steps: int = 0,
    total_steps: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
    num_epochs: Optional[int] = None,
):
    if scheduler_name is None:
        return None

    scheduler_name = str(scheduler_name).lower()
    if scheduler_name in {"none", "null", "false"}:
        return None
    if scheduler_name != "cosine":
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    if total_steps is None:
        if steps_per_epoch is None or num_epochs is None:
            raise ValueError(
                "total_steps or both steps_per_epoch and num_epochs must be provided."
            )
        total_steps = int(steps_per_epoch) * int(num_epochs)

    total_steps = int(total_steps)
    warmup_steps = int(warmup_steps or 0)
    if total_steps <= 0:
        return None

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    logging.info(
        "Using cosine LR scheduler: warmup_steps=%d, total_steps=%d",
        warmup_steps,
        total_steps,
    )
    return LambdaLR(optimizer, lr_lambda)
