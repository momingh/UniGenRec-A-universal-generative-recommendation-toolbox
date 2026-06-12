import logging
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: Optional[float] = None,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc="Training"):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = outputs["loss"]
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(train_loader), 1)


def evaluate(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    topk_list: List[int],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    metric_sums: Dict[str, float] = {}
    total_count = 0.0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            batch = move_batch_to_device(batch, device)
            batch_metrics = model.evaluate_step(batch=batch, topk_list=topk_list)
            count = float(batch_metrics.pop("count", 0.0))
            if count <= 0:
                logging.warning("model.evaluate_step() returned an empty count; skipping batch.")
                continue
            total_count += count
            for metric, value in batch_metrics.items():
                metric_sums[metric] = metric_sums.get(metric, 0.0) + float(value)

    return {
        metric: value / total_count if total_count > 0 else 0.0
        for metric, value in metric_sums.items()
    }
