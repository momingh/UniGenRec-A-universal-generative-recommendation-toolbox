import logging
from typing import Dict, List

import torch
from tqdm import tqdm


TQDM_KWARGS = {"dynamic_ncols": True, "leave": False}


def _move_to_device(batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _get_loss(outputs):
    return outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else outputs["loss"]


def train_one_epoch(model, train_loader, optimizer, device, scheduler=None, max_grad_norm=None):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc="Training", **TQDM_KWARGS):
        batch = _move_to_device(batch, device)
        optimizer.zero_grad()

        loss = _get_loss(model.forward(batch))
        loss.backward()

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / max(len(train_loader), 1)


def _batch_count(batch: Dict[str, torch.Tensor], batch_metrics: Dict[str, float]) -> float:
    count = float(batch_metrics.pop("count", 0))
    if count:
        return count

    # Fallback keeps evaluation usable if a model forgets to return count.
    inferred = float(batch.get("input_ids", torch.empty(0)).shape[0])
    if inferred:
        logging.warning("model.evaluate_step() did not return 'count'. Inferring from batch size.")
    return inferred


def evaluate(model, eval_loader, topk_list: List[int], device) -> Dict[str, float]:
    model.eval()
    total_metrics: Dict[str, float] = {}
    metric_denoms: Dict[str, float] = {}
    total_count = 0.0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating", **TQDM_KWARGS):
            batch = _move_to_device(batch, device)
            batch_metrics = model.evaluate_step(batch=batch, topk_list=topk_list)

            # evaluate_step returns metric sums for the batch; aggregate first, divide once.
            total_count += _batch_count(batch, batch_metrics)
            for metric, value in batch_metrics.items():
                if metric.startswith("_valid_"):
                    base_metric = metric[len("_valid_"):]
                    metric_denoms[base_metric] = metric_denoms.get(base_metric, 0.0) + float(value)
                    continue

                total_metrics[metric] = total_metrics.get(metric, 0.0) + float(value)
                if metric.startswith(("Recall@", "NDCG@")):
                    metric_denoms[metric] = total_count
                else:
                    metric_denoms.setdefault(metric, total_count)

    avg_metrics = {}
    for metric, value in total_metrics.items():
        denom = metric_denoms.get(metric, total_count)
        avg_metrics[metric] = value / denom if denom > 0 else 0.0
    return avg_metrics
