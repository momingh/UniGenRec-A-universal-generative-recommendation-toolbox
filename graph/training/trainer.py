from dataclasses import dataclass
import logging
import time
from collections.abc import Mapping
from contextlib import nullcontext
from numbers import Real
from typing import Optional

import torch
from tqdm import tqdm

try:
    from tqdm.contrib.logging import logging_redirect_tqdm
except ImportError:
    logging_redirect_tqdm = nullcontext

TQDM_KWARGS = {"dynamic_ncols": True, "leave": False}


@dataclass
class TrainingState:
    epoch: int = 0
    global_step: int = 0
    best_epoch: int = 0
    best_metric: Optional[float] = None


class GraphTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler=None,
        max_train_batches: Optional[int] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.max_train_batches = max_train_batches
        self.state = TrainingState()

    def fit(
        self,
        train_loader,
        num_epochs: int,
        valid_data=None,
        valid_split: str = "valid",
        test_data=None,
        test_split: str = "test",
        full_sort_batch_size: Optional[int] = None,
        monitor_metric: Optional[str] = None,
        early_stop_patience: Optional[int] = None,
        eval_interval: int = 1,
    ) -> TrainingState:
        eval_interval = int(eval_interval)
        if eval_interval <= 0:
            raise ValueError(f"eval_interval must be positive, got {eval_interval}")

        best_state = None
        best_metrics = None
        best_test_metrics = None
        epochs_without_improvement = 0
        epoch_separator = "-" * 80
        num_epochs = int(num_epochs)

        with logging_redirect_tqdm():
            for epoch in range(1, num_epochs + 1):
                self.state.epoch = epoch
                epoch_start = time.perf_counter()
                logging.info(epoch_separator)
                logging.info(
                    f"Epoch {epoch:03d}/{num_epochs:03d} started | "
                    f"lr={self._format_lr()}"
                )
                train_loss = self.train_one_epoch(train_loader)

                should_eval = (
                    valid_data is not None
                    and (epoch % eval_interval == 0 or epoch == num_epochs)
                )

                if should_eval:
                    logging.info(
                        f"Evaluating epoch {epoch:03d} | lr={self._format_lr()}"
                    )
                    metrics = self.evaluate_full_sort(
                        valid_data,
                        split=valid_split,
                        batch_size=full_sort_batch_size,
                    )
                    logging.info(format_metrics_line("Validation", metrics))

                    test_metrics = None
                    if test_data is not None:
                        test_metrics = self.evaluate_full_sort(
                            test_data,
                            split=test_split,
                            batch_size=full_sort_batch_size,
                        )
                        logging.info(format_metrics_line("Test", test_metrics))

                    if monitor_metric is not None:
                        metric_value = self._get_metric(metrics, monitor_metric)
                        if (
                            self.state.best_metric is None
                            or metric_value > self.state.best_metric
                        ):
                            self.state.best_epoch = epoch
                            self.state.best_metric = metric_value
                            best_state = self._clone_model_state()
                            best_metrics = metrics
                            best_test_metrics = test_metrics
                            epochs_without_improvement = 0
                            logging.info(
                                f"New best | epoch={epoch:03d} | "
                                f"{monitor_metric}={format_metric_value(metric_value)}"
                            )
                        else:
                            epochs_without_improvement += 1
                            logging.info(
                                f"No improvement | "
                                f"best_epoch={self.state.best_epoch:03d} | "
                                f"early_stop_checks={epochs_without_improvement}/"
                                f"{early_stop_patience}"
                            )

                    epoch_time = time.perf_counter() - epoch_start
                    best_metric = format_metric_value(self.state.best_metric)
                    current_metric = (
                        format_metric_value(self._get_metric(metrics, monitor_metric))
                        if monitor_metric is not None
                        else "N/A"
                    )
                    current_test_metric = (
                        format_metric_value(
                            self._get_metric(test_metrics, monitor_metric)
                        )
                        if monitor_metric is not None and test_metrics is not None
                        else "N/A"
                    )
                    logging.info(
                        f"Epoch {epoch:03d}/{num_epochs:03d} | "
                        f"loss={train_loss:.8f} | "
                        f"lr={self._format_lr()} | "
                        f"val {monitor_metric}={current_metric} | "
                        f"test {monitor_metric}={current_test_metric} | "
                        f"best={best_metric} | "
                        f"time={epoch_time:.2f}s"
                    )

                    if (
                        monitor_metric is not None
                        and early_stop_patience is not None
                        and epochs_without_improvement >= int(early_stop_patience)
                    ):
                        logging.info("Early stopping triggered.")
                        break
                else:
                    epoch_time = time.perf_counter() - epoch_start
                    eval_status = (
                        "not_configured"
                        if valid_data is None
                        else f"skipped_next_interval={eval_interval}"
                    )
                    logging.info(
                        f"Epoch {epoch:03d}/{num_epochs:03d} | "
                        f"loss={train_loss:.8f} | "
                        f"lr={self._format_lr()} | "
                        f"eval={eval_status} | "
                        f"time={epoch_time:.2f}s"
                    )

        if test_data is not None:
            if best_state is None:
                raise RuntimeError("No best validation model was selected.")
            self.model.load_state_dict(best_state)
            if best_test_metrics is None:
                with logging_redirect_tqdm():
                    best_test_metrics = self.evaluate_full_sort(
                        test_data,
                        split=test_split,
                        batch_size=full_sort_batch_size,
                    )
            logging.info("=" * 50)
            logging.info("Training finished.")
            logging.info(f"Best Epoch: {self.state.best_epoch:03d}")
            logging.info(format_metrics_line("Best Validation", best_metrics))
            logging.info(
                format_metrics_line(
                    "Corresponding Test",
                    best_test_metrics,
                )
            )
            logging.info("=" * 50)
        elif best_state is not None:
            self.model.load_state_dict(best_state)
            logging.info("=" * 50)
            logging.info("Training finished.")
            logging.info(f"Best Epoch: {self.state.best_epoch:03d}")
            logging.info(format_metrics_line("Best Validation", best_metrics))
            logging.info("=" * 50)
        else:
            logging.info("=" * 50)
            logging.info("Training finished.")
            logging.info("=" * 50)

        return self.state

    def train_one_epoch(self, train_loader) -> float:
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for batch_idx, batch in enumerate(
            tqdm(train_loader, desc="Training", **TQDM_KWARGS),
            start=1,
        ):
            batch = self._move_batch_to_device(batch)
            self.optimizer.zero_grad()

            outputs = self.model(batch)
            loss = self._extract_loss(outputs)
            loss.backward()

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            self.state.global_step += 1
            total_loss += float(loss.item())
            total_batches += 1

            if self.max_train_batches is not None and batch_idx >= self.max_train_batches:
                break

        return total_loss / max(1, total_batches)

    @torch.no_grad()
    def evaluate_full_sort(
        self,
        data,
        split: str,
        batch_size: Optional[int],
    ) -> dict[str, float]:
        if not hasattr(self.model, "evaluate_full_sort"):
            raise AttributeError("Model must implement evaluate_full_sort(...).")
        if batch_size is None:
            raise ValueError("full_sort_batch_size must be configured.")

        self.model.eval()
        data = self._move_batch_to_device(data)
        return self.model.evaluate_full_sort(
            data,
            split=split,
            batch_size=int(batch_size),
        )

    def _move_batch_to_device(self, batch):
        if hasattr(batch, "to"):
            return batch.to(self.device)
        if isinstance(batch, dict):
            return {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
        return batch

    @staticmethod
    def _extract_loss(outputs) -> torch.Tensor:
        if isinstance(outputs, dict):
            return outputs["loss"]
        if hasattr(outputs, "loss"):
            return outputs.loss
        if torch.is_tensor(outputs):
            return outputs
        raise TypeError("Model output must be a Tensor, dict with loss, or object.loss.")

    def _clone_model_state(self):
        return {
            name: value.detach().cpu().clone()
            for name, value in self.model.state_dict().items()
        }

    def _format_lr(self) -> str:
        if not self.optimizer.param_groups:
            return "0.00000000e+00"
        return f"{float(self.optimizer.param_groups[0].get('lr', 0.0)):.8e}"

    @staticmethod
    def _get_metric(metrics: dict[str, float], metric_name: str) -> float:
        if metric_name not in metrics:
            available = ", ".join(sorted(metrics))
            raise KeyError(
                f"Monitor metric not found: {metric_name}. Available: {available}"
            )
        return float(metrics[metric_name])


def format_metric_value(value, digits: int = 8) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, Real) and not isinstance(value, bool):
        return f"{float(value):.{digits}f}"
    return repr(value)


def _metric_sort_key(metric: str):
    metric_order = {
        "Recall": 0,
        "NDCG": 1,
        "Precision": 2,
        "Hit": 3,
        "MRR": 4,
    }
    if "@" not in metric:
        return (1, metric, 0, "")

    name, k_text = metric.rsplit("@", 1)
    try:
        k_value = int(k_text)
    except ValueError:
        k_value = 10**9
    return (0, k_value, metric_order.get(name, 99), name)


def _ordered_metric_items(metrics: Mapping):
    return sorted(metrics.items(), key=lambda item: _metric_sort_key(str(item[0])))


def format_metrics(metrics: Mapping | None, digits: int = 8) -> str:
    if metrics is None:
        return "None"

    return " | ".join(
        f"{key}={format_metric_value(value, digits)}"
        for key, value in _ordered_metric_items(metrics)
    )


def format_metrics_line(
    label: str,
    metrics: Mapping | None,
    digits: int = 8,
    label_width: int = 18,
) -> str:
    return f"{label:<{label_width}} | {format_metrics(metrics, digits)}"
