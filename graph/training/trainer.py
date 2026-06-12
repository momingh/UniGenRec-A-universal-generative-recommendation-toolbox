from dataclasses import dataclass
from typing import Optional

import torch
from tqdm import tqdm


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
    ) -> TrainingState:
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(1, int(num_epochs) + 1):
            self.state.epoch = epoch
            train_loss = self.train_one_epoch(train_loader)
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}")

            if valid_data is not None:
                metrics = self.evaluate_full_sort(
                    valid_data,
                    split=valid_split,
                    batch_size=full_sort_batch_size,
                )
                print(f"Epoch {epoch}: valid_metrics={metrics}")

                if monitor_metric is not None:
                    metric_value = self._get_metric(metrics, monitor_metric)
                    if (
                        self.state.best_metric is None
                        or metric_value > self.state.best_metric
                    ):
                        self.state.best_epoch = epoch
                        self.state.best_metric = metric_value
                        best_state = self._clone_model_state()
                        epochs_without_improvement = 0
                        print(
                            f"Epoch {epoch}: best_{monitor_metric}={metric_value:.6f}"
                        )
                    else:
                        epochs_without_improvement += 1
                        if (
                            early_stop_patience is not None
                            and epochs_without_improvement >= int(early_stop_patience)
                        ):
                            print(
                                "Early stopping: "
                                f"{monitor_metric} did not improve for "
                                f"{early_stop_patience} epochs."
                            )
                            break

        if test_data is not None:
            if best_state is None:
                raise RuntimeError("No best validation model was selected.")
            self.model.load_state_dict(best_state)
            print(
                "Best validation model: "
                f"epoch={self.state.best_epoch}, "
                f"{monitor_metric}={self.state.best_metric:.6f}"
            )
            test_metrics = self.evaluate_full_sort(
                test_data,
                split=test_split,
                batch_size=full_sort_batch_size,
            )
            print(f"Test metrics on best validation model: {test_metrics}")

        return self.state

    def train_one_epoch(self, train_loader) -> float:
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training"), start=1):
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

    @staticmethod
    def _get_metric(metrics: dict[str, float], metric_name: str) -> float:
        if metric_name not in metrics:
            available = ", ".join(sorted(metrics))
            raise KeyError(
                f"Monitor metric not found: {metric_name}. Available: {available}"
            )
        return float(metrics[metric_name])
