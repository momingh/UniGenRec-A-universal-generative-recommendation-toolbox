import logging
import random
import sys
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config file is empty or invalid: {config_path}")
    return config


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def setup_logging(log_path: Path) -> None:
    class ColorFormatter(logging.Formatter):
        red = "\033[31m"
        blue = "\033[34m"
        reset = "\033[0m"

        def format(self, record):
            message = record.getMessage()
            color = None
            if "New best |" in message:
                color = self.red
            elif (
                message.startswith("Best Epoch:")
                or message.startswith("Best Validation")
                or message.startswith("Corresponding Test")
            ):
                color = self.blue

            if color is None:
                return super().format(record)

            original_msg = record.msg
            original_args = record.args
            try:
                record.msg = f"{color}{message}{self.reset}"
                record.args = ()
                return super().format(record)
            finally:
                record.msg = original_msg
                record.args = original_args

    ensure_dir(log_path.parent)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(ColorFormatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(template: str, project_root: Path, **format_args: Any) -> Path:
    formatted = template.format(**format_args)
    path = Path(formatted)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()
