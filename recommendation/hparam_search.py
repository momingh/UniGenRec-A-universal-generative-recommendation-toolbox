import argparse
import csv
import gc
import itertools
import json
import logging
import os
import pprint
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from transformers.optimization import get_scheduler


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(ROOT)

from dataset import GenRecDataset, item2code
from tokenizer import get_tokenizer
from trainer import train_one_epoch, evaluate
from utils import (
    format_metric_value,
    format_metrics_line,
    get_model_class,
    load_and_process_config,
    set_seed,
    setup_logging,
)
from recommendation.models.generation.prefix_tree import build_trie_from_codebook


try:
    from tqdm.contrib.logging import logging_redirect_tqdm
except ImportError:
    logging_redirect_tqdm = nullcontext


# Minimal search space. Edit this dict when you want to change the search.
SEARCH_SPACE = {
    "training_params.lr": [0.001, 0.003, 0.01],
    "training_params.batch_size": [64, 128, 256, 512],
    "model_params.temperature": [0.03, 0.05,0.07],
    "model_params.attn_pdrop": [0.3, 0.5, 0.7],
}


def get_total_steps(training_params: Dict[str, Any], train_loader: DataLoader) -> int:
    total_steps = training_params.get("steps")
    if total_steps is None:
        total_steps = training_params.get("total_steps")
    if total_steps is not None:
        return int(total_steps)

    num_epochs = training_params.get("epochs")
    if num_epochs is None:
        num_epochs = training_params.get("num_epochs")
    if num_epochs is None:
        raise ValueError("training_params must define either steps/total_steps or epochs/num_epochs")

    return len(train_loader) * int(num_epochs)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    training_params: Dict[str, Any],
):
    scheduler_name = training_params.get("scheduler")
    if scheduler_name is None:
        scheduler_name = training_params.get("lr_scheduler")
    if scheduler_name is None:
        return None

    scheduler_name = str(scheduler_name).lower()
    if scheduler_name in {"none", "null", "false"}:
        return None

    scheduler_aliases = {
        "cosine": "cosine",
        "constant_with_warmup": "constant_with_warmup",
        "warmup_constant": "constant_with_warmup",
    }
    if scheduler_name not in scheduler_aliases:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    scheduler_name = scheduler_aliases[scheduler_name]

    warmup_steps = int(training_params.get("warmup_steps", 0) or 0)
    total_steps = get_total_steps(training_params, train_loader)
    if total_steps <= 0:
        return None

    logging.info(
        f"Using HuggingFace {scheduler_name} LR scheduler: warmup_steps={warmup_steps}, "
        f"total_steps={total_steps}"
    )
    return get_scheduler(
        name=scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def format_lr(optimizer: torch.optim.Optimizer) -> str:
    return f"{get_current_lr(optimizer):.8e}"


def set_nested(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    current = config
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value


def apply_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    for dotted_key, value in overrides.items():
        set_nested(config, dotted_key, value)


def reset_trial_seed(seed: int) -> None:
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_worker_init_fn(seed: int):
    def seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return seed_worker


def iter_grid(search_space: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(search_space.keys())
    values = [search_space[key] for key in keys]
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(data), f, ensure_ascii=False, indent=2)


def write_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_serializable(data), f, allow_unicode=True, sort_keys=False)


def flatten_result_for_csv(result: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "trial": result.get("trial"),
        "trial_name": result.get("trial_name"),
        "seed": result.get("seed"),
        "metric": result.get("metric"),
        "best_metric": result.get("best_metric"),
        "best_epoch": result.get("best_epoch"),
        "trial_dir": result.get("trial_dir"),
        "save_path": result.get("save_path"),
    }

    for key, value in result.get("params", {}).items():
        row[f"param.{key}"] = value
    for key, value in (result.get("best_val_results") or {}).items():
        row[f"val.{key}"] = value
    for key, value in (result.get("best_test_results") or {}).items():
        row[f"test.{key}"] = value
    return to_serializable(row)


def write_results_csv(path: Path, results: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [flatten_result_for_csv(result) for result in results]
    if not rows:
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "trial",
        "trial_name",
        "seed",
        "metric",
        "best_metric",
        "best_epoch",
        "trial_dir",
        "save_path",
    ]
    fieldnames = [key for key in preferred if key in fieldnames] + [
        key for key in fieldnames if key not in preferred
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_trial_config(args, trial_name: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    config = load_and_process_config(
        args.model,
        args.dataset,
        args.quant_method,
        embedding_modality=args.embedding_modality,
    )
    apply_overrides(config, overrides)

    base_output_dir = Path(config["save_path"]).parent
    trial_dir = base_output_dir / "search" / args.search_name / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)

    config["log_path"] = trial_dir / "training.log"
    config["save_path"] = trial_dir / "best_model.pth"
    config["trial_name"] = trial_name
    config["trial_params"] = dict(overrides)
    return config


def run_trial(args, trial_index: int, overrides: Dict[str, Any]) -> Dict[str, Any]:
    trial_name = f"trial_{trial_index:03d}"
    logging.getLogger().handlers.clear()
    config = prepare_trial_config(args, trial_name, overrides)

    trial_seed = int(config["training_params"].get("seed", 2023))
    reset_trial_seed(trial_seed)

    log_path = Path(config["log_path"])
    if log_path.exists():
        log_path.unlink()
    setup_logging(log_path)

    logging.info(f"Starting {trial_name}")
    logging.info(f"Trial seed reset to: {trial_seed}")
    logging.info(f"Trial overrides: {pprint.pformat(overrides)}")
    logging.info("=" * 50)
    logging.info("\n" + pprint.pformat(config))
    logging.info("=" * 50)

    write_yaml(Path(config["save_path"]).parent / "config.yaml", config)

    try:
        device = torch.device(config["training_params"]["device"] if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        num_workers = config["training_params"].get("num_workers", 4)

        logging.info("Loading item to code mapping...")
        item_to_code_map, _ = item2code(
            config["code_path"],
            config["vocab_sizes"],
            config["bases"],
        )
        logging.info(f"Item to code map loaded. Total items mapped: {len(item_to_code_map)}")

        prefix_trie = None
        use_trie = config.get("evaluation_params", {}).get("use_prefix_trie", False)
        if use_trie:
            logging.info("Building Prefix Trie (enabled in config)...")
            prefix_trie = build_trie_from_codebook(
                token_sequences=list(item_to_code_map.values()),
                eos_token_id=config["token_params"]["eos_token_id"],
            )
        else:
            logging.info("Prefix Trie is DISABLED (default or as per config).")

        logging.info(f"Dynamically loading model: {args.model}")
        ModelClass = get_model_class(args.model)
        model = ModelClass(config, prefix_trie=prefix_trie, item_to_code_map=item_to_code_map)
        model.to(device)
        logging.info(model.n_parameters)
        logging.info("=" * 50)

        weight_decay = float(config["training_params"].get("weight_decay", 0.01))
        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(config["training_params"]["lr"]),
            weight_decay=weight_decay,
        )

        logging.info(f"Initializing tokenizer for model: {args.model}")
        collate_fn = get_tokenizer(
            model_name=args.model,
            config=config,
            item_to_code_map=item_to_code_map,
        )
        logging.info("Tokenizer initialized.")

        logging.info("Creating Datasets...")
        train_dataset = GenRecDataset(config=config, mode="train")
        validation_dataset = GenRecDataset(config=config, mode="valid")
        test_dataset = GenRecDataset(config=config, mode="test")

        logging.info("Creating DataLoaders...")
        is_gpu_training = torch.cuda.is_available() and num_workers > 0
        train_generator = torch.Generator()
        train_generator.manual_seed(trial_seed)

        loader_kwargs = {
            "num_workers": num_workers,
            "collate_fn": collate_fn,
            "pin_memory": is_gpu_training,
            "persistent_workers": is_gpu_training if num_workers > 0 else False,
            "worker_init_fn": make_worker_init_fn(trial_seed),
        }
        eval_loader_kwargs = {
            **loader_kwargs,
            "batch_size": config["evaluation_params"]["batch_size"],
            "shuffle": False,
        }

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training_params"]["batch_size"],
            shuffle=True,
            generator=train_generator,
            **loader_kwargs,
        )
        validation_loader = DataLoader(validation_dataset, **eval_loader_kwargs)
        test_loader = DataLoader(test_dataset, **eval_loader_kwargs)

        topk_list = config["evaluation_params"]["topk_list"]
        save_path = config["save_path"]

        def run_eval(label: str, loader: DataLoader) -> Dict[str, float]:
            results = evaluate(model, loader, topk_list, device)
            logging.info(format_metrics_line(label, results))
            return results

        scheduler = build_lr_scheduler(optimizer, train_loader, config["training_params"])
        max_grad_norm = config["training_params"].get("max_grad_norm")
        if max_grad_norm is not None:
            max_grad_norm = float(max_grad_norm)
            logging.info(f"Gradient clipping enabled: max_grad_norm={max_grad_norm}")

        best_metric = -float("inf")
        best_epoch = 0
        best_val_results = None
        best_test_results = None
        early_stop_counter = 0

        eval_interval = config["training_params"].get("eval_interval", 1)
        num_epochs = config["training_params"]["num_epochs"]
        metric_name = args.metric
        early_stop_limit = config["training_params"]["early_stop"] * eval_interval
        epoch_separator = "-" * 80

        logging.info(f"Evaluation interval set to: {eval_interval} epoch(s)")
        logging.info(f"Search metric set to: {metric_name}")

        with logging_redirect_tqdm():
            for epoch in range(num_epochs):
                epoch_num = epoch + 1
                epoch_start = time.perf_counter()
                logging.info(epoch_separator)
                logging.info(f"Epoch {epoch_num:03d}/{num_epochs:03d} started | lr={format_lr(optimizer)}")

                train_loss = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    scheduler=scheduler,
                    max_grad_norm=max_grad_norm,
                )

                if epoch_num % eval_interval == 0:
                    logging.info(f"Evaluating epoch {epoch_num:03d} | lr={format_lr(optimizer)}")
                    val_results = run_eval("Validation", validation_loader)
                    current_metric = float(val_results.get(metric_name, 0.0))

                    if current_metric > best_metric:
                        best_metric = current_metric
                        early_stop_counter = 0

                        test_results = run_eval("Test", test_loader)
                        best_epoch = epoch_num
                        best_val_results = val_results
                        best_test_results = test_results

                        torch.save(model.state_dict(), save_path)
                        logging.info(
                            f"New best | epoch={epoch_num:03d} | "
                            f"{metric_name}={format_metric_value(best_metric)} | "
                            f"saved={save_path}"
                        )
                    else:
                        early_stop_counter += eval_interval
                        logging.info(
                            f"No improvement | best_epoch={best_epoch:03d} | "
                            f"early_stop={early_stop_counter}/{early_stop_limit}"
                        )

                    epoch_time = time.perf_counter() - epoch_start
                    logging.info(
                        f"Epoch {epoch_num:03d}/{num_epochs:03d} | "
                        f"loss={train_loss:.8f} | "
                        f"lr={format_lr(optimizer)} | "
                        f"val {metric_name}={format_metric_value(current_metric)} | "
                        f"best={format_metric_value(best_metric)} | "
                        f"time={epoch_time:.2f}s"
                    )

                    if early_stop_counter >= early_stop_limit:
                        logging.info("Early stopping triggered.")
                        break
                else:
                    epoch_time = time.perf_counter() - epoch_start
                    logging.info(
                        f"Epoch {epoch_num:03d}/{num_epochs:03d} | "
                        f"loss={train_loss:.8f} | "
                        f"lr={format_lr(optimizer)} | "
                        f"eval=skipped | "
                        f"best={format_metric_value(best_metric)} | "
                        f"time={epoch_time:.2f}s"
                    )

        result = {
            "trial": trial_index,
            "trial_name": trial_name,
            "seed": trial_seed,
            "params": dict(overrides),
            "metric": metric_name,
            "best_epoch": best_epoch,
            "best_metric": None if best_metric == -float("inf") else best_metric,
            "best_val_results": best_val_results,
            "best_test_results": best_test_results,
            "trial_dir": str(Path(save_path).parent),
            "save_path": str(save_path),
        }

        write_json(Path(save_path).parent / "result.json", result)

        logging.info("=" * 50)
        logging.info(f"{trial_name} finished.")
        logging.info(f"Best Epoch: {best_epoch:03d}")
        logging.info(f"Best {metric_name}: {format_metric_value(result['best_metric'])}")
        if best_val_results:
            logging.info(format_metrics_line("Best Validation", best_val_results))
        if best_test_results:
            logging.info(format_metrics_line("Corresponding Test", best_test_results))
        logging.info("=" * 50)
        return result
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal hyperparameter search for GenRec.")
    parser.add_argument("--model", type=str, default="RPG", help="Model name, e.g. RPG, TIGER, GPT2")
    parser.add_argument("--dataset", type=str, default="amazon-musical-instruments-23", help="Dataset name")
    parser.add_argument(
        "--quant_method",
        type=str,
        default="rqvae",
        choices=["rqvae", "rqvae_faiss", "opq", "qinco", "qinco_aux", "qinco_v2", "rqkmeans", "rqkmeans_plus"],
        help="Quantization method",
    )
    parser.add_argument(
        "--embedding_modality",
        type=str,
        default="text",
        help="Codebook modality, e.g. text or graph-lightgcn",
    )
    parser.add_argument("--search_name", type=str, default="simple_search", help="Name for this search run")
    parser.add_argument("--metric", type=str, default="NDCG@10", help="Validation metric used to pick best trial")
    parser.add_argument("--max_trials", type=int, default=None, help="Optional limit for quick checks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []

    all_trials = list(iter_grid(SEARCH_SPACE))
    if args.max_trials is not None:
        all_trials = all_trials[: args.max_trials]

    print(f"Total trials: {len(all_trials)}")
    print(f"Search space: {pprint.pformat(SEARCH_SPACE)}")

    search_root = None
    for trial_index, overrides in enumerate(all_trials):
        print(f"\n===== Running trial_{trial_index:03d}: {overrides} =====")
        result = run_trial(args, trial_index, overrides)
        results.append(result)

        trial_dir = Path(result["trial_dir"])
        search_root = trial_dir.parent
        write_json(search_root / "search_results.json", results)
        write_results_csv(search_root / "search_results.csv", results)

    if not results:
        print("No trials were run.")
        return

    best_result = max(
        results,
        key=lambda item: float("-inf") if item["best_metric"] is None else float(item["best_metric"]),
    )

    if search_root is not None:
        write_json(search_root / "best_trial.json", best_result)
        write_results_csv(search_root / "search_results.csv", results)

    print("\n===== Search finished =====")
    print(f"Best trial: {best_result['trial_name']}")
    print(f"Best {args.metric}: {best_result['best_metric']}")
    print(f"Best epoch: {best_result['best_epoch']}")
    print(f"Params: {pprint.pformat(best_result['params'])}")
    print(f"Trial dir: {best_result['trial_dir']}")
    if best_result.get("best_test_results"):
        print(f"Test results: {pprint.pformat(best_result['best_test_results'])}")


if __name__ == "__main__":
    main()
