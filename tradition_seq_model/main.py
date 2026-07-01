import argparse
import logging
import pprint
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import (
    SASRecEvalCollator,
    SASRecEvalDataset,
    SASRecTrainCollator,
    SASRecTrainDataset,
    SEQUENCE_AUTOREGRESSIVE,
    infer_num_items,
    load_sasrec_examples,
)
from models import SASRec
from trainer import evaluate, train_one_epoch
from utils import (
    ensure_dir,
    format_metric_value,
    format_metrics_line,
    load_config,
    resolve_path,
    set_seed,
    setup_logging,
)

try:
    from tqdm.contrib.logging import logging_redirect_tqdm
except ImportError:
    logging_redirect_tqdm = nullcontext


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train SASRec under tradition_seq_model.")
    parser.add_argument("--config", type=Path, default=script_dir / "configs" / "sasrec.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    return parser.parse_args()


def build_dataloaders(config, dataset_dir: Path, dataset_name: str, num_items: int):
    model_params = config["model_params"]
    training_params = config["training_params"]
    evaluation_params = config["evaluation_params"]
    max_len = int(model_params["max_len"])
    data_format = str(config.get("data_format", SEQUENCE_AUTOREGRESSIVE)).lower()

    inter_json = dataset_dir / f"{dataset_name}.inter.json"

    logging.info("Creating Datasets...")
    logging.info("Loading SASRec data from %s with data_format=%s", inter_json, data_format)
    train_examples, valid_examples, test_examples, seen_items_by_user = (
        load_sasrec_examples(inter_json, max_len=max_len, data_format=data_format)
    )

    logging.info(
        "Loaded data: users=%d, train_examples=%d, valid_examples=%d, test_examples=%d",
        len(seen_items_by_user),
        len(train_examples),
        len(valid_examples),
        len(test_examples),
    )

    train_workers = int(training_params.get("num_workers", 0) or 0)
    eval_workers = int(evaluation_params.get("num_workers", train_workers) or 0)

    logging.info("Creating DataLoaders...")
    train_loader = DataLoader(
        SASRecTrainDataset(train_examples),
        batch_size=int(training_params["batch_size"]),
        shuffle=True,
        num_workers=train_workers,
        collate_fn=SASRecTrainCollator(max_len=max_len, num_items=num_items),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=train_workers > 0,
    )
    valid_loader = DataLoader(
        SASRecEvalDataset(valid_examples),
        batch_size=int(evaluation_params["batch_size"]),
        shuffle=False,
        num_workers=eval_workers,
        collate_fn=SASRecEvalCollator(max_len=max_len),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=eval_workers > 0,
    )
    test_loader = DataLoader(
        SASRecEvalDataset(test_examples),
        batch_size=int(evaluation_params["batch_size"]),
        shuffle=False,
        num_workers=eval_workers,
        collate_fn=SASRecEvalCollator(max_len=max_len),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=eval_workers > 0,
    )
    return train_loader, valid_loader, test_loader, seen_items_by_user


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def format_lr(optimizer: torch.optim.Optimizer) -> str:
    return f"{get_current_lr(optimizer):.8e}"


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    config_path = args.config if args.config.is_absolute() else (Path.cwd() / args.config)
    if not config_path.exists():
        config_path = script_dir / args.config
    config = load_config(config_path.resolve())

    dataset_name = args.dataset or config.get("dataset", "Beauty")
    if args.num_epochs is not None:
        config["training_params"]["num_epochs"] = args.num_epochs
    if args.device is not None:
        config["training_params"]["device"] = args.device

    dataset_dir = resolve_path(
        config["paths"]["dataset_root"],
        project_root=script_dir,
        dataset_name=dataset_name,
    )
    output_root = resolve_path(
        config["paths"]["output_root"],
        project_root=script_dir,
        dataset_name=dataset_name,
    )
    ensure_dir(output_root)
    log_path = output_root / "training.log"
    save_path = output_root / "best_model.pth"

    setup_logging(log_path)
    set_seed(int(config["training_params"].get("seed", 2023)))
    logging.info("Configuration loaded for SASRec on %s.", dataset_name)
    logging.info("=" * 50)
    logging.info("\n%s", pprint.pformat(config))
    logging.info("=" * 50)
    logging.info("Dataset directory: %s", dataset_dir)
    logging.info("Output directory: %s", output_root)
    logging.info("Full-sort evaluation with seen-history items filtered.")

    device_name = config["training_params"].get("device", "cuda:0")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    num_items = infer_num_items(dataset_dir, dataset_name)
    logging.info("Inferred item count: %d", num_items)

    train_loader, valid_loader, test_loader, _ = build_dataloaders(
        config=config,
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        num_items=num_items,
    )

    model = SASRec(config=config, num_items=num_items)
    model.to(device)
    logging.info(model.n_parameters)
    logging.info("=" * 50)

    topk_list = [int(k) for k in config["evaluation_params"]["topk_list"]]
    checkpoint_path = args.checkpoint or save_path

    def run_eval(label, loader):
        results = evaluate(model, loader, topk_list, device)
        logging.info(format_metrics_line(label, results))
        return results

    if args.eval_only:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        logging.info("[Eval-Only] Loaded checkpoint from %s", checkpoint_path)
        run_eval("[Eval-Only] Test", test_loader)
        return

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training_params"]["lr"]),
        weight_decay=float(config["training_params"].get("weight_decay", 0.0)),
    )

    max_grad_norm = config["training_params"].get("max_grad_norm")
    if max_grad_norm is not None:
        max_grad_norm = float(max_grad_norm)
        logging.info("Gradient clipping enabled: max_grad_norm=%s", max_grad_norm)

    best_ndcg = 0.0
    best_epoch = 0
    best_val_results = None
    best_test_results = None
    early_stop_counter = 0
    eval_interval = int(config["training_params"].get("eval_interval", 1))
    early_stop = int(config["training_params"].get("early_stop", 10))
    num_epochs = int(config["training_params"]["num_epochs"])
    best_metric_name = "NDCG@10"
    fallback_best_metric_name = f"NDCG@{topk_list[-1]}"
    early_stop_limit = early_stop * eval_interval
    epoch_separator = "-" * 80

    logging.info("Evaluation interval set to: %d epoch(s)", eval_interval)

    with logging_redirect_tqdm():
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.perf_counter()
            logging.info(epoch_separator)
            logging.info("Epoch %03d/%03d started | lr=%s", epoch, num_epochs, format_lr(optimizer))

            train_metrics = train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                device=device,
                max_grad_norm=max_grad_norm,
            )
            train_loss = train_metrics["loss"]

            if epoch % eval_interval == 0:
                logging.info("Evaluating epoch %03d | lr=%s", epoch, format_lr(optimizer))
                val_results = run_eval("Validation", valid_loader)
                current_ndcg = val_results.get(
                    best_metric_name,
                    val_results.get(fallback_best_metric_name, 0.0),
                )

                if current_ndcg > best_ndcg:
                    best_ndcg = current_ndcg
                    best_epoch = epoch
                    best_val_results = val_results
                    early_stop_counter = 0

                    best_test_results = run_eval("Test", test_loader)

                    torch.save(model.state_dict(), save_path)
                    logging.info(
                        "New best | epoch=%03d | %s=%s | saved=%s",
                        epoch,
                        best_metric_name,
                        format_metric_value(best_ndcg),
                        save_path,
                    )
                else:
                    early_stop_counter += eval_interval
                    logging.info(
                        "No improvement | best_epoch=%03d | early_stop=%d/%d",
                        best_epoch,
                        early_stop_counter,
                        early_stop_limit,
                    )

                epoch_time = time.perf_counter() - epoch_start
                logging.info(
                    (
                        "Epoch %03d/%03d | loss=%.8f | loss_pos=%.8f | loss_neg=%.8f "
                        "| lr=%s | val %s=%s | best=%s | time=%.2fs"
                    ),
                    epoch,
                    num_epochs,
                    train_loss,
                    train_metrics["positive_loss"],
                    train_metrics["negative_loss"],
                    format_lr(optimizer),
                    best_metric_name,
                    format_metric_value(current_ndcg),
                    format_metric_value(best_ndcg),
                    epoch_time,
                )

                if early_stop_counter >= early_stop_limit:
                    logging.info("Early stopping triggered.")
                    break
            else:
                epoch_time = time.perf_counter() - epoch_start
                logging.info(
                    (
                        "Epoch %03d/%03d | loss=%.8f | loss_pos=%.8f | loss_neg=%.8f "
                        "| lr=%s | eval=skipped | best=%s | time=%.2fs"
                    ),
                    epoch,
                    num_epochs,
                    train_loss,
                    train_metrics["positive_loss"],
                    train_metrics["negative_loss"],
                    format_lr(optimizer),
                    format_metric_value(best_ndcg),
                    epoch_time,
                )

    logging.info("=" * 50)
    logging.info("Training finished.")
    if best_test_results is not None:
        logging.info("Best Epoch: %03d", best_epoch)
        logging.info(format_metrics_line("Best Validation", best_val_results))
        logging.info(format_metrics_line("Corresponding Test", best_test_results))
    else:
        logging.info("No improvement was observed.")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()
