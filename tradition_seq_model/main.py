import argparse
import logging
import pprint
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import (
    SASRecEvalCollator,
    SASRecEvalDataset,
    SASRecTrainCollator,
    SASRecTrainDataset,
    expand_train_sequences,
    infer_num_items,
    load_eval_examples,
    restore_train_sequences,
)
from models import SASRec
from trainer import evaluate, train_one_epoch
from utils import ensure_dir, load_config, resolve_path, set_seed, setup_logging


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

    train_json = dataset_dir / f"{dataset_name}.train.jsonl"
    valid_json = dataset_dir / f"{dataset_name}.valid.jsonl"
    test_json = dataset_dir / f"{dataset_name}.test.jsonl"

    logging.info("Restoring train sequences from %s", train_json)
    user_sequences, seen_items_by_user = restore_train_sequences(train_json)
    train_examples = expand_train_sequences(user_sequences, max_len=max_len)
    valid_examples = load_eval_examples(valid_json, max_len=max_len)
    test_examples = load_eval_examples(test_json, max_len=max_len)

    logging.info(
        "Loaded data: users=%d, train_examples=%d, valid_examples=%d, test_examples=%d",
        len(seen_items_by_user),
        len(train_examples),
        len(valid_examples),
        len(test_examples),
    )

    train_workers = int(training_params.get("num_workers", 0) or 0)
    eval_workers = int(evaluation_params.get("num_workers", train_workers) or 0)

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
    logging.info("Configuration:\n%s", pprint.pformat(config))
    logging.info("Dataset directory: %s", dataset_dir)
    logging.info("Output directory: %s", output_root)
    logging.info("Full-sort evaluation over all items WITHOUT filtering seen items.")

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

    topk_list = [int(k) for k in config["evaluation_params"]["topk_list"]]
    checkpoint_path = args.checkpoint or save_path

    if args.eval_only:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        logging.info("[Eval-Only] Loaded checkpoint from %s", checkpoint_path)
        test_results = evaluate(model, test_loader, topk_list, device)
        logging.info("[Eval-Only] Test Results: %s", test_results)
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

    best_ndcg = -1.0
    best_epoch = 0
    best_val_results = None
    best_test_results = None
    early_stop_counter = 0
    eval_interval = int(config["training_params"].get("eval_interval", 1))
    early_stop = int(config["training_params"].get("early_stop", 10))
    num_epochs = int(config["training_params"]["num_epochs"])

    for epoch in range(1, num_epochs + 1):
        logging.info("--- Epoch %d/%d ---", epoch, num_epochs)
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            max_grad_norm=max_grad_norm,
        )
        logging.info("Training loss: %.6f", train_loss)

        if epoch % eval_interval != 0:
            continue

        val_results = evaluate(model, valid_loader, topk_list, device)
        logging.info("Validation Results: %s", val_results)
        current_ndcg = val_results.get("NDCG@10", val_results.get(f"NDCG@{topk_list[-1]}", 0.0))

        if current_ndcg > best_ndcg:
            best_ndcg = current_ndcg
            best_epoch = epoch
            best_val_results = val_results
            early_stop_counter = 0

            test_results = evaluate(model, test_loader, topk_list, device)
            best_test_results = test_results
            logging.info("Test Results: %s", test_results)

            torch.save(model.state_dict(), save_path)
            logging.info("Best model saved to %s", save_path)
        else:
            early_stop_counter += 1
            logging.info(
                "No improvement since epoch %d. Early stop counter: %d/%d",
                best_epoch,
                early_stop_counter,
                early_stop,
            )
            if early_stop_counter >= early_stop:
                logging.info("Early stopping triggered.")
                break

    logging.info("=" * 50)
    logging.info("Training finished.")
    if best_test_results is not None:
        logging.info("Best epoch: %d", best_epoch)
        logging.info("Best validation results: %s", best_val_results)
        logging.info("Corresponding test results: %s", best_test_results)
    else:
        logging.info("No validation improvement was observed.")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()
