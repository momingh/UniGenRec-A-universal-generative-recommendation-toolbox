import argparse
import logging
import random
import torch
import torch.optim as optim
import pprint
import time
from contextlib import nullcontext

import numpy as np
from torch.utils.data import DataLoader
from transformers.optimization import get_scheduler
from dataset import GenRecDataset, item2code
from tokenizer import get_tokenizer
from trainer import train_one_epoch, evaluate
from utils import (
    load_and_process_config,
    setup_logging,
    set_seed,
    get_model_class,
    format_metric_value,
    format_metrics_line,
)

try:
    from tqdm.contrib.logging import logging_redirect_tqdm
except ImportError:
    logging_redirect_tqdm = nullcontext

import sys
from pathlib import Path

root = Path(__file__).resolve().parent
root_parent = root.parent

if str(root_parent) not in sys.path:
    sys.path.insert(0, str(root_parent))

from recommendation.models.generation.prefix_tree import build_trie_from_codebook


def get_total_steps(training_params, train_loader):
    total_steps = training_params.get('steps')
    if total_steps is None:
        total_steps = training_params.get('total_steps')
    if total_steps is not None:
        return int(total_steps)

    num_epochs = training_params.get('epochs')
    if num_epochs is None:
        num_epochs = training_params.get('num_epochs')
    if num_epochs is None:
        raise ValueError("training_params must define either steps/total_steps or epochs/num_epochs")

    return len(train_loader) * int(num_epochs)


def build_lr_scheduler(optimizer, train_loader, training_params):
    scheduler_name = training_params.get('scheduler')
    if scheduler_name is None:
        scheduler_name = training_params.get('lr_scheduler')
    if scheduler_name is None:
        return None

    scheduler_name = str(scheduler_name).lower()
    if scheduler_name in {'none', 'null', 'false'}:
        return None
    scheduler_aliases = {
        'cosine': 'cosine',
        'constant_with_warmup': 'constant_with_warmup',
        'warmup_constant': 'constant_with_warmup',
    }
    if scheduler_name not in scheduler_aliases:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    scheduler_name = scheduler_aliases[scheduler_name]

    warmup_steps = int(training_params.get('warmup_steps', 0) or 0)
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


def get_current_lr(optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def format_lr(optimizer) -> str:
    return f"{get_current_lr(optimizer):.8e}"


def reset_run_seed(seed: int) -> None:
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


def main():
    parser = argparse.ArgumentParser(description="GenRec Universal Training Pipeline")
    parser.add_argument('--model', type=str, default="TIGER", help='模型名稱 (e.g., TIGER, GPT2, RPG)')
    parser.add_argument('--dataset', type=str, default="amazon-musical-instruments-23", help='数据集名稱 (e.g., Beauty)')
    parser.add_argument('--quant_method', type=str, default="rqvae", choices=['rqvae', 'rqvae_faiss', 'opq', 'qinco', 'qinco_aux', 'qinco_v2', 'rqkmeans', 'rqkmeans_plus'], help='量化方法')
    parser.add_argument('--embedding_modality', type=str, default='text', help='量化模态类型，对应不同的 codebook (默认 text，例如 graph-lightgcn)')
    parser.add_argument('--eval_only', action='store_true', help='仅加载已有模型，在测试集上直接评估')

    args = parser.parse_args()
    eval_only = args.eval_only

    config = load_and_process_config(
        args.model,
        args.dataset,
        args.quant_method,
        embedding_modality=args.embedding_modality
    )

    setup_logging(config['log_path'])
    run_seed = int(config['training_params']['seed'])
    reset_run_seed(run_seed)
    logging.info(f"Configuration loaded for {args.model} on {args.dataset} with {args.quant_method}.")
    logging.info(f"Run seed reset to: {run_seed}")
    logging.info("=" * 50)
    config_str = pprint.pformat(config)
    logging.info("\n" + config_str)
    logging.info("=" * 50)

    device = torch.device(config['training_params']['device'] if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    num_workers = config['training_params'].get('num_workers', 4)

    logging.info("Loading item to code mapping...")
    item_to_code_map, _ = item2code(
        config['code_path'],
        config['vocab_sizes'],
        config['bases']
    )
    logging.info(f"Item to code map loaded. Total items mapped: {len(item_to_code_map)}")

    prefix_trie = None
    use_trie = config.get('evaluation_params', {}).get('use_prefix_trie', False)

    if use_trie:
        logging.info("Building Prefix Trie (enabled in config)...")
        prefix_trie = build_trie_from_codebook(
            token_sequences=list(item_to_code_map.values()),
            eos_token_id=config['token_params']['eos_token_id']
        )
    else:
        logging.info("Prefix Trie is DISABLED (default or as per config).")

    logging.info(f"Dynamically loading model: {args.model}")
    ModelClass = get_model_class(args.model)
    model = ModelClass(config, prefix_trie=prefix_trie, item_to_code_map=item_to_code_map)

    model.to(device)
    logging.info(model.n_parameters)
    logging.info("=" * 50)
    weight_decay = float(config['training_params'].get('weight_decay', 0.01))
    optimizer = None
    if not eval_only:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(config['training_params']['lr']),
            weight_decay=weight_decay
        )

    logging.info(f"Initializing tokenizer for model: {args.model}")
    collate_fn = get_tokenizer(
        model_name=args.model,
        config=config,
        item_to_code_map=item_to_code_map
    )
    logging.info("Tokenizer initialized.")

    logging.info("Creating Datasets...")
    if eval_only:
        test_dataset = GenRecDataset(config=config, mode='test')
    else:
        train_dataset = GenRecDataset(config=config, mode='train')
        validation_dataset = GenRecDataset(config=config, mode='valid')
        test_dataset = GenRecDataset(config=config, mode='test')

    logging.info("Creating DataLoaders...")

    is_gpu_training = (torch.cuda.is_available() and num_workers > 0)
    train_generator = torch.Generator()
    train_generator.manual_seed(run_seed)
    loader_kwargs = {
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": is_gpu_training,
        "persistent_workers": is_gpu_training if num_workers > 0 else False,
        "worker_init_fn": make_worker_init_fn(run_seed),
    }
    eval_loader_kwargs = {
        **loader_kwargs,
        "batch_size": config['evaluation_params']['batch_size'],
        "shuffle": False,
    }

    if eval_only:
        test_loader = DataLoader(test_dataset, **eval_loader_kwargs)
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training_params']['batch_size'],
            shuffle=True,
            generator=train_generator,
            **loader_kwargs
        )
        validation_loader = DataLoader(validation_dataset, **eval_loader_kwargs)
        test_loader = DataLoader(test_dataset, **eval_loader_kwargs)

    topk_list = config['evaluation_params']['topk_list']
    save_path = config['save_path']

    def run_eval(label, loader):
        results = evaluate(model, loader, topk_list, device)
        logging.info(format_metrics_line(label, results))
        return results

    if eval_only:
        ckpt_path = Path(save_path)
        if not ckpt_path.exists():
            logging.error(f"[Eval-Only] Checkpoint not found: {ckpt_path}")
            return
        if ckpt_path.is_dir() and hasattr(model, "load_pretrained"):
            model.load_pretrained(str(ckpt_path))
            model.to(device)
            logging.info(f"[Eval-Only] Loaded pretrained checkpoint from {ckpt_path}")
        else:
            state_dict = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state_dict)
            logging.info(f"[Eval-Only] Loaded checkpoint from {ckpt_path}")

        run_eval("[Eval-Only] Test", test_loader)
        return

    scheduler = build_lr_scheduler(optimizer, train_loader, config['training_params'])
    max_grad_norm = config['training_params'].get('max_grad_norm')
    if max_grad_norm is not None:
        max_grad_norm = float(max_grad_norm)
        logging.info(f"Gradient clipping enabled: max_grad_norm={max_grad_norm}")

    best_ndcg = 0.0
    early_stop_counter = 0
    best_epoch = 0
    best_val_results = None
    best_test_results = None

    eval_interval = config['training_params'].get('eval_interval', 1)
    logging.info(f"Evaluation interval set to: {eval_interval} epoch(s)")

    num_epochs = config['training_params']['num_epochs']
    best_metric_name = 'NDCG@10'
    fallback_best_metric_name = 'NDCG@20'
    early_stop_limit = config['training_params']['early_stop'] * eval_interval
    epoch_separator = "-" * 80

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
                max_grad_norm=max_grad_norm
            )

            if epoch_num % eval_interval == 0:
                logging.info(f"Evaluating epoch {epoch_num:03d} | lr={format_lr(optimizer)}")
                val_results = run_eval("Validation", validation_loader)

                current_ndcg = val_results.get(best_metric_name, val_results.get(fallback_best_metric_name, 0.0))

                if current_ndcg > best_ndcg:
                    best_ndcg = current_ndcg
                    early_stop_counter = 0

                    test_results = run_eval("Test", test_loader)

                    best_epoch = epoch_num
                    best_val_results = val_results
                    best_test_results = test_results

                    torch.save(model.state_dict(), save_path)
                    logging.info(
                        f"New best | epoch={epoch_num:03d} | "
                        f"{best_metric_name}={format_metric_value(best_ndcg)} | "
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
                    f"val {best_metric_name}={format_metric_value(current_ndcg)} | "
                    f"best={format_metric_value(best_ndcg)} | "
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
                    f"best={format_metric_value(best_ndcg)} | "
                    f"time={epoch_time:.2f}s"
                )

    logging.info("="*50)
    logging.info("Training finished.")
    if best_test_results:
        logging.info(f"Best Epoch: {best_epoch:03d}")
        logging.info(format_metrics_line("Best Validation", best_val_results))
        logging.info(format_metrics_line("Corresponding Test", best_test_results))
    else:
        logging.info("No improvement was observed.")
    logging.info("="*50)


if __name__ == "__main__":
    main()
