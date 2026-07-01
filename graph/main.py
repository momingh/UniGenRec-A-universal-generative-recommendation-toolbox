import argparse
import json
import logging
import pprint
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.sampler import NegativeSampling

from build_graph import (
    TARGET_EDGE_TYPE,
    build_hetero_graph,
    summarize_hetero_graph,
)
from data import load_dataset
from models import build_han_link_predictor, build_lightgcn_link_predictor
from training import GraphTrainer, build_lr_scheduler

GRAPH_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "Beauty"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large-pca512"
DEFAULT_EMBEDDING_MODALITY = "text"
DEFAULT_CONFIG = GRAPH_ROOT / "config" / "lightgcn.yaml"
LOG_SEPARATOR = "=" * 50


def main():
    parser = argparse.ArgumentParser(
        description="Train graph recommendation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--data_root", type=Path, default=PROJECT_ROOT / "datasets")
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model tag, e.g. text-embedding-3-large or Qwen3-Embedding-8B-pca512.",
    )
    parser.add_argument(
        "--embedding_modality",
        type=str,
        default=DEFAULT_EMBEDDING_MODALITY,
        help="Embedding modality tag used in the filename, e.g. text or review-ui.",
    )
    parser.add_argument(
        "--embedding_path",
        type=Path,
        default=None,
        help="Optional explicit .npy embedding path or filename under the dataset embeddings dir.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Training config YAML path.",
    )
    args = parser.parse_args()

    config_path = resolve_train_config_path(args.config)
    train_config = load_train_config(config_path)
    model_config = require_config(train_config, "model", "config")
    model_name = require_config(model_config, "name", "config.model")
    setup_logging()
    logging.info(f"Configuration loaded for {model_name} on {args.dataset}.")
    logging.info(LOG_SEPARATOR)
    logging.info(
        "\n" + pprint.pformat(build_log_config(args, config_path, train_config))
    )
    logging.info(LOG_SEPARATOR)

    logging.info("Loading graph dataset...")
    data = load_dataset(
        args.data_root,
        args.dataset,
        embedding_model=args.embedding_model,
        embedding_modality=args.embedding_modality,
        embedding_path=args.embedding_path,
    )
    logging.info("Building heterogeneous graph...")
    graph_data = build_hetero_graph(data)
    graph_summary = summarize_hetero_graph(graph_data)
    log_data_summary(data, graph_summary)

    logging.info(f"Training config: {config_path}")
    run_training(
        train_config,
        graph_data,
        dataset_name=args.dataset,
        data_root=args.data_root,
    )


def setup_logging():
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

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        ColorFormatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)


def build_log_config(args, config_path: Path, train_config):
    return {
        "dataset_name": args.dataset,
        "data_root": str(args.data_root),
        "embedding_model": args.embedding_model,
        "embedding_modality": args.embedding_modality,
        "embedding_path": str(args.embedding_path) if args.embedding_path else None,
        "training_config": str(config_path),
        **train_config,
    }


def log_data_summary(data, graph_summary):
    logging.info(
        "Dataset loaded. "
        f"Users={data['num_users']} | "
        f"Items={data['num_items']} | "
        f"Users with samples={len(data['samples'])} | "
        f"Train interactions={sum(len(x.train_items) for x in data['samples'])} | "
        f"Valid targets={len(data['samples'])} | "
        f"Test targets={len(data['samples'])}"
    )
    logging.info(
        "Item metadata loaded. "
        f"Used items={len(data['used_item_ids'])} | "
        f"Used item metadata={len(data['item_info'])} | "
        f"All item metadata={len(data.get('item_metadata', {}))}"
    )
    if data["item_embeddings"] is not None:
        logging.info(
            "Item embeddings loaded. "
            f"path={data['embedding_path']} | "
            f"shape={data['item_embeddings'].shape} | "
            f"dtype={data['item_embeddings'].dtype}"
        )

    logging.info(
        "Graph built. "
        f"node_types={graph_summary['node_types']} | "
        f"edge_types={graph_summary['edge_types']} | "
        f"item_feature_shape={graph_summary['item_feature_shape']}"
    )
    logging.info(
        "Graph splits. "
        f"train_edges={graph_summary['train_edges']} | "
        f"valid_edges={graph_summary['valid_edges']} | "
        f"test_edges={graph_summary['test_edges']}"
    )
    if "num_brands" in graph_summary:
        logging.info(
            "Brand graph. "
            f"nodes={graph_summary['num_brands']} | "
            f"edges={graph_summary['brand_edges']}"
        )
    if "num_categories" in graph_summary:
        logging.info(
            "Category graph. "
            f"nodes={graph_summary['num_categories']} | "
            f"edges={graph_summary['category_edges']}"
        )


def resolve_train_config_path(path: Path) -> Path:
    candidates = [path]
    if not path.is_absolute():
        candidates.extend(
            [
                PROJECT_ROOT / path,
                GRAPH_ROOT / path,
                GRAPH_ROOT / "config" / path,
                GRAPH_ROOT / "config" / path.name,
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    tried = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Training config not found: {path}\nTried:\n{tried}")


def load_train_config(path: Path):
    with path.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp) or {}
    return config


def require_config(config, key: str, path: str):
    if key not in config:
        raise KeyError(f"Missing required config field: {path}.{key}")
    return config[key]


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_train_loader(
    graph_data,
    batch_size: int,
    num_neighbors,
    negative_samples: int,
    num_workers: int,
):
    edge_store = graph_data[TARGET_EDGE_TYPE]
    num_neighbors_by_edge = {
        edge_type: list(num_neighbors)
        for edge_type in graph_data.edge_types
    }

    return LinkNeighborLoader(
        graph_data,
        num_neighbors=num_neighbors_by_edge,
        edge_label_index=(TARGET_EDGE_TYPE, edge_store.train_edge_label_index),
        neg_sampling=NegativeSampling("triplet", amount=int(negative_samples)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )


def run_training(config, graph_data, dataset_name: str, data_root: Path):
    model_config = require_config(config, "model", "config")
    loader_config = require_config(config, "loader", "config")
    metrics_config = require_config(config, "metrics", "config")
    training_config = require_config(config, "training", "config")
    output_config = config.get("output") or {}
    seed = int(require_config(training_config, "seed", "config.training"))
    set_random_seed(seed)

    hidden_dim = int(require_config(model_config, "hidden_dim", "config.model"))
    num_layers = int(require_config(model_config, "num_layers", "config.model"))
    model_name = require_config(model_config, "name", "config.model").lower()

    batch_size = int(require_config(loader_config, "batch_size", "config.loader"))
    num_neighbors = require_config(loader_config, "num_neighbors", "config.loader")
    negative_samples = int(
        require_config(loader_config, "negative_samples", "config.loader")
    )
    num_workers = int(require_config(loader_config, "num_workers", "config.loader"))

    eval_top_k = [
        int(k)
        for k in require_config(metrics_config, "top_k", "config.metrics")
    ]
    monitor_metric = f"NDCG@{eval_top_k[0]}"
    full_sort_batch_size = int(
        require_config(
            metrics_config,
            "full_sort_batch_size",
            "config.metrics",
        )
    )

    device_name = require_config(training_config, "device", "config.training")
    epochs = int(require_config(training_config, "epochs", "config.training"))
    lr = float(require_config(training_config, "lr", "config.training"))
    weight_decay = float(
        require_config(training_config, "weight_decay", "config.training")
    )
    warmup_steps = int(
        require_config(training_config, "warmup_steps", "config.training")
    )
    scheduler_name = require_config(training_config, "scheduler", "config.training")
    max_train_batches = require_config(
        training_config,
        "max_train_batches",
        "config.training",
    )
    early_stop_patience = int(
        require_config(training_config, "early_stop_patience", "config.training")
    )

    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    logging.info(f"Using device: {device}")

    logging.info("Creating DataLoaders...")
    train_loader = build_train_loader(
        graph_data=graph_data,
        batch_size=batch_size,
        num_neighbors=num_neighbors,
        negative_samples=negative_samples,
        num_workers=num_workers,
    )
    logging.info(f"Dynamically loading model: {model_name}")
    model = build_model(
        model_name=model_name,
        model_config=model_config,
        graph_data=graph_data,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        eval_top_k=eval_top_k,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    steps_per_epoch = len(train_loader)
    if max_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, int(max_train_batches))
    scheduler = build_lr_scheduler(
        optimizer,
        scheduler_name=scheduler_name,
        warmup_steps=warmup_steps,
        steps_per_epoch=steps_per_epoch,
        num_epochs=epochs,
    )

    logging.info(
        "Training setup. "
        f"seed={seed} | "
        f"model={model_name} | "
        f"loss=BPR | "
        f"train_batches={len(train_loader)} | "
        f"negative_samples={negative_samples} | "
        f"full_sort_batch_size={full_sort_batch_size} | "
        f"monitor_metric={monitor_metric} | "
        f"early_stop_patience={early_stop_patience}"
    )
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    logging.info(LOG_SEPARATOR)

    trainer = GraphTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        max_train_batches=max_train_batches,
    )
    trainer.fit(
        train_loader,
        num_epochs=epochs,
        valid_data=graph_data,
        valid_split="valid",
        test_data=graph_data,
        test_split="test",
        full_sort_batch_size=full_sort_batch_size,
        monitor_metric=monitor_metric,
        early_stop_patience=early_stop_patience,
    )
    save_final_item_embeddings(
        model=model,
        graph_data=graph_data,
        device=device,
        output_config=output_config,
        dataset_name=dataset_name,
        model_name=model_name,
        item2id_path=data_root / dataset_name / f"{dataset_name}.item2id",
    )


def build_model(
    model_name: str,
    model_config,
    graph_data,
    hidden_dim: int,
    num_layers: int,
    eval_top_k,
):
    if model_name == "han":
        return build_han_link_predictor(
            graph_data,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=int(require_config(model_config, "heads", "config.model")),
            negative_slope=float(
                require_config(model_config, "negative_slope", "config.model")
            ),
            dropout=float(require_config(model_config, "dropout", "config.model")),
            eval_top_k=eval_top_k,
            target_edge_type=TARGET_EDGE_TYPE,
        )
    if model_name == "lightgcn":
        return build_lightgcn_link_predictor(
            graph_data,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            eval_top_k=eval_top_k,
            target_edge_type=TARGET_EDGE_TYPE,
        )
    raise ValueError(f"Unsupported model.name: {model_name}")


def resolve_output_root(output_config, dataset_name: str, model_name: str) -> Path:
    output_root = output_config.get(
        "output_root",
        "../ckpt/graph/{dataset_name}/{model_name}",
    )
    path = Path(
        str(output_root).format(
            dataset_name=dataset_name,
            model_name=model_name,
        )
    )
    if not path.is_absolute():
        path = GRAPH_ROOT / path
    return path.resolve()


def load_item_row_mapping(item2id_path: Path, num_items: int) -> list[str]:
    id_to_raw_item = [None] * int(num_items)
    with item2id_path.open("r", encoding="utf-8") as fp:
        for line_num, line in enumerate(fp, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"Bad item2id line in {item2id_path}:{line_num}: {line}"
                )
            raw_item_id, item_id_text = parts
            item_id = int(item_id_text)
            if not 0 <= item_id < num_items:
                raise ValueError(
                    f"item_id out of range in {item2id_path}:{line_num}: {item_id}"
                )
            id_to_raw_item[item_id] = raw_item_id

    missing = [idx for idx, raw_item in enumerate(id_to_raw_item) if raw_item is None]
    if missing:
        raise ValueError(
            f"{item2id_path} missing raw item ids for remapped ids: {missing[:10]}"
        )
    return id_to_raw_item


def save_final_item_embeddings(
    model,
    graph_data,
    device: torch.device,
    output_config,
    dataset_name: str,
    model_name: str,
    item2id_path: Path,
):
    if not hasattr(model, "encode"):
        raise AttributeError("Model must implement encode(...) to save item embeddings.")

    output_root = resolve_output_root(output_config, dataset_name, model_name)
    output_root.mkdir(parents=True, exist_ok=True)
    embedding_path = output_root / output_config.get(
        "item_embeddings",
        "final_item_embeddings.npy",
    )
    mapping_path = output_root / output_config.get(
        "item_mapping",
        "final_item_embedding_mapping.jsonl",
    )

    model.eval()
    graph_data = graph_data.to(device)
    with torch.no_grad():
        z_dict = model.encode(graph_data)
        if "item" not in z_dict:
            raise KeyError("Encoded graph output does not contain an 'item' embedding.")
        item_embeddings = z_dict["item"].detach().cpu().numpy()

    np.save(embedding_path, item_embeddings)

    id_to_raw_item = load_item_row_mapping(item2id_path, item_embeddings.shape[0])
    with mapping_path.open("w", encoding="utf-8") as fp:
        for row_index, raw_item_id in enumerate(id_to_raw_item):
            fp.write(
                json.dumps(
                    {
                        "row_index": row_index,
                        "item_id": row_index,
                        "raw_item_id": raw_item_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    logging.info(
        "Saved final item embeddings. "
        f"embeddings={embedding_path} | "
        f"mapping={mapping_path} | "
        f"shape={item_embeddings.shape}"
    )


if __name__ == "__main__":
    main()
