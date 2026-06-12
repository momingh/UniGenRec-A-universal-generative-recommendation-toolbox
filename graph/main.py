import argparse
import random
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


def main():
    parser = argparse.ArgumentParser(
        description="Load graph recommendation splits",
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
        "--train",
        action="store_true",
        help="Run graph link prediction training.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Training config YAML path.",
    )
    args = parser.parse_args()

    data = load_dataset(
        args.data_root,
        args.dataset,
        embedding_model=args.embedding_model,
        embedding_modality=args.embedding_modality,
        embedding_path=args.embedding_path,
    )
    graph_data = build_hetero_graph(data)
    graph_summary = summarize_hetero_graph(graph_data)

    label_width = 38

    def print_row(label, value):
        print(f"{label:<{label_width}}: {value}")

    print_row("Dataset", args.dataset)
    print_row("Users", data["num_users"])
    print_row("Items", data["num_items"])
    print_row("Users with samples", len(data["samples"]))
    print_row(
        "Train interactions from valid.history",
        sum(len(x.train_items) for x in data["samples"]),
    )
    print_row("Valid targets from valid.target", len(data["samples"]))
    print_row("Test targets from test.target", len(data["samples"]))
    print_row("Used items", len(data["used_item_ids"]))
    print_row("Loaded item metadata", len(data["item_info"]))
    print_row("All item metadata", len(data.get("item_metadata", {})))
    if data["item_embeddings"] is not None:
        print_row("Embedding path", data["embedding_path"])
        print_row("Embedding shape", data["item_embeddings"].shape)
        print_row("Embedding dtype", data["item_embeddings"].dtype)
    print_row("Graph node types", graph_summary["node_types"])
    print_row("Graph edge types", graph_summary["edge_types"])
    print_row("Graph item feature shape", graph_summary["item_feature_shape"])
    if "num_brands" in graph_summary:
        print_row("Graph brand nodes", graph_summary["num_brands"])
        print_row("Graph brand edges", graph_summary["brand_edges"])
    if "num_categories" in graph_summary:
        print_row("Graph category nodes", graph_summary["num_categories"])
        print_row("Graph category edges", graph_summary["category_edges"])
    print_row("Graph train edges", graph_summary["train_edges"])
    print_row("Graph valid edges", graph_summary["valid_edges"])
    print_row("Graph test edges", graph_summary["test_edges"])

    if data["samples"]:
        print_row("First sample", data["samples"][0])
        first_item = data["samples"][0].train_items[0]
        print_row(
            "First used item info",
            f"{first_item} -> {data['item_info'][first_item]}",
        )

    if args.train:
        config_path = resolve_train_config_path(args.config)
        train_config = load_train_config(config_path)
        print_row("Training config", config_path)
        run_training(train_config, graph_data, print_row)


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


def run_training(config, graph_data, print_row):
    model_config = require_config(config, "model", "config")
    loader_config = require_config(config, "loader", "config")
    metrics_config = require_config(config, "metrics", "config")
    training_config = require_config(config, "training", "config")
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

    device = torch.device(device_name)
    print_row("Random seed", seed)
    print_row("Training device", device)

    train_loader = build_train_loader(
        graph_data=graph_data,
        batch_size=batch_size,
        num_neighbors=num_neighbors,
        negative_samples=negative_samples,
        num_workers=num_workers,
    )
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

    print_row("Train batches", len(train_loader))
    print_row("Training loss", "BPR")
    print_row("Model name", model_name)
    print_row("Negatives per positive", negative_samples)
    print_row("Full-sort eval batch size", full_sort_batch_size)
    print_row("Monitor metric", monitor_metric)
    print_row("Early stop patience", early_stop_patience)
    print_row("Model parameters", sum(p.numel() for p in model.parameters()))

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


if __name__ == "__main__":
    main()
