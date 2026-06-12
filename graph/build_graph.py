from typing import Dict, Iterable, List, Mapping, Tuple

import torch
from torch_geometric.data import HeteroData

from data import UserTrainValidTest

USER_NODE = "user"
ITEM_NODE = "item"
BRAND_NODE = "brand"
CATEGORY_NODE = "category"
TARGET_EDGE_TYPE = (USER_NODE, "interacts", ITEM_NODE)
REVERSE_EDGE_TYPE = (ITEM_NODE, "rev_interacts", USER_NODE)
ITEM_BRAND_EDGE = (ITEM_NODE, "has_brand", BRAND_NODE)
BRAND_ITEM_EDGE = (BRAND_NODE, "brand_of", ITEM_NODE)
ITEM_CATEGORY_EDGE = (ITEM_NODE, "has_category", CATEGORY_NODE)
CATEGORY_ITEM_EDGE = (CATEGORY_NODE, "category_of", ITEM_NODE)


def build_hetero_graph(dataset: Mapping[str, object]) -> HeteroData:
    graph_data = HeteroData()
    num_users = int(dataset["num_users"])
    num_items = int(dataset["num_items"])
    samples = dataset["samples"]
    item_metadata = dataset.get("item_metadata", {})

    graph_data[USER_NODE].num_nodes = num_users
    graph_data[ITEM_NODE].num_nodes = num_items

    item_embeddings = dataset.get("item_embeddings")
    if item_embeddings is not None:
        graph_data[ITEM_NODE].x = torch.tensor(item_embeddings, dtype=torch.float32)

    train_edge_index = build_edge_index(samples, split="train")
    valid_edge_index = build_edge_index(samples, split="valid")
    test_edge_index = build_edge_index(samples, split="test")

    graph_data[TARGET_EDGE_TYPE].edge_index = train_edge_index
    graph_data[REVERSE_EDGE_TYPE].edge_index = train_edge_index.flip(0)

    edge_store = graph_data[TARGET_EDGE_TYPE]
    edge_store.train_edge_label_index = train_edge_index
    edge_store.valid_edge_label_index = valid_edge_index
    edge_store.test_edge_label_index = test_edge_index

    # Build brand and category nodes and edges
    if item_metadata:
        brand_edges, num_brands = build_brand_edges(item_metadata, num_items)
        category_edges, num_categories = build_category_edges(item_metadata, num_items)

        if num_brands > 0:
            graph_data[BRAND_NODE].num_nodes = num_brands
            graph_data[ITEM_BRAND_EDGE].edge_index = brand_edges
            graph_data[BRAND_ITEM_EDGE].edge_index = brand_edges.flip(0)

        if num_categories > 0:
            graph_data[CATEGORY_NODE].num_nodes = num_categories
            graph_data[ITEM_CATEGORY_EDGE].edge_index = category_edges
            graph_data[CATEGORY_ITEM_EDGE].edge_index = category_edges.flip(0)

    return graph_data


def build_edge_index(
    samples: Iterable[UserTrainValidTest],
    split: str,
) -> torch.Tensor:
    users = []
    items = []

    for sample in samples:
        if split == "train":
            target_items = sample.train_items
        elif split == "valid":
            target_items = [sample.valid_item]
        elif split == "test":
            target_items = [sample.test_item]

        for item in target_items:
            users.append(int(sample.user))
            items.append(int(item))

    if not users:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([users, items], dtype=torch.long)


def build_brand_edges(item_metadata: Dict[int, dict], num_items: int) -> Tuple[torch.Tensor, int]:
    """Build item->brand edges and return (edge_index, num_brands)."""
    brand_to_id = {}
    item_ids = []
    brand_ids = []

    for item_id in range(num_items):
        if item_id not in item_metadata:
            continue
        brand = item_metadata[item_id].get("brand", "")
        if not brand or brand == "":
            continue

        if brand not in brand_to_id:
            brand_to_id[brand] = len(brand_to_id)

        item_ids.append(item_id)
        brand_ids.append(brand_to_id[brand])

    if not item_ids:
        return torch.empty((2, 0), dtype=torch.long), 0

    edge_index = torch.tensor([item_ids, brand_ids], dtype=torch.long)
    return edge_index, len(brand_to_id)


def build_category_edges(item_metadata: Dict[int, dict], num_items: int) -> Tuple[torch.Tensor, int]:
    """Build item->category edges and return (edge_index, num_categories).

    Each item can have multiple category paths (e.g., [['Beauty', 'Makeup', 'Face']]).
    We flatten all categories and create one edge per unique category.
    """
    category_to_id = {}
    item_ids = []
    category_ids = []

    for item_id in range(num_items):
        if item_id not in item_metadata:
            continue
        categories = item_metadata[item_id].get("categories", [])
        if not categories:
            continue

        # Flatten all category paths and extract unique categories
        unique_cats = set()
        for cat_path in categories:
            if isinstance(cat_path, list):
                unique_cats.update(cat_path)
            elif isinstance(cat_path, str):
                unique_cats.add(cat_path)

        for cat in unique_cats:
            if not cat or cat == "":
                continue
            if cat not in category_to_id:
                category_to_id[cat] = len(category_to_id)

            item_ids.append(item_id)
            category_ids.append(category_to_id[cat])

    if not item_ids:
        return torch.empty((2, 0), dtype=torch.long), 0

    edge_index = torch.tensor([item_ids, category_ids], dtype=torch.long)
    return edge_index, len(category_to_id)

def summarize_hetero_graph(graph_data: HeteroData) -> Dict[str, object]:
    edge_store = graph_data[TARGET_EDGE_TYPE]
    item_x = getattr(graph_data[ITEM_NODE], "x", None)

    summary = {
        "node_types": list(graph_data.node_types),
        "edge_types": list(graph_data.edge_types),
        "num_users": int(graph_data[USER_NODE].num_nodes),
        "num_items": int(graph_data[ITEM_NODE].num_nodes),
        "item_feature_shape": tuple(item_x.shape) if item_x is not None else None,
        "train_edges": int(edge_store.train_edge_label_index.size(1)),
        "valid_edges": int(edge_store.valid_edge_label_index.size(1)),
        "test_edges": int(edge_store.test_edge_label_index.size(1)),
    }

    # Add brand and category node counts if they exist
    if BRAND_NODE in graph_data.node_types:
        summary["num_brands"] = int(graph_data[BRAND_NODE].num_nodes)
        summary["brand_edges"] = int(graph_data[ITEM_BRAND_EDGE].edge_index.size(1))

    if CATEGORY_NODE in graph_data.node_types:
        summary["num_categories"] = int(graph_data[CATEGORY_NODE].num_nodes)
        summary["category_edges"] = int(graph_data[ITEM_CATEGORY_EDGE].edge_index.size(1))

    return summary
