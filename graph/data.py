import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np


@dataclass
class UserTrainValidTest:
    user: int
    train_items: List[int]
    valid_item: int
    test_item: int


def load_id_count(path: Path) -> int:
    max_id = -1
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"Bad id mapping line in {path}: {line}")
            max_id = max(max_id, int(parts[1]))
    return max_id + 1


@dataclass
class SplitRow:
    user: int
    history: List[int]
    target: int


def load_split(path: Path) -> List[SplitRow]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line_num, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            try:
                rows.append(
                    SplitRow(
                        user=int(obj["user"]),
                        history=[int(x) for x in obj.get("history", [])],
                        target=int(obj["target"]),
                    )
                )
            except KeyError as exc:
                raise KeyError(f"{path}:{line_num} missing field {exc}") from exc
    print(f"Loaded {path.name}: {len(rows)} users")
    return rows


def merge_valid_test(valid_rows: List[SplitRow], test_rows: List[SplitRow]) -> List[UserTrainValidTest]:
    test_by_user = {row.user: row for row in test_rows}
    samples = []
    for valid_row in valid_rows:
        test_row = test_by_user.get(valid_row.user)
        if test_row is None:
            raise ValueError(f"Missing test row for user {valid_row.user}")
        samples.append(
            UserTrainValidTest(
                user=valid_row.user,
                train_items=valid_row.history,
                valid_item=valid_row.target,
                test_item=test_row.target,
            )
        )
    return samples


def collect_used_item_ids(samples: List[UserTrainValidTest]) -> Set[int]:
    item_ids = set()
    for sample in samples:
        item_ids.update(sample.train_items)
        item_ids.add(sample.valid_item)
        item_ids.add(sample.test_item)
    return item_ids


def load_used_item_info(path: Path, item_ids: Set[int]) -> Dict[int, dict]:
    with path.open("r", encoding="utf-8") as fp:
        all_item_info = json.load(fp)

    item_info = {}
    missing_ids = []
    for item_id in sorted(item_ids):
        key = str(item_id)
        if key in all_item_info:
            raw_info = all_item_info[key]
            item_info[item_id] = {
                "brand": raw_info.get("brand", ""),
                "categories": raw_info.get("categories", []),
            }
        else:
            missing_ids.append(item_id)

    if missing_ids:
        raise KeyError(f"{path} missing metadata for item ids: {missing_ids[:10]}")

    print(f"Loaded {path.name}: {len(item_info)} used items")
    return item_info


def load_all_item_metadata(path: Path, num_items: int) -> Dict[int, dict]:
    """Load brand and categories for all items in the dataset."""
    with path.open("r", encoding="utf-8") as fp:
        all_item_info = json.load(fp)

    item_metadata = {}
    for item_id in range(num_items):
        key = str(item_id)
        if key in all_item_info:
            raw_info = all_item_info[key]
            item_metadata[item_id] = {
                "brand": raw_info.get("brand", ""),
                "categories": raw_info.get("categories", []),
            }
        else:
            # Missing metadata, use empty values
            item_metadata[item_id] = {
                "brand": "",
                "categories": [],
            }

    print(f"Loaded {path.name}: {len(item_metadata)} items with brand/category metadata")
    return item_metadata


def resolve_embedding_path(
    dataset_dir: Path,
    dataset: str,
    embedding_model: Optional[str],
    embedding_modality: str = "text",
    embedding_path: Optional[Path] = None,
) -> Optional[Path]:
    if embedding_path is not None:
        if embedding_path.is_file():
            return embedding_path
        candidate = dataset_dir / "embeddings" / embedding_path
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

    if not embedding_model:
        return None

    model_path = Path(embedding_model)
    candidates = []
    if model_path.is_file():
        candidates.append(model_path)
    if model_path.suffix == ".npy":
        candidates.append(dataset_dir / "embeddings" / model_path.name)
    candidates.append(
        dataset_dir
        / "embeddings"
        / f"{dataset}.emb-{embedding_modality}-{embedding_model}.npy"
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    embedding_dir = dataset_dir / "embeddings"
    available = sorted(path.name for path in embedding_dir.glob("*.npy"))
    raise FileNotFoundError(
        f"Embedding file not found for model={embedding_model!r}, "
        f"modality={embedding_modality!r}. Available files: {available}"
    )


def load_embeddings(path: Path, expected_items: int) -> np.ndarray:
    embeddings = np.load(path, mmap_mode="r")
    if embeddings.ndim != 2:
        raise ValueError(
            f"Embedding file must be 2D, got shape={embeddings.shape}: {path}"
        )
    if embeddings.shape[0] != expected_items:
        raise ValueError(
            f"Embedding row count ({embeddings.shape[0]}) does not match "
            f"num_items ({expected_items}): {path}"
        )

    print(f"Loaded {path.name}: shape={embeddings.shape}, dtype={embeddings.dtype}")
    return embeddings


def load_dataset(
    data_root: Path,
    dataset: str,
    embedding_model: Optional[str] = None,
    embedding_modality: str = "text",
    embedding_path: Optional[Path] = None,
) -> Dict[str, object]:
    dataset_dir = data_root / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    valid_rows = load_split(dataset_dir / f"{dataset}.valid.jsonl")
    test_rows = load_split(dataset_dir / f"{dataset}.test.jsonl")
    samples = merge_valid_test(valid_rows, test_rows)
    used_item_ids = collect_used_item_ids(samples)
    num_users = load_id_count(dataset_dir / f"{dataset}.user2id")
    num_items = load_id_count(dataset_dir / f"{dataset}.item2id")
    resolved_embedding_path = resolve_embedding_path(
        dataset_dir,
        dataset,
        embedding_model=embedding_model,
        embedding_modality=embedding_modality,
        embedding_path=embedding_path,
    )
    item_embeddings = (
        load_embeddings(resolved_embedding_path, num_items)
        if resolved_embedding_path is not None
        else None
    )

    item_json_path = dataset_dir / f"{dataset}.item.json"

    return {
        "num_users": num_users,
        "num_items": num_items,
        "samples": samples,
        "used_item_ids": used_item_ids,
        "item_info": load_used_item_info(item_json_path, used_item_ids),
        "item_metadata": load_all_item_metadata(item_json_path, num_items),
        "embedding_path": resolved_embedding_path,
        "item_embeddings": item_embeddings,
    }
