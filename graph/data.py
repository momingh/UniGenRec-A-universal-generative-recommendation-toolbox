import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set


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


def load_dataset(data_root: Path, dataset: str) -> Dict[str, object]:
    dataset_dir = data_root / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    valid_rows = load_split(dataset_dir / f"{dataset}.valid.jsonl")
    test_rows = load_split(dataset_dir / f"{dataset}.test.jsonl")
    samples = merge_valid_test(valid_rows, test_rows)
    used_item_ids = collect_used_item_ids(samples)

    return {
        "num_users": load_id_count(dataset_dir / f"{dataset}.user2id"),
        "num_items": load_id_count(dataset_dir / f"{dataset}.item2id"),
        "samples": samples,
        "used_item_ids": used_item_ids,
        "item_info": load_used_item_info(dataset_dir / f"{dataset}.item.json", used_item_ids),
    }
