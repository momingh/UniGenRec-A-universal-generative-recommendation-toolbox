import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch.utils.data import Dataset


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _to_int_list(values: Iterable[Any]) -> List[int]:
    return [int(value) for value in values]


def restore_train_sequences(train_json: Path) -> Tuple[List[Tuple[int, List[int]]], Dict[int, List[int]]]:
    user_to_seq: Dict[int, List[int]] = {}
    user_order: List[int] = []
    userless_sequences: List[Tuple[int, List[int]]] = []
    next_userless_id = -1

    for obj in load_jsonl(train_json):
        history = _to_int_list(obj.get("history", []))
        target = int(obj["target"])
        item_seq = history + [target]
        if len(item_seq) < 2:
            continue

        raw_user = obj.get("user")
        if raw_user is None:
            userless_sequences.append((next_userless_id, item_seq))
            next_userless_id -= 1
            continue

        user = int(raw_user)
        if user not in user_to_seq:
            user_to_seq[user] = item_seq
            user_order.append(user)
            continue

        expected_history = user_to_seq[user][-len(history):] if history else []
        if expected_history != history:
            raise ValueError(
                f"Train records for user {user} are not in prefix order; "
                "cannot restore the full sequence from train.jsonl."
            )
        user_to_seq[user].append(target)

    ordered_sequences = [(user, user_to_seq[user]) for user in user_order]
    ordered_sequences.extend(userless_sequences)
    seen_items = {user: sorted(set(seq)) for user, seq in ordered_sequences if user >= 0}
    return ordered_sequences, seen_items


def expand_train_sequences(
    user_sequences: List[Tuple[int, List[int]]],
    max_len: int,
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for _user, item_seq in user_sequences:
        if len(item_seq) < 2:
            continue

        n_return_examples = max(len(item_seq) - max_len, 1)
        first_window = item_seq[: min(len(item_seq), max_len + 1)]
        examples.append(
            {
                "item_seq": first_window,
                "label_all_positions": True,
            }
        )

        for start in range(1, n_return_examples):
            window = item_seq[start : start + max_len + 1]
            examples.append(
                {
                    "item_seq": window,
                    "label_all_positions": False,
                }
            )
    return examples


def load_eval_examples(eval_json: Path, max_len: int) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for obj in load_jsonl(eval_json):
        history = _to_int_list(obj.get("history", []))
        if not history:
            continue
        examples.append(
            {
                "history": history[-max_len:],
                "target": int(obj["target"]),
            }
        )
    return examples


def infer_num_items(dataset_dir: Path, dataset_name: str) -> int:
    item2id_path = dataset_dir / f"{dataset_name}.item2id"
    item_count = 0
    if item2id_path.exists():
        with item2id_path.open("r", encoding="utf-8") as f:
            item_count = sum(1 for line in f if line.strip())

    max_item_id = -1
    for split in ("train", "valid", "test"):
        json_path = dataset_dir / f"{dataset_name}.{split}.jsonl"
        if not json_path.exists():
            continue
        for obj in load_jsonl(json_path):
            for item_id in obj.get("history", []):
                max_item_id = max(max_item_id, int(item_id))
            max_item_id = max(max_item_id, int(obj["target"]))

    return max(item_count, max_item_id + 1)


class SASRecTrainDataset(Dataset):
    def __init__(self, examples: List[Dict[str, Any]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.examples[index]


class SASRecEvalDataset(Dataset):
    def __init__(self, examples: List[Dict[str, Any]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.examples[index]


class SASRecTrainCollator:
    def __init__(self, max_len: int, num_items: int):
        self.max_len = max_len
        self.num_items = num_items

    def _build_example(self, item_seq: List[int], label_all_positions: bool) -> Dict[str, List[int]]:
        input_items = item_seq[:-1]
        target_items = item_seq[1:]
        seq_len = len(input_items)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len={self.max_len}.")

        pad_len = self.max_len - seq_len
        labels = [-100] * self.max_len
        if label_all_positions:
            labels[:seq_len] = target_items
        elif seq_len > 0:
            labels[seq_len - 1] = item_seq[-1]

        return {
            "input_ids": [item_id + 1 for item_id in input_items] + [0] * pad_len,
            "attention_mask": [1] * seq_len + [0] * pad_len,
            "labels_seq": labels,
        }

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        labels_seq: List[List[int]] = []

        for item in batch:
            built = self._build_example(
                [int(x) for x in item["item_seq"]],
                bool(item["label_all_positions"]),
            )

            input_ids.append(built["input_ids"])
            attention_mask.append(built["attention_mask"])
            labels_seq.append(built["labels_seq"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels_seq": torch.tensor(labels_seq, dtype=torch.long),
        }


class SASRecEvalCollator:
    def __init__(self, max_len: int):
        self.max_len = max_len

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        target_ids: List[int] = []

        for item in batch:
            history = [int(x) for x in item["history"]][-self.max_len:]
            seq_len = len(history)
            pad_len = self.max_len - seq_len
            target_ids.append(int(item["target"]))
            input_ids.append([item_id + 1 for item_id in history] + [0] * pad_len)
            attention_mask.append([1] * seq_len + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
        }
