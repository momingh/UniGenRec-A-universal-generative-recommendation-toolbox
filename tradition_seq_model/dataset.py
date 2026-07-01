import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.data_split import (
    build_prefix_train_samples,
    build_sequence_window_train_examples,
    split_from_interactions,
)

PREFIX_TARGET = "prefix_target"
SEQUENCE_AUTOREGRESSIVE = "sequence_autoregressive"
SUPPORTED_DATA_FORMATS = {PREFIX_TARGET, SEQUENCE_AUTOREGRESSIVE}


def _normalize_data_format(data_format: str) -> str:
    normalized = str(data_format).lower()
    if normalized not in SUPPORTED_DATA_FORMATS:
        raise ValueError(
            f"Unsupported data_format: {normalized}. "
            f"Supported values: {sorted(SUPPORTED_DATA_FORMATS)}"
        )
    return normalized


def _build_eval_examples(samples, max_len: int) -> List[Dict[str, Any]]:
    return [
        {
            "history": [int(item) for item in sample.history[-max_len:]],
            "filter_items": [int(item) for item in sample.history],
            "target": int(sample.target),
        }
        for sample in samples
        if len(sample.history) > 0
    ]


def _build_sequence_autoregressive_train_examples(split_data, max_len: int) -> List[Dict[str, Any]]:
    train_examples: List[Dict[str, Any]] = []
    for user_sequence in split_data.train_sequences:
        item_seq = [int(item) for item in user_sequence.items]
        for example in build_sequence_window_train_examples([user_sequence], max_len=max_len):
            example["exclude_items"] = item_seq
            train_examples.append(example)
    return train_examples


def _build_prefix_target_train_examples(split_data, max_len: int) -> List[Dict[str, Any]]:
    user_train_items = {
        int(user_sequence.user): [int(item) for item in user_sequence.items]
        for user_sequence in split_data.train_sequences
    }
    train_examples: List[Dict[str, Any]] = []

    for sample in build_prefix_train_samples(split_data.train_sequences, max_history_len=max_len):
        history = [int(item) for item in sample.get("history", [])]
        target = int(sample["target"])
        item_seq = [*history, target]
        if len(item_seq) < 2:
            continue

        raw_user = sample.get("user")
        exclude_items = item_seq
        if raw_user is not None:
            exclude_items = user_train_items.get(int(raw_user), item_seq)
        train_examples.append(
            {
                "item_seq": item_seq,
                "exclude_items": exclude_items,
                "label_all_positions": False,
            }
        )
    return train_examples


def load_sasrec_examples(
    inter_json: Path,
    max_len: int,
    data_format: str = SEQUENCE_AUTOREGRESSIVE,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[int, List[int]]]:
    data_format = _normalize_data_format(data_format)
    split_data = split_from_interactions(
        inter_json,
        min_sequence_len=3,
        max_history_len=None,
    )

    if data_format == SEQUENCE_AUTOREGRESSIVE:
        train_examples = _build_sequence_autoregressive_train_examples(split_data, max_len)
    else:
        train_examples = _build_prefix_target_train_examples(split_data, max_len)

    valid_examples = _build_eval_examples(split_data.valid_samples, max_len)
    test_examples = _build_eval_examples(split_data.test_samples, max_len)

    return train_examples, valid_examples, test_examples, split_data.seen_items_by_user


def load_sequence_autoregressive_examples(
    inter_json: Path,
    max_len: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[int, List[int]]]:
    return load_sasrec_examples(
        inter_json=inter_json,
        max_len=max_len,
        data_format=SEQUENCE_AUTOREGRESSIVE,
    )


def infer_num_items(dataset_dir: Path, dataset_name: str) -> int:
    item2id_path = dataset_dir / f"{dataset_name}.item2id"
    item_count = 0
    if item2id_path.exists():
        with item2id_path.open("r", encoding="utf-8") as f:
            item_count = sum(1 for line in f if line.strip())

    max_item_id = -1
    inter_json_path = dataset_dir / f"{dataset_name}.inter.json"
    if inter_json_path.exists():
        with inter_json_path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)
        for raw_interactions in raw_data.values():
            for raw_interaction in raw_interactions:
                if isinstance(raw_interaction, dict):
                    item_id = int(raw_interaction["item"])
                elif isinstance(raw_interaction, (list, tuple)):
                    item_id = int(raw_interaction[0])
                else:
                    item_id = int(raw_interaction)
                max_item_id = max(max_item_id, item_id)

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

    def _sample_negative(self, excluded_items: set[int]) -> int:
        if self.num_items <= 0:
            raise ValueError("num_items must be positive for negative sampling.")
        if len(excluded_items) >= self.num_items:
            return random.randrange(self.num_items)

        sampled_item = random.randrange(self.num_items)
        while sampled_item in excluded_items:
            sampled_item = random.randrange(self.num_items)
        return sampled_item

    def _build_example(
        self,
        item_seq: List[int],
        exclude_items: List[int],
        label_all_positions: bool,
    ) -> Dict[str, List[int]]:
        input_items = item_seq[:-1]
        target_items = item_seq[1:]
        seq_len = len(input_items)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len={self.max_len}.")

        pad_len = self.max_len - seq_len
        labels = [-100] * self.max_len
        negative_labels = [-100] * self.max_len
        excluded_item_set = set(exclude_items)
        if label_all_positions:
            labels[pad_len:] = target_items
            for index in range(seq_len):
                negative_labels[pad_len + index] = self._sample_negative(excluded_item_set)
        elif seq_len > 0:
            label_index = pad_len + seq_len - 1
            labels[label_index] = item_seq[-1]
            negative_labels[label_index] = self._sample_negative(excluded_item_set)

        return {
            "input_ids": [0] * pad_len + [item_id + 1 for item_id in input_items],
            "attention_mask": [0] * pad_len + [1] * seq_len,
            "labels_seq": labels,
            "negative_labels_seq": negative_labels,
        }

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        labels_seq: List[List[int]] = []
        negative_labels_seq: List[List[int]] = []

        for item in batch:
            built = self._build_example(
                [int(x) for x in item["item_seq"]],
                [int(x) for x in item.get("exclude_items", item["item_seq"])],
                bool(item["label_all_positions"]),
            )

            input_ids.append(built["input_ids"])
            attention_mask.append(built["attention_mask"])
            labels_seq.append(built["labels_seq"])
            negative_labels_seq.append(built["negative_labels_seq"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels_seq": torch.tensor(labels_seq, dtype=torch.long),
            "negative_labels_seq": torch.tensor(negative_labels_seq, dtype=torch.long),
        }


class SASRecEvalCollator:
    def __init__(self, max_len: int):
        self.max_len = max_len

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        target_ids: List[int] = []
        filter_items: List[List[int]] = []

        for item in batch:
            history = [int(x) for x in item["history"]][-self.max_len:]
            seq_len = len(history)
            pad_len = self.max_len - seq_len
            target_ids.append(int(item["target"]))
            input_ids.append([0] * pad_len + [item_id + 1 for item_id in history])
            attention_mask.append([0] * pad_len + [1] * seq_len)
            filter_items.append([int(x) for x in item.get("filter_items", history)])

        max_filter_len = max((len(items) for items in filter_items), default=0)
        padded_filter_items = [
            items + [-1] * (max_filter_len - len(items))
            for items in filter_items
        ]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "filter_item_ids": torch.tensor(padded_filter_items, dtype=torch.long),
        }
