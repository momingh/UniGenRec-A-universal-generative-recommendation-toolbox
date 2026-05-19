import json
from typing import Iterator

import numpy as np
from torch.utils.data import Dataset


def truncate_sequence(sequence, max_len):
    if len(sequence) > max_len:
        return sequence[-max_len:]
    return sequence


def item2code(code_path, vocab_sizes, bases):
    data = np.load(code_path, allow_pickle=True)
    mat = np.vstack(data) if data.dtype == object else data

    num_levels = len(vocab_sizes)
    assert mat.shape[1] == num_levels, f"Expect {num_levels} columns in codebook, got {mat.shape[1]}"

    item_to_code = {}
    code_to_item = {}

    for index, row in enumerate(mat):
        code_values = [int(c) for c in row]
        for i, code_val in enumerate(code_values):
            if not (0 <= code_val < vocab_sizes[i]):
                raise ValueError(
                    f"Out-of-range code {code_val} at index {i} for row {row} with vocab_sizes={vocab_sizes}"
                )

        tokens = [code_val + bases[i] + 1 for i, code_val in enumerate(code_values)]
        item_id = index + 1
        item_to_code[item_id] = tokens
        code_to_item[tuple(tokens)] = item_id

    return item_to_code, code_to_item


def _load_jsonl(file_path) -> Iterator[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def process_jsonl(file_path, max_len):
    return [
        {
            "history": truncate_sequence(
                [int(x) for x in obj.get("history", [])],
                max_len,
            ),
            "target": int(obj["target"]),
        }
        for obj in _load_jsonl(file_path)
    ]


def process_rpg_jsonl(file_path, mode, max_len):
    if mode == "train":
        return process_rpg_train_jsonl(file_path, max_len)

    return [
        {"item_seq": [*map(int, obj.get("history", [])), int(obj["target"])]}
        for obj in _load_jsonl(file_path)
        if len(obj.get("history", [])) > 0
    ]


def _expand_rpg_train_sequence(item_seq, max_len):
    n_return_examples = max(len(item_seq) - max_len, 1)
    first_window = item_seq[: min(len(item_seq), max_len + 1)]
    examples = [{"item_seq": first_window, "label_all_positions": True}]

    for start in range(1, n_return_examples):
        examples.append(
            {
                "item_seq": item_seq[start : start + max_len + 1],
                "label_all_positions": False,
            }
        )
    return examples


def process_rpg_train_jsonl(file_path, max_len):
    user_to_seq = {}
    user_order = []
    userless_seqs = []

    for obj in _load_jsonl(file_path):
        history = [int(x) for x in obj.get("history", [])]
        target = int(obj["target"])
        item_seq = history + [target]
        if len(item_seq) < 2:
            continue

        user = obj.get("user")
        if user is None:
            userless_seqs.append(item_seq)
            continue

        if user not in user_to_seq:
            user_to_seq[user] = item_seq
            user_order.append(user)
            continue

        # train.jsonl is written in per-user prefix-target order. This restores
        # the original train sequence while still detecting shuffled records.
        expected_history = user_to_seq[user][-len(history):] if history else []
        if expected_history != history:
            raise ValueError(
                f"RPG train records for user {user} are not in prefix order; "
                "cannot restore the full training sequence from train.jsonl."
            )
        user_to_seq[user].append(target)

    examples = []
    for item_seq in [*[user_to_seq[user] for user in user_order], *userless_seqs]:
        examples.extend(_expand_rpg_train_sequence(item_seq, max_len))
    return examples


class GenRecDataset(Dataset):
    def __init__(self, config: dict, mode: str):
        self.config = config
        self.mode = mode
        self.model_name = self.config["model_name"].upper()
        self.max_len = self.config["model_params"]["max_len"]
        self.dataset_path = self.config[f"{mode}_json"]
        if self.model_name == "RPG":
            self.data = process_rpg_jsonl(self.dataset_path, mode, self.max_len)
            for item in self.data:
                item["mode"] = mode
        else:
            self.data = process_jsonl(self.dataset_path, self.max_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
