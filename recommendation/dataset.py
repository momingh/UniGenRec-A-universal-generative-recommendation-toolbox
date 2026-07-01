import sys
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.data_split import (
    build_eval_samples,
    build_prefix_train_samples,
    build_sequence_window_train_examples,
    split_from_interactions,
)

PREFIX_TARGET = "prefix_target"
SEQUENCE_AUTOREGRESSIVE = "sequence_autoregressive"
SUPPORTED_DATA_FORMATS = {PREFIX_TARGET, SEQUENCE_AUTOREGRESSIVE}


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


def _build_prefix_target_data(split_data, mode, max_len):
    if mode == "train":
        samples = build_prefix_train_samples(
            split_data.train_sequences,
            max_history_len=max_len,
        )
    elif mode == "valid":
        samples = build_eval_samples(split_data.valid_samples)
    elif mode == "test":
        samples = build_eval_samples(split_data.test_samples)
    else:
        raise ValueError(f"Unsupported dataset mode: {mode}")

    return [
        {
            "user": obj.get("user"),
            "history": truncate_sequence(
                [int(x) for x in obj.get("history", [])],
                max_len,
            ),
            "target": int(obj["target"]),
        }
        for obj in samples
    ]


def _build_sequence_autoregressive_data(split_data, mode, max_len):
    if mode == "train":
        return build_sequence_window_train_examples(
            split_data.train_sequences,
            max_len=max_len,
        )

    if mode == "valid":
        samples = split_data.valid_samples
    elif mode == "test":
        samples = split_data.test_samples
    else:
        raise ValueError(f"Unsupported dataset mode: {mode}")

    return [
        {"item_seq": [*sample.history[-max_len:], int(sample.target)]}
        for sample in samples
        if len(sample.history) > 0
    ]


class GenRecDataset(Dataset):
    def __init__(self, config: dict, mode: str):
        self.config = config
        self.mode = mode
        self.model_name = self.config["model_name"].upper()
        self.data_format = str(self.config.get("data_format", PREFIX_TARGET)).lower()
        if self.data_format not in SUPPORTED_DATA_FORMATS:
            raise ValueError(
                f"Unsupported data_format: {self.data_format}. "
                f"Supported values: {sorted(SUPPORTED_DATA_FORMATS)}"
            )
        self.max_len = self.config["model_params"]["max_len"]
        self.dataset_path = self.config["inter_json"]
        if not Path(self.dataset_path).is_file():
            raise FileNotFoundError(f"Interaction file not found: {self.dataset_path}")

        split_data = split_from_interactions(
            self.dataset_path,
            min_sequence_len=3,
            max_history_len=self.max_len,
        )

        if self.data_format == SEQUENCE_AUTOREGRESSIVE:
            self.data = _build_sequence_autoregressive_data(split_data, mode, self.max_len)
            for item in self.data:
                item["mode"] = mode
        elif self.data_format == PREFIX_TARGET:
            self.data = _build_prefix_target_data(split_data, mode, self.max_len)

        if not self.data:
            raise ValueError(
                f"No {mode} samples were built from {self.dataset_path} "
                f"for model {self.model_name} with data_format={self.data_format}."
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
