import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def pad_or_truncate(sequence, max_len, PAD_TOKEN=0):
    if len(sequence) > max_len:
        return sequence[-max_len:]
    return [PAD_TOKEN] * (max_len - len(sequence)) + sequence


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


def process_parquet(file_path, mode, max_len, PAD_TOKEN=0):
    df = pd.read_parquet(file_path)
    df["sequence"] = df["history"].apply(lambda x: list(x)) + df["target"].apply(lambda x: [x])

    processed_data = []
    if mode == "train":
        for row in df.itertuples(index=False):
            sequence = row.sequence
            for i in range(1, len(sequence)):
                processed_data.append(
                    {
                        "history": pad_or_truncate(sequence[:i], max_len, PAD_TOKEN),
                        "target": sequence[i],
                    }
                )
    elif mode == "evaluation":
        for row in df.itertuples(index=False):
            sequence = row.sequence
            processed_data.append(
                {
                    "history": pad_or_truncate(sequence[:-1], max_len, PAD_TOKEN),
                    "target": sequence[-1],
                }
            )
    else:
        raise ValueError("Mode must be 'train' or 'evaluation'.")
    return processed_data


def process_jsonl(file_path, max_len, PAD_TOKEN=0):
    processed = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            history = [int(x) for x in obj.get("history", [])]
            target = int(obj.get("target"))
            processed.append(
                {
                    "history": pad_or_truncate(history, max_len, PAD_TOKEN),
                    "target": target,
                }
            )
    return processed


def process_instruction_jsonl(file_path):
    processed = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            processed.append(json.loads(line))
    return processed


def _offset_token_to_special_token(token_id: int, bases: List[int], level: int) -> str:
    semantic_value = token_id - bases[level] - 1
    return f"<SID_{level}_{semantic_value}>"


def _item_to_semantic_text(item_id_0based: int, item_to_code_map: Dict[int, List[int]], bases: List[int]) -> str:
    code_tokens = item_to_code_map[item_id_0based + 1]
    semantic_tokens = [
        _offset_token_to_special_token(token_id, bases, level)
        for level, token_id in enumerate(code_tokens)
    ]
    return " ".join(semantic_tokens)


def _history_to_semantic_input(
    history_ids: List[int],
    item_to_code_map: Dict[int, List[int]],
    bases: List[int],
    history_sep: str,
    add_prefix: bool,
) -> str:
    item_strings = [_item_to_semantic_text(item_id, item_to_code_map, bases) for item_id in history_ids]
    if add_prefix:
        item_strings = [f"{idx + 1}. {value}" for idx, value in enumerate(item_strings)]
    return history_sep.join(item_strings) if item_strings else "None"


def ensure_lcrec_instruction_jsonl(config: dict, mode: str) -> Path:
    output_path = Path(config[f"{mode}_instruction_json"])
    source_path = Path(config[f"{mode}_json"])
    if not source_path.is_file():
        raise FileNotFoundError(f"LCRec source jsonl not found: {source_path}")
    if output_path.is_file() and output_path.stat().st_mtime >= source_path.stat().st_mtime:
        return output_path

    item_to_code_map, _ = item2code(
        config["code_path"],
        config["vocab_sizes"],
        config["bases"],
    )
    model_params = config["model_params"]
    instruction_template = model_params.get(
        "instruction_template",
        "Predict the next item Semantic ID from the user's historical interactions.",
    )
    history_sep = model_params.get("history_sep", ", ")
    add_prefix = model_params.get("history_add_index_prefix", False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(source_path, "r", encoding="utf-8") as src, open(output_path, "w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            history = [int(x) for x in obj.get("history", [])]
            target = int(obj["target"])

            input_text = _history_to_semantic_input(
                history,
                item_to_code_map,
                config["bases"],
                history_sep,
                add_prefix,
            )
            output_text = _item_to_semantic_text(target, item_to_code_map, config["bases"])
            record = {
                "task": "seqrec",
                "instruction": instruction_template,
                "input": input_text,
                "output": output_text,
                "target_item": target,
                "target_semantic_id": output_text,
            }
            json.dump(record, dst, ensure_ascii=False)
            dst.write("\n")
    return output_path


class GenRecDataset(Dataset):
    def __init__(self, config: dict, mode: str):
        self.config = config
        self.mode = mode
        self.model_name = self.config["model_name"].upper()
        self.max_len = self.config["model_params"]["max_len"]

        if self.model_name == "LCREC":
            self.dataset_path = ensure_lcrec_instruction_jsonl(self.config, mode)
            self.data = process_instruction_jsonl(self.dataset_path)
        else:
            self.dataset_path = self.config[f"{mode}_json"]
            self.data = process_jsonl(self.dataset_path, self.max_len, PAD_TOKEN=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
