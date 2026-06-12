#!/usr/bin/env python3
"""Convert original RPG .sem_ids into UniGenRec codebook row order.

Input .sem_ids format:
  {"raw_item_id": [code0, code1, ...]}

Output codebook.npy format used by UniGenRec recommendation:
  np.ndarray shape (n_items, code_len), where row i is UniGenRec item id i.

Output codebook.json format used by UniGenRec quantization artifacts:
  {"0": "<L0_code0> <L1_code1> ...", ...}

Important mapping:
  output key i is UniGenRec's 0-based item id from datasets/Beauty/Beauty.item2id.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np


DEFAULT_SEM_IDS = (
    "/at2-data/mominghao/gen_rec/RPG_KDD2025/cache/AmazonReviews2014/Beauty/"
    "processed/text-embedding-3-large_OPQ32,IVF1,PQ32x8.sem_ids"
)
DEFAULT_ITEM2ID = (
    "/at2-data/mominghao/gen_rec/UniGenRec-A-universal-generative-recommendation-toolbox/"
    "datasets/Beauty/Beauty.item2id"
)
DEFAULT_OUTPUT = (
    "/at2-data/mominghao/gen_rec/UniGenRec-A-universal-generative-recommendation-toolbox/"
    "datasets/Beauty/codebooks/Beauty.text.opq.codebook.json"
)
DEFAULT_NPY_OUTPUT = (
    "/at2-data/mominghao/gen_rec/UniGenRec-A-universal-generative-recommendation-toolbox/"
    "datasets/Beauty/codebooks/Beauty.text.opq.npy"
)


def load_unigenrec_item2id(path: Path) -> dict[str, int]:
    item2id: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Bad item2id line {line_no} in {path}: {line!r}")
            item, item_id = parts
            item2id[item] = int(item_id)

    ids = sorted(item2id.values())
    expected_ids = list(range(len(ids)))
    if ids != expected_ids:
        raise ValueError(f"UniGenRec item ids are not contiguous 0..N-1 in {path}.")
    return item2id


def load_sem_ids(path: Path) -> dict[str, list[int]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(item): [int(code) for code in codes] for item, codes in raw.items()}


def validate_inputs(
    sem_ids: dict[str, list[int]],
    item2id: dict[str, int],
    code_len: int,
    codebook_size: int,
) -> None:
    sem_items = set(sem_ids)
    uni_items = set(item2id)
    if sem_items != uni_items:
        only_sem = sorted(sem_items - uni_items)[:20]
        only_uni = sorted(uni_items - sem_items)[:20]
        raise ValueError(
            "Item set mismatch between .sem_ids and UniGenRec item2id. "
            f"only_sem(sample)={only_sem}, only_unigenrec(sample)={only_uni}"
        )

    for item, codes in sem_ids.items():
        if len(codes) != code_len:
            raise ValueError(f"Item {item} has code length {len(codes)}, expected {code_len}.")
        bad = [code for code in codes if code < 0 or code >= codebook_size]
        if bad:
            raise ValueError(
                f"Item {item} has out-of-range codes {bad[:10]}, "
                f"expected 0..{codebook_size - 1}."
            )


def convert_to_codebook_json(
    sem_ids: dict[str, list[int]],
    item2id: dict[str, int],
    code_len: int,
) -> dict[str, str]:
    output: dict[str, str] = {}
    for item, uni_id in sorted(item2id.items(), key=lambda kv: kv[1]):
        codes = sem_ids[item]
        output[str(uni_id)] = " ".join(f"<L{level}_{codes[level]}>" for level in range(code_len))
    return output


def convert_to_codebook_matrix(
    sem_ids: dict[str, list[int]],
    item2id: dict[str, int],
    code_len: int,
) -> np.ndarray:
    matrix = np.empty((len(item2id), code_len), dtype=np.int64)
    for item, uni_id in item2id.items():
        matrix[uni_id] = np.asarray(sem_ids[item], dtype=np.int64)
    return matrix


def atomic_write_json(path: Path, obj: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as f:
        tmp_name = f.name
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp_name, path)


def atomic_write_npy(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as f:
        tmp_name = f.name
        np.save(f, matrix)
    os.replace(tmp_name, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace UniGenRec Beauty.text.opq codebook artifacts using original RPG .sem_ids."
    )
    parser.add_argument("--sem-ids", default=DEFAULT_SEM_IDS)
    parser.add_argument("--item2id", default=DEFAULT_ITEM2ID)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--npy-output", default=DEFAULT_NPY_OUTPUT)
    parser.add_argument("--code-len", type=int, default=32)
    parser.add_argument("--codebook-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sem_path = Path(args.sem_ids)
    item2id_path = Path(args.item2id)
    output_path = Path(args.output)
    npy_output_path = Path(args.npy_output)

    sem_ids = load_sem_ids(sem_path)
    item2id = load_unigenrec_item2id(item2id_path)
    validate_inputs(
        sem_ids=sem_ids,
        item2id=item2id,
        code_len=args.code_len,
        codebook_size=args.codebook_size,
    )
    output = convert_to_codebook_json(
        sem_ids=sem_ids,
        item2id=item2id,
        code_len=args.code_len,
    )
    matrix = convert_to_codebook_matrix(
        sem_ids=sem_ids,
        item2id=item2id,
        code_len=args.code_len,
    )
    atomic_write_json(output_path, output)
    atomic_write_npy(npy_output_path, matrix)

    print(f"loaded_sem_ids={len(sem_ids)} from {sem_path}")
    print(f"loaded_item2id={len(item2id)} from {item2id_path}")
    print(f"saved_json={output_path}")
    print(f"saved_npy={npy_output_path}")
    print(f"rows={len(output)} code_len={args.code_len} codebook_size={args.codebook_size}")
    print(f"npy_shape={matrix.shape} npy_dtype={matrix.dtype}")
    for item, uni_id in list(sorted(item2id.items(), key=lambda kv: kv[1]))[:5]:
        print(
            f"check item={item} uni_row={uni_id} "
            f"codes={matrix[uni_id].tolist()[:8]} json={output[str(uni_id)][:120]}"
        )


if __name__ == "__main__":
    main()
