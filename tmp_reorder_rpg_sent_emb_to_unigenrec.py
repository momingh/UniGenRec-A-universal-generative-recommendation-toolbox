#!/usr/bin/env python3
"""
Temporarily convert the original RPG raw .sent_emb file into a standard .npy
file that UniGenRec's quantization/main.py can load, then optionally apply the
same PCA whitening used by the original RPG tokenizer.

RPG layout:
  text-embedding-3-large.sent_emb is raw float32 with no npy header.
  sent_emb[row] corresponds to RPG id_mapping item id row + 1 because id 0 is
  [PAD].

Default output layout:
  Keep RPG's original row order. This is useful for checking whether a new OPQ
  run reproduces the original RPG .sem_ids.

Optional UniGenRec layout:
  quantization/main.py reads:
    datasets/<dataset>/embeddings/<dataset>.emb-text-<embedding_model>.npy
  row i must correspond to datasets/<dataset>/<dataset>.item2id item id i.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_RPG_SENT_EMB = (
    "/at2-data/mominghao/gen_rec/RPG_KDD2025/cache/AmazonReviews2014/"
    "Beauty/processed/text-embedding-3-large.sent_emb"
)
DEFAULT_RPG_ID_MAPPING = (
    "/at2-data/mominghao/gen_rec/RPG_KDD2025/cache/AmazonReviews2014/"
    "Beauty/processed/id_mapping.json"
)
DEFAULT_UNIGENREC_ITEM2ID = (
    "/at2-data/mominghao/gen_rec/UniGenRec-A-universal-generative-recommendation-toolbox/"
    "datasets/Beauty/Beauty.item2id"
)
DEFAULT_OUTPUT = (
    "/at2-data/mominghao/gen_rec/UniGenRec-A-universal-generative-recommendation-toolbox/"
    "datasets/Beauty/embeddings/Beauty.emb-text-text-embedding-3-large-rpg-original-order-pca512.npy"
)


def load_rpg_item2id(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if "item2id" in obj:
        item2id = {str(item): int(idx) for item, idx in obj["item2id"].items()}
    elif "id2item" in obj:
        item2id = {
            str(item): int(idx)
            for idx, item in obj["id2item"].items()
            if str(item) != "[PAD]"
        }
    else:
        raise ValueError(f"{path} must contain item2id or id2item.")

    pad_id = item2id.pop("[PAD]", None)
    if pad_id not in (None, 0):
        raise ValueError(f"Unexpected RPG [PAD] id: {pad_id}")

    return item2id


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
            item, idx = parts
            item2id[item] = int(idx)

    ids = sorted(item2id.values())
    if ids != list(range(len(ids))):
        raise ValueError(f"UniGenRec item ids are not contiguous 0..N-1 in {path}.")

    return item2id


def load_raw_sent_emb(path: Path, dim: int) -> np.ndarray:
    flat = np.fromfile(path, dtype=np.float32)
    if flat.size % dim != 0:
        raise ValueError(
            f"{path} has {flat.size} float32 values, not divisible by dim={dim}."
        )
    return flat.reshape(-1, dim)


def reorder_embeddings(
    rpg_emb: np.ndarray,
    rpg_item2id: dict[str, int],
    unigenrec_item2id: dict[str, int],
) -> np.ndarray:
    rpg_items = set(rpg_item2id)
    unigenrec_items = set(unigenrec_item2id)
    if rpg_items != unigenrec_items:
        only_rpg = sorted(rpg_items - unigenrec_items)[:10]
        only_uni = sorted(unigenrec_items - rpg_items)[:10]
        raise ValueError(
            "RPG and UniGenRec item sets differ. "
            f"only_rpg(sample)={only_rpg}, only_unigenrec(sample)={only_uni}"
        )

    expected_rpg_rows = len(rpg_item2id)
    if rpg_emb.shape[0] != expected_rpg_rows:
        raise ValueError(
            f"RPG embedding rows ({rpg_emb.shape[0]}) != RPG real item count "
            f"({expected_rpg_rows})."
        )

    reordered = np.empty((len(unigenrec_item2id), rpg_emb.shape[1]), dtype=np.float32)
    for item, uni_idx in unigenrec_item2id.items():
        rpg_idx = rpg_item2id[item]
        if rpg_idx <= 0:
            raise ValueError(f"RPG item {item} has invalid id {rpg_idx}; expected > 0.")
        reordered[uni_idx] = rpg_emb[rpg_idx - 1]

    return reordered


def apply_rpg_pca(embeddings: np.ndarray, pca_dim: int) -> tuple[np.ndarray, float]:
    if pca_dim <= 0:
        return embeddings, 1.0
    if embeddings.shape[1] <= pca_dim:
        return embeddings, 1.0

    from sklearn.decomposition import PCA

    # Match original RPG_KDD2025 tokenizer.py:
    #   PCA(n_components=sent_emb_pca, whiten=True).fit_transform(sent_embs)
    pca = PCA(n_components=pca_dim, whiten=True)
    reduced = pca.fit_transform(embeddings).astype(np.float32)
    explained = float(np.sum(pca.explained_variance_ratio_))
    return reduced, explained


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert original RPG .sent_emb into a UniGenRec-readable .npy file."
    )
    parser.add_argument("--rpg-sent-emb", default=DEFAULT_RPG_SENT_EMB)
    parser.add_argument("--rpg-id-mapping", default=DEFAULT_RPG_ID_MAPPING)
    parser.add_argument("--unigenrec-item2id", default=DEFAULT_UNIGENREC_ITEM2ID)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--layout",
        choices=["rpg-original", "unigenrec"],
        default="rpg-original",
        help=(
            "rpg-original keeps RPG .sent_emb row order; unigenrec reorders rows "
            "to UniGenRec Beauty.item2id order."
        ),
    )
    parser.add_argument("--dim", type=int, default=3072)
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=512,
        help="Apply RPG-style PCA whitening to this dimension. Use <=0 to disable.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rpg_sent_emb_path = Path(args.rpg_sent_emb)
    rpg_id_mapping_path = Path(args.rpg_id_mapping)
    output_path = Path(args.output)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")

    rpg_emb = load_raw_sent_emb(rpg_sent_emb_path, args.dim)
    rpg_item2id = load_rpg_item2id(rpg_id_mapping_path)
    if rpg_emb.shape[0] != len(rpg_item2id):
        raise ValueError(
            f"RPG embedding rows ({rpg_emb.shape[0]}) != RPG real item count "
            f"({len(rpg_item2id)})."
        )

    if args.layout == "unigenrec":
        unigenrec_item2id = load_unigenrec_item2id(Path(args.unigenrec_item2id))
        arranged = reorder_embeddings(rpg_emb, rpg_item2id, unigenrec_item2id)
    else:
        arranged = rpg_emb

    saved_emb, explained = apply_rpg_pca(arranged, args.pca_dim)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, saved_emb)

    print(f"loaded_rpg_shape={rpg_emb.shape}")
    print(f"layout={args.layout}")
    print(f"arranged_shape={arranged.shape}")
    print(f"pca_dim={args.pca_dim}")
    print(f"pca_explained_variance_ratio_sum={explained:.6f}")
    print(f"saved_shape={saved_emb.shape}")
    print(f"saved_to={output_path}")

    if args.layout == "unigenrec":
        unigenrec_item2id = load_unigenrec_item2id(Path(args.unigenrec_item2id))
        for item, uni_idx in list(sorted(unigenrec_item2id.items(), key=lambda kv: kv[1]))[:5]:
            rpg_row = rpg_item2id[item] - 1
            max_abs_diff = float(np.max(np.abs(arranged[uni_idx] - rpg_emb[rpg_row])))
            print(
                f"check item={item} uni_row={uni_idx} rpg_row={rpg_row} "
                f"pre_pca_max_abs_diff={max_abs_diff:.6g}"
            )
    else:
        for item, rpg_idx in list(sorted(rpg_item2id.items(), key=lambda kv: kv[1]))[:5]:
            rpg_row = rpg_idx - 1
            max_abs_diff = float(np.max(np.abs(arranged[rpg_row] - rpg_emb[rpg_row])))
            print(
                f"check item={item} rpg_id={rpg_idx} rpg_row={rpg_row} "
                f"pre_pca_max_abs_diff={max_abs_diff:.6g}"
            )


if __name__ == "__main__":
    main()
