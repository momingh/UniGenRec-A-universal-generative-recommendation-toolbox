#!/usr/bin/env python3
"""Check whether rerunning the original RPG PCA+OPQ path reproduces cached .sem_ids."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


BASE = Path("/at2-data/mominghao/gen_rec/RPG_KDD2025/cache/AmazonReviews2014/Beauty/processed")
SENT_EMB = BASE / "text-embedding-3-large.sent_emb"
ID_MAPPING = BASE / "id_mapping.json"
ALL_ITEM_SEQS = BASE / "all_item_seqs.json"
SEM_IDS = BASE / "text-embedding-3-large_OPQ32,IVF1,PQ32x8.sem_ids"


def build_train_mask(id_mapping: dict) -> np.ndarray:
    all_item_seqs = json.load(open(ALL_ITEM_SEQS, "r", encoding="utf-8"))
    item2id = id_mapping["item2id"]
    items_for_training = set()
    seqs = all_item_seqs.values() if isinstance(all_item_seqs, dict) else all_item_seqs
    for full_seq in seqs:
        for item in full_seq[:-2]:
            items_for_training.add(item)

    mask = np.zeros(len(item2id) - 1, dtype=bool)
    for item in items_for_training:
        mask[int(item2id[item]) - 1] = True
    return mask


def run_opq(sent_embs: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    import faiss

    faiss.omp_set_num_threads(32)
    index = faiss.index_factory(512, "OPQ32,IVF1,PQ32x8", faiss.METRIC_INNER_PRODUCT)
    index.train(sent_embs[train_mask])
    index.add(sent_embs)

    ivf_index = faiss.downcast_index(index.index)
    invlists = faiss.extract_index_ivf(ivf_index).invlists
    list_size = invlists.list_size(0)
    packed = faiss.rev_swig_ptr(invlists.get_codes(0), list_size * invlists.code_size)
    packed = packed.reshape(-1, invlists.code_size)

    codes = np.empty((packed.shape[0], 32), dtype=np.int64)
    for row_idx, u8code in enumerate(packed):
        u8code = np.ascontiguousarray(u8code)
        reader = faiss.BitstringReader(faiss.swig_ptr(u8code), packed.shape[1])
        for i in range(32):
            codes[row_idx, i] = reader.read(8)
    return codes


def compare(codes: np.ndarray, id_mapping: dict) -> None:
    sem_ids = json.load(open(SEM_IDS, "r", encoding="utf-8"))
    id2item = id_mapping["id2item"]
    full = 0
    prefix4 = 0
    layer = np.zeros(codes.shape[1], dtype=np.int64)
    samples = []

    for row in range(codes.shape[0]):
        item = id2item[row + 1] if isinstance(id2item, list) else id2item[str(row + 1)]
        pred = codes[row].astype(int).tolist()
        gold = list(map(int, sem_ids[item]))
        full += pred == gold
        prefix4 += pred[:4] == gold[:4]
        layer += np.array([a == b for a, b in zip(pred, gold)], dtype=np.int64)
        if len(samples) < 5:
            samples.append((row, item, pred[:8], gold[:8], pred == gold))

    n = codes.shape[0]
    print(f"full_match={full}/{n} rate={full / n:.8f}")
    print(f"prefix4_match={prefix4}/{n} rate={prefix4 / n:.8f}")
    print("per_layer_match_rate=", (layer / n).round(6).tolist())
    for sample in samples:
        print("sample", sample)


def main() -> None:
    id_mapping = json.load(open(ID_MAPPING, "r", encoding="utf-8"))
    raw = np.fromfile(SENT_EMB, dtype=np.float32).reshape(-1, 3072)
    train_mask = build_train_mask(id_mapping)
    print("raw_shape=", raw.shape)
    print("train_items=", int(train_mask.sum()), "of", len(train_mask))

    pca = PCA(n_components=512, whiten=True)
    reduced = pca.fit_transform(raw)
    print("pca_shape=", reduced.shape, "dtype=", reduced.dtype)
    print("pca_explained_variance_ratio_sum=", float(np.sum(pca.explained_variance_ratio_)))

    codes = run_opq(reduced, train_mask)
    print("codes_shape=", codes.shape)
    compare(codes, id_mapping)


if __name__ == "__main__":
    main()
