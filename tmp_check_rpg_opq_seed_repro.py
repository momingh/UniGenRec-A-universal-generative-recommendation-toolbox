#!/usr/bin/env python3
"""Run the original RPG PCA+OPQ path twice with fixed seeds and compare outputs."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


BASE = Path("/at2-data/mominghao/gen_rec/RPG_KDD2025/cache/AmazonReviews2014/Beauty/processed")
SENT_EMB = BASE / "text-embedding-3-large.sent_emb"
ID_MAPPING = BASE / "id_mapping.json"
ALL_ITEM_SEQS = BASE / "all_item_seqs.json"
SEM_IDS = BASE / "text-embedding-3-large_OPQ32,IVF1,PQ32x8.sem_ids"

SEED = 2024
N_THREADS = 32


def load_inputs() -> tuple[np.ndarray, dict, np.ndarray]:
    raw = np.fromfile(SENT_EMB, dtype=np.float32).reshape(-1, 3072)
    id_mapping = json.load(open(ID_MAPPING, "r", encoding="utf-8"))
    all_item_seqs = json.load(open(ALL_ITEM_SEQS, "r", encoding="utf-8"))

    item2id = id_mapping["item2id"]
    seqs = all_item_seqs.values() if isinstance(all_item_seqs, dict) else all_item_seqs
    items_for_training = set()
    for full_seq in seqs:
        for item in full_seq[:-2]:
            items_for_training.add(item)

    train_mask = np.zeros(len(item2id) - 1, dtype=bool)
    for item in items_for_training:
        train_mask[int(item2id[item]) - 1] = True
    return raw, id_mapping, train_mask


def run_pca(raw: np.ndarray, seed: int) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)
    pca = PCA(n_components=512, whiten=True, random_state=seed)
    reduced = pca.fit_transform(raw).astype(np.float32)
    print(
        "pca_explained_variance_ratio_sum=",
        float(np.sum(pca.explained_variance_ratio_)),
    )
    return reduced


def make_index(seed: int):
    import faiss

    faiss.omp_set_num_threads(N_THREADS)
    index = faiss.index_factory(512, "OPQ32,IVF1,PQ32x8", faiss.METRIC_INNER_PRODUCT)

    inner = faiss.downcast_index(index.index)
    inner.cp.seed = seed
    inner.pq.cp.seed = seed
    return faiss, index


def run_opq(reduced: np.ndarray, train_mask: np.ndarray, seed: int) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)
    faiss, index = make_index(seed)

    index.train(reduced[train_mask])
    index.add(reduced)

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


def compare_to_cached(codes: np.ndarray, id_mapping: dict) -> None:
    sem_ids = json.load(open(SEM_IDS, "r", encoding="utf-8"))
    id2item = id_mapping["id2item"]
    full = 0
    layer = np.zeros(codes.shape[1], dtype=np.int64)

    for row in range(codes.shape[0]):
        item = id2item[row + 1] if isinstance(id2item, list) else id2item[str(row + 1)]
        pred = codes[row].astype(int).tolist()
        gold = list(map(int, sem_ids[item]))
        full += pred == gold
        layer += np.array([a == b for a, b in zip(pred, gold)], dtype=np.int64)

    n = codes.shape[0]
    print(f"cached_full_match={full}/{n} rate={full / n:.8f}")
    print("cached_per_layer_match_rate=", (layer / n).round(6).tolist())


def main() -> None:
    raw, id_mapping, train_mask = load_inputs()
    print("raw_shape=", raw.shape)
    print("train_items=", int(train_mask.sum()), "of", len(train_mask))
    print("seed=", SEED, "faiss_threads=", N_THREADS)

    print("\n[PCA run 1]")
    pca1 = run_pca(raw, SEED)
    print("[PCA run 2]")
    pca2 = run_pca(raw, SEED)
    print("pca_equal=", bool(np.array_equal(pca1, pca2)))
    print("pca_max_abs_diff=", float(np.max(np.abs(pca1 - pca2))))

    print("\n[OPQ run 1]")
    codes1 = run_opq(pca1, train_mask, SEED)
    print("[OPQ run 2]")
    codes2 = run_opq(pca1, train_mask, SEED)
    print("opq_equal_same_pca=", bool(np.array_equal(codes1, codes2)))
    print("opq_diff_entries_same_pca=", int(np.sum(codes1 != codes2)))

    print("\n[OPQ run 3 on PCA run 2]")
    codes3 = run_opq(pca2, train_mask, SEED)
    print("full_pipeline_equal=", bool(np.array_equal(codes1, codes3)))
    print("full_pipeline_diff_entries=", int(np.sum(codes1 != codes3)))

    print("\n[compare run 1 to cached original .sem_ids]")
    compare_to_cached(codes1, id_mapping)


if __name__ == "__main__":
    main()
