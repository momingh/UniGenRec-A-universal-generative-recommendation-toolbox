import logging
import math

import numpy as np
import torch
from torch import nn

from .common.model_utils import as_float32_numpy


class OPQ(nn.Module):
    """
    基于官方 RPG 的 FAISS OPQ semantic ID 生成器。

    官方实现使用 index_factory 创建 OPQ + IVF1 + PQ 组合索引：
    OPQ{n_codebook},IVF1,PQ{n_codebook}x{bits}。
    本地仍保留 Trainer 需要的 fit/get_codes 接口。
    """

    fit_on_cpu = True

    def __init__(self, config: dict, input_size: int, item_embeddings=None):
        super().__init__()

        model_params = config.get("opq", {}).get("model_params", config)

        self.config = config
        self.input_size = int(input_size)
        self.n_codebook = int(model_params["n_codebook"])
        self.codebook_size = int(model_params["codebook_size"])

        if self.n_codebook <= 0:
            raise ValueError("OPQ n_codebook must be positive.")
        if self.codebook_size <= 0:
            raise ValueError("OPQ codebook_size must be positive.")

        self.n_codebook_bits = self._get_codebook_bits(self.codebook_size)
        if self.n_codebook_bits > 16:
            raise ValueError("FAISS PQ supports at most 16 bits per sub-codebook.")
        if self.input_size % self.n_codebook != 0:
            raise ValueError(
                f"OPQ/PQ requires input_size ({self.input_size}) to be divisible "
                f"by n_codebook ({self.n_codebook})."
            )

        self.faiss_omp_num_threads = int(model_params["faiss_omp_num_threads"])
        self.index_factory = (
            f"OPQ{self.n_codebook},IVF1,PQ{self.n_codebook}x{self.n_codebook_bits}"
        )

        self.index = None
        self.fitted = False
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def is_iterative(self) -> bool:
        return False

    def _get_codebook_bits(self, codebook_size):
        bits = math.log2(codebook_size)
        if not bits.is_integer() or bits < 0:
            raise ValueError("Invalid value for codebook_size.")
        return int(bits)

    def fit(self, xs, train_mask=None):
        import faiss

        data = as_float32_numpy(xs)
        train_data = data if train_mask is None else data[np.asarray(train_mask, dtype=bool)]

        if train_data.shape[0] < self.codebook_size:
            self.logger.warning(
                "OPQ training samples (%d) are fewer than codebook_size (%d); "
                "FAISS may fail to train the PQ codebooks.",
                train_data.shape[0],
                self.codebook_size,
            )

        if self.faiss_omp_num_threads > 0:
            faiss.omp_set_num_threads(self.faiss_omp_num_threads)

        self.logger.info(
            "Training FAISS index_factory=%s: train_vectors=%d, dim=%d",
            self.index_factory,
            train_data.shape[0],
            self.input_size,
        )

        index = faiss.index_factory(
            self.input_size,
            self.index_factory,
            faiss.METRIC_INNER_PRODUCT,
        )

        self.logger.info("Training OPQ/IVF1/PQ index.")
        index.train(train_data)

        self.index = index
        self.fitted = True
        self.logger.info("FAISS OPQ/IVF1/PQ training completed.")
        return self

    def forward(self, xs=None, x=None):
        if xs is None:
            xs = x
        if xs is None:
            raise ValueError("OPQ.forward requires xs or x.")
        if not self.fitted:
            self.fit(xs)
        codes = self.get_codes(xs)
        device = xs.device if isinstance(xs, torch.Tensor) else "cpu"
        zero = torch.tensor(0.0, device=device)
        return xs, zero, codes

    def compute_loss(self, outputs=None, *args, batch_data=None, xs=None, **kwargs):
        target = xs if xs is not None else batch_data
        device = target.device if isinstance(target, torch.Tensor) else "cpu"
        zero = torch.tensor(0.0, device=device)
        return {"loss_total": zero}

    def _extract_ivf_pq_codes(self, faiss, index, expected_count):
        ivf_index = faiss.downcast_index(index.index if hasattr(index, "index") else index)
        invlists = faiss.extract_index_ivf(ivf_index).invlists
        list_size = invlists.list_size(0)
        if list_size != expected_count:
            raise RuntimeError(
                "OPQ IVF1 list size mismatch: "
                f"expected {expected_count}, got {list_size}."
            )
        packed_codes = faiss.rev_swig_ptr(
            invlists.get_codes(0),
            list_size * invlists.code_size,
        )
        return packed_codes.reshape(-1, invlists.code_size)

    def _unpack_pq_codes(self, faiss, packed_codes):
        n_vectors = packed_codes.shape[0]
        n_bytes = packed_codes.shape[1]
        codes = np.empty((n_vectors, self.n_codebook), dtype=np.int64)

        for row_idx, packed in enumerate(packed_codes):
            packed = np.ascontiguousarray(packed)
            reader = faiss.BitstringReader(faiss.swig_ptr(packed), n_bytes)
            for level in range(self.n_codebook):
                codes[row_idx, level] = reader.read(self.n_codebook_bits)

        return codes

    def _encode_with_index_factory(self, faiss, data):
        index = faiss.clone_index(self.index)

        index.add(data)

        packed_codes = self._extract_ivf_pq_codes(
            faiss=faiss,
            index=index,
            expected_count=data.shape[0],
        )
        return self._unpack_pq_codes(faiss, packed_codes)

    @torch.no_grad()
    def get_codes(self, xs):
        import faiss

        if not self.fitted or self.index is None:
            raise RuntimeError("OPQ model must be fitted before get_codes().")

        data = np.ascontiguousarray(as_float32_numpy(xs))

        # 编码路径与官方 RPG 保持一致：克隆训练好的 OPQ/IVF1/PQ 索引，
        # add 当前向量，再从 IVF inverted list 中读取 PQ codes。
        codes = self._encode_with_index_factory(faiss, data)
        device = xs.device if isinstance(xs, torch.Tensor) else "cpu"
        return torch.from_numpy(codes).long().to(device)
