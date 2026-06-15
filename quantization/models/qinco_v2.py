import logging

import torch
from torch import nn

from .common.model_utils import (
    as_float32_numpy,
    assign_batch_multiple,
    assign_to_codebook,
    copy_codebooks_to_embeddings,
    get_faiss_rq_codebooks,
)


class QINCoV2Step(nn.Module):
    """Single neural residual quantization step used by QINCo V2."""

    def __init__(self, d, K, L, h):
        super().__init__()

        self.d, self.K, self.L, self.h = d, K, L, h
        self.codebook = nn.Embedding(K, d)
        self.MLPconcat = nn.Linear(d, d)

        self.residual_blocks = nn.ModuleList(
            nn.Sequential(
                nn.Linear(d, h, bias=False),
                nn.ReLU(),
                nn.Linear(h, d, bias=False),
            )
            for _ in range(L)
        )

    def decode(self, xhat, codes):
        zqs = self.codebook(codes)
        zqs = zqs + self.MLPconcat(zqs + xhat)

        for residual_block in self.residual_blocks:
            zqs = zqs + residual_block(zqs)

        return zqs

    def encode(self, xhat, x):
        zqs = self.codebook.weight
        K, d = zqs.shape
        batch_size = xhat.shape[0]

        zqs_r = zqs.repeat(batch_size, 1, 1).reshape(batch_size * K, d)
        xhat_r = (
            xhat.reshape(batch_size, 1, d)
            .repeat(1, K, 1)
            .reshape(batch_size * K, d)
        )

        zqs_r = zqs_r + self.MLPconcat(zqs_r + xhat_r)

        for residual_block in self.residual_blocks:
            zqs_r = zqs_r + residual_block(zqs_r)

        candidates = zqs_r.reshape(batch_size, K, d) + xhat.reshape(batch_size, 1, d)
        codes, xhat_next = assign_batch_multiple(x, candidates)

        return codes, xhat_next - xhat


class QINCO_V2(nn.Module):
    """
    QINCo V2 neural residual vector quantizer.

    This keeps QINCo's residual-code conditioning path, while registering as a
    separate quantization method (`qinco_v2`) so experiments can run without
    overwriting existing QINCo checkpoints or codebooks.
    """

    def __init__(self, config: dict, input_size: int, item_embeddings=None):
        super().__init__()

        model_cfg = config["qinco_v2"]
        model_params = model_cfg["model_params"]

        d = int(input_size)
        K = int(model_params["codebook_size"])
        L = int(model_params["num_residual_blocks"])
        M = int(model_params["num_levels"])
        h = int(model_params["hidden_size"])
        codebook_sizes = [
            int(size)
            for size in model_params.get("codebook_sizes", [K] * M)
        ]
        faiss_init = bool(model_params.get("faiss_init", True))
        faiss_verbose = bool(model_params.get("faiss_verbose", True))

        if M <= 0:
            raise ValueError("QINCO_V2 num_levels must be positive.")
        if K <= 0:
            raise ValueError("QINCO_V2 codebook_size must be positive.")
        if L < 0:
            raise ValueError("QINCO_V2 num_residual_blocks cannot be negative.")
        if h <= 0:
            raise ValueError("QINCO_V2 hidden_size must be positive.")
        if len(codebook_sizes) != M:
            raise ValueError("QINCO_V2 codebook_sizes must have the same length as num_levels.")
        if any(size != K for size in codebook_sizes):
            raise ValueError("QINCO_V2 currently requires every codebook_sizes entry to equal codebook_size.")
        if faiss_init and item_embeddings is None:
            raise ValueError("QINCO_V2 requires item_embeddings for FAISS initialization.")

        self.config = config
        self.d, self.K, self.L, self.M, self.h = d, K, L, M, h
        self.loss_weights = model_params.get("loss_weights")
        self.db_scale = self._resolve_db_scale(config, model_params, item_embeddings)
        logging.info("[QINCo_V2] Setting scaling factor to %s", self.db_scale)

        self.codebook0 = nn.Embedding(K, d)
        self.steps = nn.ModuleList(
            QINCoV2Step(d, K, L, h)
            for _ in range(1, M)
        )

        if faiss_init:
            self._init_codebooks_from_faiss(
                item_embeddings=item_embeddings,
                codebook_sizes=codebook_sizes,
                verbose=faiss_verbose,
            )

    @staticmethod
    def _resolve_db_scale(config, model_params, item_embeddings):
        scale = model_params.get(
            "db_scale",
            config.get("data_limits", {}).get("db_scale", -1),
        )
        try:
            scale = float(scale)
        except (TypeError, ValueError):
            scale = -1

        if scale <= 0:
            if item_embeddings is None:
                raise ValueError("QINCO_V2 db_scale <= 0 requires item_embeddings.")
            scale = float(as_float32_numpy(item_embeddings).max())

        if scale == 0:
            raise ValueError("QINCO_V2 db_scale resolved to 0.")
        return scale

    def _normalize_input(self, x):
        return x / self.db_scale

    def _init_codebooks_from_faiss(self, item_embeddings, codebook_sizes, verbose):
        train_data = as_float32_numpy(item_embeddings)
        codebooks = get_faiss_rq_codebooks(
            train_data,
            codebook_sizes=codebook_sizes,
            verbose=verbose,
        )
        codebooks = [codebook / self.db_scale for codebook in codebooks]

        qinco_codebooks = [self.codebook0] + [step.codebook for step in self.steps]
        copy_codebooks_to_embeddings(codebooks, qinco_codebooks, label="QINCo_V2")

    def decode(self, codes):
        xhat = self.codebook0(codes[:, 0])

        for idx, step in enumerate(self.steps):
            xhat = xhat + step.decode(xhat, codes[:, idx + 1])

        return xhat

    def encode(self, xs, code0=None):
        batch_size = xs.shape[0]
        codes = torch.zeros(batch_size, self.M, dtype=torch.long, device=xs.device)

        if code0 is None:
            code0 = assign_to_codebook(xs, self.codebook0.weight)

        codes[:, 0] = code0
        xhat = self.codebook0.weight[code0]

        for idx, step in enumerate(self.steps):
            codes[:, idx + 1], to_add = step.encode(xhat, xs)
            xhat = xhat + to_add

        return codes, xhat

    def forward(self, xs, code0=None):
        xs = self._normalize_input(xs)
        with torch.no_grad():
            codes, _ = self.encode(xs, code0=code0)

        losses = torch.zeros(self.M, device=xs.device)

        xhat = self.codebook0(codes[:, 0])
        losses[0] = (xhat - xs).pow(2).sum()

        for idx, step in enumerate(self.steps):
            xhat = xhat + step.decode(xhat, codes[:, idx + 1])
            losses[idx + 1] = (xhat - xs).pow(2).sum()

        return codes, xhat, losses

    @torch.no_grad()
    def get_codes(self, xs):
        xs = self._normalize_input(xs)
        codes, _ = self.encode(xs)
        return codes

    def compute_loss(self, outputs=None, *args, batch_data=None, xs=None, **kwargs):
        if outputs is None:
            raise ValueError("QINCO_V2.compute_loss requires forward outputs.")
        if not isinstance(outputs, (tuple, list)) or len(outputs) < 3:
            raise ValueError("QINCO_V2 outputs must contain codes, xhat, and losses.")

        _, xhat, losses = outputs[:3]
        target = xs if xs is not None else batch_data
        if target is None:
            raise ValueError("QINCO_V2.compute_loss requires batch_data or xs.")
        target = self._normalize_input(target)

        normalizer = max(int(target.numel()), 1)
        if self.loss_weights is not None:
            weights = torch.as_tensor(self.loss_weights, dtype=losses.dtype, device=losses.device)
            if weights.numel() != losses.numel():
                raise ValueError("QINCO_V2 loss_weights must have length num_levels.")
            loss_total = (losses * weights).sum() / normalizer
        else:
            loss_total = losses.sum() / normalizer

        batch_size = max(int(target.shape[0]), 1)
        loss_recon = (xhat - target).pow(2).sum() * (self.db_scale ** 2) / batch_size
        return {
            "loss_total": loss_total,
            "loss_recon": loss_recon,
        }
