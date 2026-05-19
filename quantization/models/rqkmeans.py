import logging

import torch
from torch import nn

from .common.model_utils import (
    as_float32_numpy,
    assign_to_codebook,
    copy_codebooks_to_embeddings,
    get_faiss_rq_codebooks,
)


class RQKMEANS(nn.Module):
    """Residual K-Means quantizer initialized by FAISS and used without training."""

    fit_on_cpu = True

    def __init__(self, config, input_size, item_embeddings=None):
        super().__init__()

        params = config["rqkmeans"]["model_params"]
        self.input_size = int(input_size)
        self.num_levels = int(params["num_levels"])
        self.codebook_size = int(params["codebook_size"])
        self.codebook_sizes = [
            int(size)
            for size in params.get(
                "codebook_sizes", [self.codebook_size] * self.num_levels
            )
        ]
        self.faiss_verbose = bool(params.get("faiss_verbose", True))

        if len(self.codebook_sizes) != self.num_levels:
            raise ValueError(
                f"rqkmeans.codebook_sizes length ({len(self.codebook_sizes)}) "
                f"must match num_levels ({self.num_levels})."
            )
        if any(size != self.codebook_size for size in self.codebook_sizes):
            raise ValueError(
                "rqkmeans currently expects uniform codebook_sizes matching "
                "codebook_size so downstream semantic-id vocab sizes stay valid."
            )
        if item_embeddings is None:
            raise ValueError(
                "RQKMEANS needs item_embeddings because FAISS initializes all "
                "codebooks in __init__."
            )

        self.codebooks = nn.ModuleList(
            [nn.Embedding(size, self.input_size) for size in self.codebook_sizes]
        )
        self._init_codebooks_from_faiss(item_embeddings)
        for codebook in self.codebooks:
            codebook.weight.requires_grad_(False)

    @property
    def is_iterative(self):
        return False

    def _init_codebooks_from_faiss(self, item_embeddings):
        """Train FAISS ResidualQuantizer codebooks once and copy them into torch."""
        logging.info("[RQKMEANS] Initializing codebooks from FAISS residual k-means.")
        train_data = as_float32_numpy(item_embeddings)
        codebooks = get_faiss_rq_codebooks(
            train_data,
            self.codebook_sizes,
            verbose=self.faiss_verbose,
        )
        copy_codebooks_to_embeddings(
            codebooks,
            self.codebooks,
            label="RQKMEANS",
        )

    def fit(self, xs=None):
        logging.info("[RQKMEANS] Skipping gradient training; FAISS codebooks are fixed.")
        return self

    def encode(self, x=None, xs=None):
        if x is None:
            x = xs
        if x is None:
            raise ValueError("RQKMEANS.encode expects x or xs.")

        residual = x
        x_hat = torch.zeros_like(x)
        codes = []

        for codebook in self.codebooks:
            code = assign_to_codebook(residual, codebook.weight)
            quantized = codebook(code)
            codes.append(code)
            x_hat = x_hat + quantized
            residual = residual - quantized

        return torch.stack(codes, dim=1), x_hat

    @torch.no_grad()
    def get_codes(self, x=None, xs=None):
        codes, _ = self.encode(x=x, xs=xs)
        return codes

    def forward(self, x=None, xs=None):
        if x is None:
            x = xs
        codes, x_hat = self.encode(x=x)
        loss = ((x_hat - x) ** 2).mean()
        return x_hat, loss, codes

    def compute_loss(self, outputs=None, batch_data=None):
        if outputs is not None:
            _, loss, _ = outputs
            return {"loss_total": loss}
        if batch_data is None:
            raise ValueError("RQKMEANS.compute_loss expects outputs or batch_data.")
        _, loss, _ = self.forward(x=batch_data)
        return {"loss_total": loss}
