import logging
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common.layers import MLPLayers
from .common.model_utils import (
    as_float32_numpy,
    copy_codebooks_to_embeddings,
    get_faiss_rq_codebooks,
)
from .common.rq import ResidualVectorQuantizer


class ResidualEncoder(nn.Module):
    """Encoder wrapper for RQ-KMeans+: z = x + f(x)."""

    def __init__(self, correction_net):
        super().__init__()
        self.correction_net = correction_net

    def forward(self, x):
        return x + self.correction_net(x)


class RQKMEANS_PLUS(nn.Module):
    """RQ-VAE with residual encoder and RQ-KMeans codebook warm start."""

    def __init__(self, config: dict, input_size: int, item_embeddings=None):
        super().__init__()

        model_cfg = config["rqkmeans_plus"]
        model_params = model_cfg["model_params"]
        train_params = model_cfg["training_params"]

        hidden_sizes = model_params["hidden_sizes"]
        latent_size = model_params.get("latent_size", input_size)
        if latent_size is None:
            latent_size = input_size
        latent_size = int(latent_size)

        num_levels = int(model_params["num_levels"])
        codebook_size = int(model_params["codebook_size"])
        codebook_sizes = [
            int(size)
            for size in model_params.get("codebook_sizes", [codebook_size] * num_levels)
        ]
        dropout = model_params["dropout"]
        bn = model_params["bn"]
        sk_epsilons = model_params["sk_epsilons"]
        sk_iters = model_params["sk_iters"]
        self.pretrained_codebook_path = model_params.get("pretrained_codebook_path")
        self.faiss_verbose = bool(model_params.get("faiss_verbose", True))

        if latent_size != input_size:
            raise ValueError(
                "RQKMEANS_PLUS requires latent_size == input_size because its "
                "residual encoder computes z = x + f(x)."
            )
        if len(codebook_sizes) != num_levels:
            raise ValueError(
                "RQKMEANS_PLUS codebook_sizes must have the same length as num_levels."
            )
        if any(size != codebook_size for size in codebook_sizes):
            raise ValueError(
                "RQKMEANS_PLUS currently expects uniform codebook_sizes matching "
                "codebook_size so downstream semantic-id vocab sizes stay valid."
            )
        if len(sk_epsilons) != len(codebook_sizes):
            raise ValueError(
                "RQKMEANS_PLUS sk_epsilons must have the same length as codebook_sizes."
            )
        if item_embeddings is None and not self.pretrained_codebook_path:
            raise ValueError(
                "RQKMEANS_PLUS needs item_embeddings for FAISS warm-start when "
                "pretrained_codebook_path is not configured."
            )

        self.config = config
        self.loss_type = train_params["loss_type"]
        self.quant_loss_weight = train_params["quant_loss_weight"]
        self.codebook_sizes = codebook_sizes

        correction_dims = [input_size] + hidden_sizes + [latent_size]
        correction_net = MLPLayers(layers=correction_dims, dropout=dropout, bn=bn)
        self._zero_init_last_linear(correction_net)
        self.encoder = ResidualEncoder(correction_net)

        self.rq = ResidualVectorQuantizer(
            codebook_sizes,
            latent_size,
            beta=train_params["beta"],
            kmeans_init=False,
            kmeans_iters=0,
            sk_epsilons=sk_epsilons,
            sk_iters=sk_iters,
        )

        self.decoder = MLPLayers(layers=correction_dims[::-1], dropout=dropout, bn=bn)
        self._warm_start_codebooks(item_embeddings)

    @staticmethod
    def _zero_init_last_linear(module):
        last_linear = None
        for submodule in reversed(list(module.modules())):
            if isinstance(submodule, nn.Linear):
                last_linear = submodule
                break
        if last_linear is None:
            logging.warning("[RQKMEANS_PLUS] No Linear layer found for zero init.")
            return

        with torch.no_grad():
            last_linear.weight.zero_()
            if last_linear.bias is not None:
                last_linear.bias.zero_()
        logging.info("[RQKMEANS_PLUS] Zero-initialized residual encoder last Linear.")

    def _warm_start_codebooks(self, item_embeddings):
        embeddings = [quantizer.embedding for quantizer in self.rq.vq_layers]
        if self.pretrained_codebook_path:
            codebooks = self._load_npz_codebooks(self.pretrained_codebook_path)
            source = self.pretrained_codebook_path
        else:
            logging.info("[RQKMEANS_PLUS] Training FAISS RQ codebooks for warm start.")
            train_data = as_float32_numpy(item_embeddings)
            codebooks = get_faiss_rq_codebooks(
                train_data,
                codebook_sizes=self.codebook_sizes,
                verbose=self.faiss_verbose,
            )
            source = "FAISS residual k-means"

        copy_codebooks_to_embeddings(codebooks, embeddings, label="RQKMEANS_PLUS")
        for quantizer in self.rq.vq_layers:
            quantizer.initted = True
        logging.info("[RQKMEANS_PLUS] Warm-started codebooks from %s.", source)

    def _load_npz_codebooks(self, codebook_path):
        path = os.path.expanduser(codebook_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"RQKMEANS_PLUS codebook file not found: {path}")

        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            codebooks = []
            for idx in range(len(self.codebook_sizes)):
                key = f"codebook_{idx}"
                if key not in data:
                    raise KeyError(f"Missing '{key}' in {path}.")
                codebooks.append(np.asarray(data[key], dtype=np.float32))
            return codebooks

        array = np.asarray(data, dtype=np.float32)
        if array.ndim != 3:
            raise ValueError(
                "RQKMEANS_PLUS .npy codebook file must have shape "
                "(num_levels, codebook_size, dim)."
            )
        return [array[idx] for idx in range(array.shape[0])]

    def forward(self, x=None, xs=None, use_sk=True):
        if x is None:
            x = xs
        if x is None:
            raise ValueError("RQKMEANS_PLUS.forward requires x or xs.")

        z = self.encoder(x)
        z_q, rq_loss, indices = self.rq(z, use_sk=use_sk)
        out = self.decoder(z_q)
        return out, rq_loss, indices

    @torch.no_grad()
    def get_codes(self, x=None, xs=None, use_sk=False):
        if x is None:
            x = xs
        if x is None:
            raise ValueError("RQKMEANS_PLUS.get_codes requires x or xs.")

        z = self.encoder(x)
        _, _, indices = self.rq(z, use_sk=use_sk)
        return indices

    def compute_loss(self, outputs=None, *args, batch_data=None, xs=None, out=None, quant_loss=None):
        if outputs is None:
            outputs = out

        if isinstance(outputs, (tuple, list)):
            if len(outputs) < 2:
                raise ValueError(
                    "RQKMEANS_PLUS.compute_loss expected outputs to contain "
                    "out and quant_loss."
                )
            out = outputs[0]
            quant_loss = outputs[1]
        else:
            out = outputs
            if quant_loss is None and args:
                quant_loss = args[0]

        target = xs if xs is not None else batch_data
        if out is None or quant_loss is None or target is None:
            raise ValueError(
                "RQKMEANS_PLUS.compute_loss requires outputs, quant_loss, "
                "and target batch data."
            )

        if self.loss_type == "mse":
            loss_recon = F.mse_loss(out, target, reduction="mean")
        elif self.loss_type == "l1":
            loss_recon = F.l1_loss(out, target, reduction="mean")
        else:
            raise ValueError("incompatible loss type")

        loss_total = loss_recon + self.quant_loss_weight * quant_loss
        return {
            "loss_total": loss_total,
            "loss_recon": loss_recon,
            "loss_latent": quant_loss,
        }
