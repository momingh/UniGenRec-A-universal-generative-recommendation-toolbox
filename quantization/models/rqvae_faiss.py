import logging

import torch
from torch import nn
from torch.nn import functional as F

from .common.layers import MLPLayers
from .common.model_utils import as_float32_numpy, copy_codebooks_to_embeddings, get_faiss_rq_codebooks
from .common.rq import ResidualVectorQuantizer


class ResidualEncoder(nn.Module):
    def __init__(self, correction_net):
        super().__init__()
        self.correction_net = correction_net

    def forward(self, x):
        return x + self.correction_net(x)


class RQVAE_FAISS(nn.Module):
    def __init__(self, config: dict, input_size: int, item_embeddings=None):
        super(RQVAE_FAISS, self).__init__()

        model_cfg = config["rqvae_faiss"]
        model_params = model_cfg["model_params"]
        train_params = model_cfg["training_params"]

        hidden_sizes = model_params["hidden_sizes"]
        latent_size = model_params.get("latent_size", input_size)
        if latent_size is None:
            latent_size = input_size
        latent_size = int(latent_size)
        num_levels = model_params["num_levels"]
        codebook_sizes = [int(size) for size in model_params["codebook_sizes"]]
        dropout = model_params["dropout"]
        bn = model_params["bn"]
        sk_epsilons = model_params["sk_epsilons"]
        sk_iters = model_params["sk_iters"]
        faiss_verbose = model_params["faiss_verbose"]

        num_emb_list = codebook_sizes
        if latent_size != input_size:
            raise ValueError(
                "RQVAE_FAISS uses original item_embeddings for FAISS initialization, "
                "so latent_size must equal input_size. Set latent_size to null in the "
                "config to use input_size automatically."
            )
        if len(num_emb_list) != num_levels:
            raise ValueError("RQVAE_FAISS codebook_sizes must have the same length as num_levels.")
        if len(sk_epsilons) != len(num_emb_list):
            raise ValueError("RQVAE_FAISS sk_epsilons must have the same length as num_emb_list.")
        if item_embeddings is None:
            raise ValueError("RQVAE_FAISS requires item_embeddings for FAISS initialization.")

        self.config = config
        self.loss_type = train_params["loss_type"]
        self.quant_loss_weight = train_params["quant_loss_weight"]

        encode_layer_dims = [input_size] + hidden_sizes + [latent_size]
        correction_net = MLPLayers(layers=encode_layer_dims, dropout=dropout, bn=bn)
        self._zero_init_last_linear(correction_net)
        self.encoder = ResidualEncoder(correction_net)

        self.rq = ResidualVectorQuantizer(num_emb_list, latent_size,
                                          beta=train_params["beta"],
                                          kmeans_init=False,
                                          kmeans_iters=0,
                                          sk_epsilons=sk_epsilons,
                                          sk_iters=sk_iters)

        self.decoder = MLPLayers(layers=encode_layer_dims[::-1], dropout=dropout, bn=bn)
        self._init_codebooks_from_faiss(
            item_embeddings=item_embeddings,
            codebook_sizes=codebook_sizes,
            verbose=faiss_verbose,
        )

    @staticmethod
    def _zero_init_last_linear(module):
        last_linear = None
        for submodule in reversed(list(module.modules())):
            if isinstance(submodule, nn.Linear):
                last_linear = submodule
                break
        if last_linear is None:
            logging.warning("[RQVAE_FAISS] No Linear layer found for zero init.")
            return

        with torch.no_grad():
            last_linear.weight.zero_()
            if last_linear.bias is not None:
                last_linear.bias.zero_()
        logging.info("[RQVAE_FAISS] Zero-initialized residual encoder last Linear.")

    def _init_codebooks_from_faiss(self, item_embeddings, codebook_sizes, verbose):
        logging.info("使用原始 item_embeddings 初始化 FAISS ResidualQuantizer codebook...")
        train_data = as_float32_numpy(item_embeddings)
        codebooks = get_faiss_rq_codebooks(
            train_data,
            codebook_sizes=codebook_sizes,
            verbose=verbose,
        )

        embeddings = [quantizer.embedding for quantizer in self.rq.vq_layers]
        copy_codebooks_to_embeddings(codebooks, embeddings, label="RQ-VAE")
        for quantizer in self.rq.vq_layers:
            quantizer.initted = True

    def forward(self, x=None, xs=None, use_sk=True):
        if x is None:
            x = xs
        if x is None:
            raise ValueError("RQVAE_FAISS.forward requires x or xs.")

        x = self.encoder(x)
        x_q, rq_loss, indices = self.rq(x, use_sk=use_sk)
        out = self.decoder(x_q)

        return out, rq_loss, indices

    @torch.no_grad()
    def get_codes(self, x=None, xs=None, use_sk=False):
        if x is None:
            x = xs
        if x is None:
            raise ValueError("RQVAE_FAISS.get_codes requires x or xs.")

        x_e = self.encoder(x)
        _, _, indices = self.rq(x_e, use_sk=use_sk)
        return indices

    def compute_loss(self, outputs, batch_data):
        out, quant_loss, _ = outputs

        if self.loss_type == "mse":
            loss_recon = F.mse_loss(out, batch_data, reduction="mean")
        elif self.loss_type == "l1":
            loss_recon = F.l1_loss(out, batch_data, reduction="mean")
        else:
            raise ValueError("incompatible loss type")

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return {
            "loss_total": loss_total,
            "loss_recon": loss_recon,
            "loss_latent": quant_loss,
        }
