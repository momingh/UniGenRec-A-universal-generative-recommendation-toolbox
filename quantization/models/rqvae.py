import torch
from torch import nn
from torch.nn import functional as F

from .common.layers import MLPLayers
from .common.rq import ResidualVectorQuantizer


class RQVAE(nn.Module):
    def __init__(self, config: dict, input_size: int, item_embeddings=None):
        super(RQVAE, self).__init__()

        model_cfg = config["rqvae"]
        model_params = model_cfg["model_params"]
        train_params = model_cfg["training_params"]

        hidden_sizes = model_params["hidden_sizes"]
        latent_size = model_params["latent_size"]
        num_levels = model_params["num_levels"]
        codebook_size = model_params["codebook_size"]
        dropout = model_params["dropout"]
        bn = model_params["bn"]
        kmeans_init = model_params["kmeans_init"]
        kmeans_iters = model_params["kmeans_iters"]
        sk_epsilons = model_params["sk_epsilons"]
        sk_iters = model_params["sk_iters"]

        num_emb_list = [codebook_size] * num_levels
        if len(sk_epsilons) != len(num_emb_list):
            raise ValueError("RQVAE sk_epsilons must have the same length as num_emb_list.")

        self.config = config
        self.loss_type = train_params["loss_type"]
        self.quant_loss_weight = train_params["quant_loss_weight"]

        encode_layer_dims = [input_size] + hidden_sizes + [latent_size]
        self.encoder = MLPLayers(layers=encode_layer_dims, dropout=dropout, bn=bn)

        self.rq = ResidualVectorQuantizer(num_emb_list, latent_size,
                                          beta=train_params["beta"],
                                          kmeans_init=kmeans_init,
                                          kmeans_iters=kmeans_iters,
                                          sk_epsilons=sk_epsilons,
                                          sk_iters=sk_iters)

        self.decoder = MLPLayers(layers=encode_layer_dims[::-1], dropout=dropout, bn=bn)

    def forward(self, x=None, xs=None, use_sk=True):
        if x is None:
            x = xs
        if x is None:
            raise ValueError("RQVAE.forward requires x or xs.")

        x = self.encoder(x)
        x_q, rq_loss, indices = self.rq(x,use_sk=use_sk)
        out = self.decoder(x_q)

        return out, rq_loss, indices

    @torch.no_grad()
    def get_codes(self, xs, use_sk=False):
        x_e = self.encoder(xs)
        _, _, indices = self.rq(x_e, use_sk=use_sk)
        return indices

    def compute_loss(self, outputs=None, *args, batch_data=None, xs=None, out=None, quant_loss=None):
        if outputs is None:
            outputs = out

        if isinstance(outputs, (tuple, list)):
            if len(outputs) < 2:
                raise ValueError("RQVAE.compute_loss expected outputs to contain out and quant_loss.")
            out = outputs[0]
            quant_loss = outputs[1]
        else:
            out = outputs
            if quant_loss is None and args:
                quant_loss = args[0]

        target = xs if xs is not None else batch_data
        if out is None or quant_loss is None or target is None:
            raise ValueError("RQVAE.compute_loss requires outputs, quant_loss, and target batch data.")

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, target, reduction='mean')
        elif self.loss_type == 'mse_l2':
            loss_recon = (out - target).pow(2).flatten(1).sum(dim=1).mean()
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, target, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return {
            "loss_total": loss_total,
            "loss_recon": loss_recon,
            "loss_latent": quant_loss,
        }
