import logging

import torch
import torch.nn.functional as F
from torch import nn

from .common.model_utils import (
    as_float32_numpy,
    assign_batch_multiple,
    assign_to_codebook,
    copy_codebooks_to_embeddings,
    get_faiss_rq_codebooks,
)


class QINCoAuxStep(nn.Module):
    """Single QINCo residual quantization step with optional usage encodings."""

    def __init__(self, d, K, L, h):
        super().__init__()

        self.d, self.K, self.L, self.h = d, K, L, h
        self.codebook = nn.Embedding(K, d)
        self.MLPconcat = nn.Linear(2 * d, d)
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
        concat = torch.cat((zqs, xhat), dim=1)
        zqs = zqs + self.MLPconcat(concat)

        for residual_block in self.residual_blocks:
            zqs = zqs + residual_block(zqs)

        return zqs

    def encode(self, xhat, x, return_encodings=False):
        zqs = self.codebook.weight
        K, d = zqs.shape
        batch_size = xhat.shape[0]

        zqs_r = zqs.repeat(batch_size, 1, 1).reshape(batch_size * K, d)
        xhat_r = (
            xhat.reshape(batch_size, 1, d)
            .repeat(1, K, 1)
            .reshape(batch_size * K, d)
        )

        concat = torch.cat((zqs_r, xhat_r), dim=1)
        zqs_r = zqs_r + self.MLPconcat(concat)

        for residual_block in self.residual_blocks:
            zqs_r = zqs_r + residual_block(zqs_r)

        candidates = zqs_r.reshape(batch_size, K, d) + xhat.reshape(batch_size, 1, d)
        codes, xhat_next = assign_batch_multiple(x, candidates)

        if not return_encodings:
            return codes, xhat_next - xhat

        encodings = torch.zeros(batch_size, K, dtype=x.dtype, device=x.device)
        encodings.scatter_(1, codes.unsqueeze(1), 1)
        return codes, xhat_next - xhat, encodings


class QINCO_AUX(nn.Module):
    """
    QINCo with auxiliary codebook diversity and utilization losses.

    This is registered as `qinco_aux` so experiments can run without
    overwriting the existing `qinco` and `qinco_v2` checkpoints/codebooks.
    """

    def __init__(self, config: dict, input_size: int, item_embeddings=None):
        super().__init__()

        model_cfg = config["qinco_aux"]
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
            raise ValueError("QINCO_AUX num_levels must be positive.")
        if K <= 0:
            raise ValueError("QINCO_AUX codebook_size must be positive.")
        if L < 0:
            raise ValueError("QINCO_AUX num_residual_blocks cannot be negative.")
        if h <= 0:
            raise ValueError("QINCO_AUX hidden_size must be positive.")
        if len(codebook_sizes) != M:
            raise ValueError("QINCO_AUX codebook_sizes must have the same length as num_levels.")
        if any(size != K for size in codebook_sizes):
            raise ValueError("QINCO_AUX currently requires every codebook_sizes entry to equal codebook_size.")
        if faiss_init and item_embeddings is None:
            raise ValueError("QINCO_AUX requires item_embeddings for FAISS initialization.")

        self.config = config
        self.d, self.K, self.L, self.M, self.h = d, K, L, M, h
        self.loss_weights = model_params.get("loss_weights")
        self.diversity_loss_weight = float(model_params.get("diversity_loss_weight", 0.01))
        self.utilization_loss_weight = float(model_params.get("utilization_loss_weight", 0.01))
        self.db_scale = self._resolve_db_scale(config, model_params, item_embeddings)
        logging.info("[QINCo_AUX] Setting scaling factor to %s", self.db_scale)
        logging.info(
            "[QINCo_AUX] Auxiliary loss weights: diversity=%s, utilization=%s",
            self.diversity_loss_weight,
            self.utilization_loss_weight,
        )

        self.codebook0 = nn.Embedding(K, d)
        self.steps = nn.ModuleList(
            QINCoAuxStep(d, K, L, h)
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
                raise ValueError("QINCO_AUX db_scale <= 0 requires item_embeddings.")
            scale = float(as_float32_numpy(item_embeddings).max())

        if scale == 0:
            raise ValueError("QINCO_AUX db_scale resolved to 0.")
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
        copy_codebooks_to_embeddings(codebooks, qinco_codebooks, label="QINCo_AUX")

    def decode(self, codes):
        xhat = self.codebook0(codes[:, 0])

        for idx, step in enumerate(self.steps):
            xhat = xhat + step.decode(xhat, codes[:, idx + 1])

        return xhat

    def encode(self, xs, code0=None, return_encodings=False):
        batch_size = xs.shape[0]
        codes = torch.zeros(batch_size, self.M, dtype=torch.long, device=xs.device)
        all_encodings = [] if return_encodings else None

        if code0 is None:
            code0 = assign_to_codebook(xs, self.codebook0.weight)

        codes[:, 0] = code0
        xhat = self.codebook0.weight[code0]

        if return_encodings:
            encodings0 = torch.zeros(batch_size, self.K, dtype=xs.dtype, device=xs.device)
            encodings0.scatter_(1, code0.unsqueeze(1), 1)
            all_encodings.append(encodings0)

        for idx, step in enumerate(self.steps):
            if return_encodings:
                codes[:, idx + 1], to_add, encodings = step.encode(
                    xhat,
                    xs,
                    return_encodings=True,
                )
                all_encodings.append(encodings)
            else:
                codes[:, idx + 1], to_add = step.encode(xhat, xs)
            xhat = xhat + to_add

        if return_encodings:
            return codes, xhat, all_encodings
        return codes, xhat

    def calculate_auxiliary_losses(self, all_encodings):
        zero = self.codebook0.weight.new_tensor(0.0)
        diversity_loss = zero
        utilization_loss = zero

        all_codebooks = [self.codebook0] + [step.codebook for step in self.steps]
        for codebook in all_codebooks:
            codebook_weight = codebook.weight
            norm_codebook = F.normalize(codebook_weight, p=2, dim=1)
            cosine_sim = norm_codebook @ norm_codebook.t()
            mask = torch.triu(
                torch.ones_like(cosine_sim, dtype=torch.bool),
                diagonal=1,
            )
            if mask.any():
                diversity_loss = diversity_loss + cosine_sim[mask].mean()

        diversity_loss = diversity_loss / len(all_codebooks)

        if all_encodings:
            for encodings in all_encodings:
                avg_probs = encodings.mean(dim=0)
                entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum()
                utilization_loss = utilization_loss - entropy
            utilization_loss = utilization_loss / len(all_encodings)

        loss_aux = (
            self.diversity_loss_weight * diversity_loss
            + self.utilization_loss_weight * utilization_loss
        )
        return loss_aux, diversity_loss, utilization_loss

    def forward(self, xs=None, x=None, code0=None, return_aux_loss=True):
        if xs is None:
            xs = x
        if xs is None:
            raise ValueError("QINCO_AUX.forward requires xs or x.")

        xs = self._normalize_input(xs)
        with torch.no_grad():
            if return_aux_loss:
                codes, _, all_encodings = self.encode(
                    xs,
                    code0=code0,
                    return_encodings=True,
                )
            else:
                codes, _ = self.encode(xs, code0=code0)

        losses = torch.zeros(self.M, device=xs.device)

        xhat = self.codebook0(codes[:, 0])
        losses[0] = (xhat - xs).pow(2).sum()

        for idx, step in enumerate(self.steps):
            xhat = xhat + step.decode(xhat, codes[:, idx + 1])
            losses[idx + 1] = (xhat - xs).pow(2).sum()

        if return_aux_loss:
            loss_aux, loss_diversity, loss_utilization = self.calculate_auxiliary_losses(all_encodings)
            return codes, xhat, losses, loss_aux, loss_diversity, loss_utilization

        return codes, xhat, losses

    @torch.no_grad()
    def get_codes(self, xs):
        xs = self._normalize_input(xs)
        codes, _ = self.encode(xs)
        return codes

    def compute_loss(self, outputs=None, *args, batch_data=None, xs=None, **kwargs):
        if outputs is None:
            raise ValueError("QINCO_AUX.compute_loss requires forward outputs.")
        if not isinstance(outputs, (tuple, list)) or len(outputs) < 3:
            raise ValueError("QINCO_AUX outputs must contain codes, xhat, and losses.")

        _, xhat, losses = outputs[:3]
        loss_aux = outputs[3] if len(outputs) > 3 else None
        loss_diversity = outputs[4] if len(outputs) > 4 else None
        loss_utilization = outputs[5] if len(outputs) > 5 else None

        target = xs if xs is not None else batch_data
        if target is None:
            raise ValueError("QINCO_AUX.compute_loss requires batch_data or xs.")
        target = self._normalize_input(target)

        normalizer = max(int(target.numel()), 1)
        if self.loss_weights is not None:
            weights = torch.as_tensor(self.loss_weights, dtype=losses.dtype, device=losses.device)
            if weights.numel() != losses.numel():
                raise ValueError("QINCO_AUX loss_weights must have length num_levels.")
            loss_recon_steps = (losses * weights).sum() / normalizer
        else:
            loss_recon_steps = losses.sum() / normalizer

        loss_total = loss_recon_steps
        if loss_aux is not None:
            loss_total = loss_total + loss_aux

        batch_size = max(int(target.shape[0]), 1)
        loss_recon = (xhat - target).pow(2).sum() * (self.db_scale ** 2) / batch_size

        loss_dict = {
            "loss_total": loss_total,
            "loss_recon": loss_recon,
        }
        if loss_aux is not None:
            loss_dict["loss_aux"] = loss_aux
        if loss_diversity is not None:
            loss_dict["loss_diversity"] = loss_diversity
        if loss_utilization is not None:
            loss_dict["loss_utilization"] = loss_utilization
        return loss_dict


QINCoAux = QINCO_AUX
