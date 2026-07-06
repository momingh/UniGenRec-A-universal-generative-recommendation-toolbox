import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

from ..abstract_model import AbstractModel

try:
    from recommendation.metrics import ndcg_at_k, recall_at_k
except ImportError:
    from metrics import ndcg_at_k, recall_at_k


logger = logging.getLogger(__name__)


class ResBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.linear(x))


class RPGv4(AbstractModel):
    def __init__(
        self,
        config: Dict[str, Any],
        item_to_code_map: Optional[Dict[int, List[int]]] = None,
        **_: Any,
    ):
        super().__init__(config)
        model_params = config["model_params"]
        token_params = config["token_params"]

        if item_to_code_map is None:
            from dataset import item2code

            item_to_code_map, _ = item2code(
                config["code_path"],
                config["vocab_sizes"],
                config["bases"],
            )

        self.hidden_size = int(model_params["n_embd"])
        self.code_len = int(config["code_len"])
        self.n_pred_head = self.code_len
        self.vocab_sizes = [int(v) for v in config["vocab_sizes"]]
        self.bases = [int(v) for v in config["bases"]]
        self.codebook_size = int(config["codebook_size"])
        self.eos_token = sum(self.vocab_sizes) + 1
        self.temperature = float(model_params["temperature"])
        self.item_ce_loss_weight = float(model_params.get("item_ce_loss_weight", 1.0))

        max_item_id = max(item_to_code_map)
        pad_token_id = int(token_params["pad_token_id"])
        item_id2tokens = torch.full(
            (max_item_id + 1, self.code_len),
            pad_token_id,
            dtype=torch.long,
        )
        for item_id, tokens in item_to_code_map.items():
            if len(tokens) != self.code_len:
                raise ValueError(
                    f"Item {item_id} code length {len(tokens)} != code_len {self.code_len}."
                )
            item_id2tokens[int(item_id)] = torch.tensor(tokens, dtype=torch.long)
        self.register_buffer("item_id2tokens", item_id2tokens, persistent=False)

        gpt2config = GPT2Config(
            vocab_size=int(token_params["vocab_size"]),
            n_positions=int(model_params["max_len"]),
            n_embd=self.hidden_size,
            n_layer=int(model_params["n_layer"]),
            n_head=int(model_params["n_head"]),
            n_inner=int(model_params.get("n_inner", 4 * self.hidden_size)),
            activation_function=model_params.get("activation_function", "gelu_new"),
            resid_pdrop=float(model_params.get("resid_pdrop", 0.1)),
            embd_pdrop=float(model_params.get("embd_pdrop", 0.1)),
            attn_pdrop=float(model_params.get("attn_pdrop", 0.1)),
            layer_norm_epsilon=float(model_params.get("layer_norm_epsilon", 1e-5)),
            initializer_range=float(model_params.get("initializer_range", 0.02)),
            eos_token_id=int(token_params["eos_token_id"]),
            pad_token_id=pad_token_id,
        )
        self.gpt2 = GPT2Model(gpt2config)
        self.norm = nn.LayerNorm(self.hidden_size)
        self.pred_heads = nn.Sequential(
            *[ResBlock(self.hidden_size) for _ in range(self.n_pred_head)]
        )
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        logger.info(
            "RPGv4 initialized: items=%d, code_len=%d, item_ce_loss_weight=%.4f.",
            self.item_id2tokens.shape[0] - 1,
            self.code_len,
            self.item_ce_loss_weight,
        )

    @property
    def task_type(self) -> str:
        return "retrieval"

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(
            p.numel()
            for p in self.gpt2.get_input_embeddings().parameters()
            if p.requires_grad
        )
        return (
            f"#Embedding parameters: {emb_params}\n"
            f"#Non-embedding parameters: {total_params - emb_params}\n"
            f"#Total trainable parameters: {total_params}\n"
        )

    def _sid_mean_embeddings(self, sid_tokens: torch.Tensor) -> torch.Tensor:
        sid_embs = self.gpt2.wte(sid_tokens)
        return self.norm(sid_embs).mean(dim=-2)

    def _compute_code_token_loss(
        self,
        final_states: torch.Tensor,
        label_mask: torch.Tensor,
        token_labels: torch.Tensor,
    ) -> torch.Tensor:
        selected_states = final_states.view(
            -1,
            self.n_pred_head,
            self.hidden_size,
        )[label_mask]
        selected_states = F.normalize(selected_states, dim=-1)
        selected_states_chunks = torch.chunk(
            selected_states,
            self.n_pred_head,
            dim=1,
        )

        token_emb = self.gpt2.wte.weight[1:self.eos_token]
        token_emb = F.normalize(token_emb, dim=-1)
        token_embs_chunks = torch.split(token_emb, self.vocab_sizes, dim=0)

        token_logits = [
            torch.matmul(
                selected_states_chunks[i].squeeze(dim=1),
                token_embs_chunks[i].T,
            )
            / self.temperature
            for i in range(self.n_pred_head)
        ]

        code_token_losses = [
            self.loss_fct(
                token_logits[i],
                token_labels[:, i] - self.bases[i] - 1,
            )
            for i in range(self.n_pred_head)
        ]

        return torch.mean(torch.stack(code_token_losses))

    def _compute_item_ce_loss(
        self,
        hs: torch.Tensor,
        label_mask: torch.Tensor,
        target_item_embs: torch.Tensor,
    ) -> torch.Tensor:
        selected_hs = hs.view(-1, self.hidden_size)[label_mask]
        item_ce_logits = torch.matmul(
            F.normalize(selected_hs, dim=-1),
            F.normalize(target_item_embs, dim=-1).T,
        ) / self.temperature
        item_ce_labels = torch.arange(
            item_ce_logits.shape[0],
            device=item_ce_logits.device,
        )

        return F.cross_entropy(item_ce_logits, item_ce_labels)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_loss: bool = True,
    ) -> torch.Tensor:
        input_tokens = self.item_id2tokens[batch["input_ids"]]
        token_emb = self.gpt2.wte(input_tokens)
        input_embs = self.norm(token_emb).mean(dim=-2)
        # input_embs = token_emb.mean(dim=-2)
        outputs = self.gpt2(inputs_embeds=input_embs, attention_mask=batch["attention_mask"])
        hs = outputs.last_hidden_state
        final_states = [self.pred_heads[i](hs).unsqueeze(-2) for i in range(self.n_pred_head)]
        final_states = torch.cat(final_states, dim=-2)
        outputs.final_states = final_states

        if return_loss:
            labels = batch["labels_seq"]
            label_mask = labels.view(-1) != -100
            target_item_ids = labels.view(-1)[label_mask] + 1
            token_labels = self.item_id2tokens[target_item_ids]
            target_item_embs = self._sid_mean_embeddings(token_labels)

            code_token_loss = self._compute_code_token_loss(
                final_states,
                label_mask,
                token_labels,
            )
            item_ce_loss = self._compute_item_ce_loss(
                hs,
                label_mask,
                target_item_embs,
            )

            outputs.target_item_embs = target_item_embs
            outputs.code_token_loss = code_token_loss
            outputs.main_loss = code_token_loss
            outputs.item_ce_loss = item_ce_loss
            outputs.loss = code_token_loss + self.item_ce_loss_weight * item_ce_loss
        return outputs

    def _rank_item_ids(
        self,
        batch: Dict[str, torch.Tensor],
        n_return_sequences: int = 1,
    ) -> torch.Tensor:
        """Return ranked 1-based item ids, matching the internal item_id2tokens table."""
        outputs = self.forward(batch, return_loss=False)
        seq_lens = batch.get("seq_lens", batch["attention_mask"].long().sum(dim=1))
        last_step_indices = (seq_lens - 1).view(-1, 1, 1, 1).expand(-1, 1, self.n_pred_head, self.hidden_size)
        states = outputs.final_states.gather(dim=1, index=last_step_indices)
        states = F.normalize(states, dim=-1)

        token_emb = self.gpt2.wte.weight[1:self.eos_token]
        token_emb = F.normalize(token_emb, dim=-1)
        token_embs_chunks = torch.split(token_emb, self.vocab_sizes, dim=0)

        logits = [
            torch.matmul(states[:, 0, i, :], token_embs_chunks[i].T) / self.temperature
            for i in range(self.n_pred_head)
        ]
        logits = [F.log_softmax(logit, dim=-1) for logit in logits]
        token_logits = torch.cat(logits, dim=-1)

        candidate_item_ids = torch.arange(1, self.item_id2tokens.shape[0], device=token_logits.device)
        item_codes_indices = self.item_id2tokens[1:self.item_id2tokens.shape[0], :] - 1
        valid_mask = (item_codes_indices >= 0).all(dim=-1) & (item_codes_indices < token_logits.shape[-1]).all(dim=-1)
        candidate_item_ids = candidate_item_ids[valid_mask]
        item_codes_indices = item_codes_indices[valid_mask]

        expanded_logits = token_logits.unsqueeze(1).expand(-1, item_codes_indices.shape[0], -1)
        expanded_indices = item_codes_indices.unsqueeze(0).expand(token_logits.shape[0], -1, -1)

        item_code_logits = torch.gather(input=expanded_logits, dim=2, index=expanded_indices)
        item_scores = item_code_logits.sum(dim=-1)

        topk_indices = item_scores.topk(min(n_return_sequences, item_scores.shape[-1]), dim=-1).indices
        return candidate_item_ids[topk_indices]

    def generate(
        self,
        batch: Dict[str, torch.Tensor],
        n_return_sequences: int = 1,
    ) -> torch.Tensor:
        topk_item_ids = self._rank_item_ids(batch, n_return_sequences=n_return_sequences)
        predicted_codebooks = self.item_id2tokens[topk_item_ids]

        return predicted_codebooks

    def evaluate_step(
        self,
        batch: Dict[str, torch.Tensor],
        topk_list: List[int],
    ) -> Dict[str, float]:
        max_k = max(topk_list)
        ranked_item_ids = self._rank_item_ids(batch, n_return_sequences=max_k) - 1
        target_ids = batch["target_ids"].unsqueeze(1)
        hits = (ranked_item_ids == target_ids).cpu()

        batch_metrics: Dict[str, float] = {"count": float(target_ids.shape[0])}
        for k in topk_list:
            batch_metrics[f"Recall@{k}"] = recall_at_k(hits, k).sum().item()
            batch_metrics[f"NDCG@{k}"] = ndcg_at_k(hits, k).sum().item()
        return batch_metrics
