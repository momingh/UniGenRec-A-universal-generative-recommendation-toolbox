from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from metrics import ndcg_at_k, recall_at_k
except ImportError:
    from ..metrics import ndcg_at_k, recall_at_k


class SASRec(nn.Module):
    def __init__(self, config: Dict, num_items: int):
        super().__init__()
        model_params = config["model_params"]
        self.config = config
        self.num_items = int(num_items)
        self.max_len = int(model_params["max_len"])
        self.hidden_size = int(model_params["hidden_size"])
        self.num_attention_heads = int(model_params["num_attention_heads"])
        self.initializer_range = float(model_params.get("initializer_range", 0.02))

        self.item_embeddings = nn.Embedding(self.num_items + 1, self.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(self.max_len + 1, self.hidden_size, padding_idx=0)
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(float(model_params.get("hidden_dropout_prob", 0.0)))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_attention_heads,
            dim_feedforward=int(model_params.get("intermediate_size", self.hidden_size * 4)),
            dropout=float(model_params.get("attention_probs_dropout_prob", 0.0)),
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(model_params["num_hidden_layers"]),
        )

        self.apply(self._init_weights)
        with torch.no_grad():
            self.item_embeddings.weight[0].fill_(0)
            self.position_embeddings.weight[0].fill_(0)

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"Total number of trainable parameters: {total_params:,}"

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

    def _attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = attention_mask.shape
        causal_mask = self._causal_mask(seq_len, attention_mask.device)
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)

        valid_query_mask = attention_mask.bool().unsqueeze(2)
        key_padding_mask = (attention_mask == 0).unsqueeze(1)
        combined_mask = causal_mask | (valid_query_mask & key_padding_mask)

        combined_mask = combined_mask.unsqueeze(1).expand(
            batch_size,
            self.num_attention_heads,
            seq_len,
            seq_len,
        )
        return combined_mask.reshape(batch_size * self.num_attention_heads, seq_len, seq_len)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(1, seq_len + 1, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)

        hidden_states = self.item_embeddings(input_ids) + self.position_embeddings(position_ids)
        hidden_states = hidden_states * attention_mask.unsqueeze(-1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        sequence_output = self.encoder(
            hidden_states,
            mask=self._attention_mask(attention_mask),
        )
        return sequence_output.masked_fill(attention_mask.unsqueeze(-1) == 0, 0.0)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels_seq = batch["labels_seq"]
        negative_labels_seq = batch["negative_labels_seq"]

        sequence_output = self.encode(input_ids, attention_mask)
        valid_mask = labels_seq != -100
        if not torch.any(valid_mask):
            zero_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
            return {
                "loss": zero_loss,
                "positive_loss": zero_loss.detach(),
                "negative_loss": zero_loss.detach(),
            }

        positive_loss, negative_loss = self._bce_loss(
            sequence_output,
            labels_seq,
            negative_labels_seq,
            valid_mask,
        )
        return {
            "loss": positive_loss + negative_loss,
            "positive_loss": positive_loss.detach(),
            "negative_loss": negative_loss.detach(),
        }

    def _bce_loss(
        self,
        sequence_output: torch.Tensor,
        labels_seq: torch.Tensor,
        negative_labels_seq: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = sequence_output[valid_mask]
        positive_item_ids = labels_seq[valid_mask] + 1
        negative_item_ids = negative_labels_seq[valid_mask] + 1

        positive_embeddings = self.item_embeddings(positive_item_ids)
        negative_embeddings = self.item_embeddings(negative_item_ids)
        positive_logits = (hidden_states * positive_embeddings).sum(dim=-1)
        negative_logits = (hidden_states * negative_embeddings).sum(dim=-1)

        positive_loss = F.binary_cross_entropy_with_logits(
            positive_logits,
            torch.ones_like(positive_logits),
        )
        negative_loss = F.binary_cross_entropy_with_logits(
            negative_logits,
            torch.zeros_like(negative_logits),
        )
        return positive_loss, negative_loss

    def _last_hidden_state(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        sequence_output = self.encode(batch["input_ids"], batch["attention_mask"])
        return sequence_output[:, -1]

    def _mask_seen_scores(
        self,
        scores: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        filter_item_ids = batch.get("filter_item_ids")
        if filter_item_ids is None or filter_item_ids.numel() == 0:
            return scores

        valid_filter = (filter_item_ids >= 0) & (filter_item_ids < scores.shape[1])
        target_ids = batch.get("target_ids")
        if target_ids is not None:
            valid_filter = valid_filter & (filter_item_ids != target_ids.unsqueeze(1))
        if not torch.any(valid_filter):
            return scores

        row_indices = torch.arange(scores.shape[0], device=scores.device).unsqueeze(1)
        row_indices = row_indices.expand_as(filter_item_ids)
        scores[row_indices[valid_filter], filter_item_ids[valid_filter]] = torch.finfo(scores.dtype).min
        return scores

    def generate_ranklist(self, batch: Dict[str, torch.Tensor], topk: int) -> torch.Tensor:
        final_state = self._last_hidden_state(batch)
        item_emb = self.item_embeddings.weight[1:]
        scores = torch.matmul(final_state, item_emb.transpose(0, 1))
        scores = self._mask_seen_scores(scores, batch)

        k = min(topk, scores.shape[1])
        _, topk_indices = torch.topk(scores, k=k, dim=1)
        return topk_indices

    def evaluate_step(self, batch: Dict[str, torch.Tensor], topk_list: List[int]) -> Dict[str, float]:
        max_k = max(topk_list)
        ranked_item_indices = self.generate_ranklist(batch, topk=max_k)
        target_ids = batch["target_ids"].unsqueeze(1)
        hits = ranked_item_indices == target_ids

        batch_metrics: Dict[str, float] = {}
        for requested_k in topk_list:
            actual_k = min(requested_k, ranked_item_indices.shape[1])
            pos_index_k = hits[:, :actual_k]
            batch_metrics[f"Recall@{requested_k}"] = recall_at_k(pos_index_k, actual_k).sum().item()
            batch_metrics[f"NDCG@{requested_k}"] = ndcg_at_k(pos_index_k, actual_k).sum().item()
        batch_metrics["count"] = float(target_ids.shape[0])
        return batch_metrics
