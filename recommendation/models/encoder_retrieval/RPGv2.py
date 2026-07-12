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


class SIDFourierPooling(nn.Module):
    def __init__(
        self,
        code_len: int,
        hidden_size: int,
        dropout: float = 0.0,
        filter_scale: float = 0.1,
        hyper_hidden_size: Optional[int] = None,
    ):
        super().__init__()
        self.code_len = code_len
        self.hidden_size = hidden_size
        self.n_freq = code_len // 2 + 1
        self.filter_scale = filter_scale
        self.hyper_hidden_size = hidden_size if hyper_hidden_size is None else int(hyper_hidden_size)
        if self.hyper_hidden_size <= 0:
            raise ValueError("SIDFourierPooling hyper_hidden_size must be positive.")

        self.freq_hyper = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.hyper_hidden_size),
            nn.SiLU(),
            nn.Linear(self.hyper_hidden_size, self.n_freq),
        )
        nn.init.zeros_(self.freq_hyper[-1].weight)
        nn.init.zeros_(self.freq_hyper[-1].bias)

        self.pool_score = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.pool_score.weight)
        nn.init.zeros_(self.pool_score.bias)

        self.out_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_emb: torch.Tensor):
        """
        token_emb: [B, L, K, D]

        return:
            pooled: [B, L, D]
            freq_filter: [B, L, F, 1]
        """
        dtype = token_emb.dtype

        x = token_emb

        # Fourier transform over code-depth dimension K.
        # Use float32 for FFT stability under fp16/bf16 training.
        x_freq = torch.fft.rfft(x.float(), dim=-2, norm="backward")  # [B, L, F, D]

        # Input-conditioned frequency filtering. The zero-initialized hypernet
        # starts with freq_filter close to 1, matching identity filtering.
        context = x.mean(dim=-2)
        gate = self.freq_hyper(context)  # [B, L, F]
        freq_filter = 1.0 + self.filter_scale * torch.tanh(gate)
        freq_filter = freq_filter.unsqueeze(-1)  # [B, L, F, 1]
        x_freq = x_freq * freq_filter.float()

        # Inverse Fourier transform back to code-depth dimension K.
        x = torch.fft.irfft(
            x_freq,
            n=self.code_len,
            dim=-2,
            norm="backward",
        ).to(dtype)  # [B, L, K, D]

        pool_weight = torch.softmax(self.pool_score(x), dim=-2)  # [B, L, K, 1]
        pooled = (x * pool_weight).sum(dim=-2)  # [B, L, D]

        pooled = self.out_norm(pooled)

        return pooled, freq_filter


class RPGv2(AbstractModel):
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
        self.sid_pool = SIDFourierPooling(
            code_len=self.code_len,
            hidden_size=self.hidden_size,
            dropout=float(model_params.get("sid_pool_dropout", 0.0)),
            filter_scale=float(model_params.get("sid_pool_filter_scale", 0.1)),
            hyper_hidden_size=model_params.get("sid_pool_hyper_hidden_size"),
        )
        self.pred_heads = nn.Sequential(
            *[ResBlock(self.hidden_size) for _ in range(self.n_pred_head)]
        )
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        logger.info(
            "RPGv2 initialized: items=%d, code_len=%d.",
            self.item_id2tokens.shape[0] - 1,
            self.code_len,
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

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_loss: bool = True,
    ) -> torch.Tensor:
        input_tokens = self.item_id2tokens[batch["input_ids"]]
        token_emb = self.gpt2.wte(input_tokens)
        input_embs, _ = self.sid_pool(token_emb)
        outputs = self.gpt2(inputs_embeds=input_embs, attention_mask=batch["attention_mask"])
        hs = outputs.last_hidden_state
        final_states = [self.pred_heads[i](hs).unsqueeze(-2) for i in range(self.n_pred_head)]
        final_states = torch.cat(final_states, dim=-2)
        outputs.final_states = final_states

        if return_loss:
            labels = batch["labels_seq"]
            label_mask = labels.view(-1) != -100
            selected_states = final_states.view(-1, self.n_pred_head, self.hidden_size)[label_mask]
            selected_states = F.normalize(selected_states, dim=-1)
            selected_states_chunks = torch.chunk(selected_states, self.n_pred_head, dim=1)
            token_emb = self.gpt2.wte.weight[1:self.eos_token]
            token_emb = F.normalize(token_emb, dim=-1)
            token_embs_chunks = torch.split(token_emb, self.vocab_sizes, dim=0)
            token_logits = [
                torch.matmul(selected_states_chunks[i].squeeze(dim=1), token_embs_chunks[i].T) / self.temperature
                for i in range(self.n_pred_head)
            ]
            token_labels = self.item_id2tokens[labels.view(-1)[label_mask] + 1]
            losses = [
                self.loss_fct(token_logits[i], token_labels[:, i] - self.bases[i] - 1)
                for i in range(self.n_pred_head)
            ]
            main_loss = torch.mean(torch.stack(losses))
            total_loss = main_loss
            outputs.loss = total_loss
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
