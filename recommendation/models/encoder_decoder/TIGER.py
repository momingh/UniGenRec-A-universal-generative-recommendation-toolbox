import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import T5Config, T5ForConditionalGeneration
from transformers.generation.logits_process import LogitsProcessorList

root = Path(__file__).resolve().parents[3]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from recommendation.metrics import ndcg_at_k, recall_at_k
from recommendation.models.abstract_model import AbstractModel
from recommendation.models.generation.logits_processor import TIGERLogitsProcessor

logger = logging.getLogger(__name__)


T5_CONFIG_KEYS = {
    "d_model",
    "d_kv",
    "d_ff",
    "num_layers",
    "num_decoder_layers",
    "num_heads",
    "relative_attention_num_buckets",
    "relative_attention_max_distance",
    "dropout_rate",
    "layer_norm_epsilon",
    "initializer_factor",
    "feed_forward_proj",
    "tie_word_embeddings",
}
T5_INPUT_KEYS = {"input_ids", "attention_mask", "labels"}


class TIGER(AbstractModel):
    def __init__(
        self,
        config: Dict[str, Any],
        prefix_trie: Optional[Any] = None,
        item_to_code_map: Optional[Dict[int, List[int]]] = None,
    ):
        super().__init__(config)

        self.t5 = T5ForConditionalGeneration(config=self._build_t5_config(config))
        self.logits_processor = self._build_logits_processor(config, item_to_code_map)
        self.n_params_str = self._format_trainable_params()

        if item_to_code_map:
            logger.info("TIGER 已接收 item_to_code_map,包含 %d 个 item。", len(item_to_code_map))
        if prefix_trie is not None:
            logger.info("TIGER 收到 prefix_trie,但当前实现使用 LogitsProcessor 约束解码。")

    @property
    def task_type(self) -> str:
        return "generative"

    @property
    def n_parameters(self) -> str:
        return self.n_params_str

    @staticmethod
    def _build_t5_config(config: Dict[str, Any]) -> T5Config:
        model_params = {
            key: value
            for key, value in config["model_params"].items()
            if key in T5_CONFIG_KEYS
        }
        return T5Config(
            **model_params,
            **config["token_params"],
            decoder_start_token_id=0,
            use_cache=True,
        )

    @staticmethod
    def _build_logits_processor(
        config: Dict[str, Any],
        item_to_code_map: Optional[Dict[int, List[int]]],
    ) -> Optional[TIGERLogitsProcessor]:
        eval_params = config.get("evaluation_params", {})
        enabled = bool(
            eval_params.get("use_logits_processor", eval_params.get("use_prefix_trie", False))
        )

        if enabled and item_to_code_map:
            logger.info("TIGER 已启用 LogitsProcessor 约束解码。")
            return TIGERLogitsProcessor(item_to_code_map, config)
        if enabled:
            logger.warning("TIGER 请求启用 LogitsProcessor,但未收到 item_to_code_map。")
        else:
            logger.info("TIGER 未启用约束解码。")
        return None

    def _format_trainable_params(self) -> str:
        def count(params):
            return sum(param.numel() for param in params if param.requires_grad)

        total = count(self.parameters())
        embedding = count(self.t5.get_input_embeddings().parameters())
        return (
            f"# Embedding parameters: {embedding:,}\n"
            f"# Non-embedding parameters: {total - embedding:,}\n"
            f"# Total trainable parameters: {total:,}\n"
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Any:
        output = self.t5(**{key: value for key, value in batch.items() if key in T5_INPUT_KEYS})
        return output

    def generate(self, **kwargs: Any) -> torch.Tensor:
        if self.logits_processor is not None:
            existing = kwargs.pop("logits_processor", None)
            processors = LogitsProcessorList([] if existing is None else list(existing))
            processors.append(self.logits_processor)
            kwargs["logits_processor"] = processors
        return self.t5.generate(**kwargs)

    def evaluate_step(self, batch: Dict[str, torch.Tensor], topk_list: List[int]) -> Dict[str, float]:
        beam_size = self.config["evaluation_params"]["beam_size"]
        code_len = self.config["code_len"]
        if max(topk_list) > beam_size:
            raise ValueError(f"topk_list 最大值 {max(topk_list)} 超过 beam_size {beam_size}")

        input_ids = batch["input_ids"]
        preds = self.generate(
            input_ids=input_ids,
            attention_mask=batch["attention_mask"],
            num_beams=beam_size,
            num_return_sequences=beam_size,
            max_new_tokens=code_len,
            early_stopping=False,
        )
        preds = preds[:, 1:1 + code_len].view(input_ids.shape[0], beam_size, code_len)
        pos_index = self._calculate_pos_index(preds, batch["labels"])

        metrics = {"count": input_ids.shape[0]}
        for k in topk_list:
            metrics[f"Recall@{k}"] = recall_at_k(pos_index, k).sum().item()
            metrics[f"NDCG@{k}"] = ndcg_at_k(pos_index, k).sum().item()
        return metrics

    @staticmethod
    def _calculate_pos_index(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()

        pred_len, label_len = preds.shape[2], labels.shape[1]
        if pred_len < label_len:
            padding = torch.zeros(
                (preds.shape[0], preds.shape[1], label_len - pred_len),
                dtype=preds.dtype,
            )
            preds = torch.cat([preds, padding], dim=2)
        elif pred_len > label_len:
            preds = preds[:, :, :label_len]

        full_match = (preds == labels.unsqueeze(1)).all(dim=2)
        return full_match & (full_match.cumsum(dim=1) == 1)
