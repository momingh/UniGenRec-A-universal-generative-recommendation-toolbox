import logging
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..abstract_model import AbstractModel
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from metrics import recall_at_k, ndcg_at_k
from tokenizer import build_semantic_special_tokens

logger = logging.getLogger(__name__)


class LCRec(AbstractModel):
    """
    Minimal LC-Rec style baseline:
    keep the LLM tokenizer/backbone, add Semantic ID special tokens,
    and train with instruction tuning.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        item_to_code_map: Dict[int, List[int]],
        **kwargs: Any,
    ) -> None:
        super().__init__(config)
        model_params = config["model_params"]
        model_name_or_path = model_params["model_name_or_path"]
        trust_remote_code = model_params.get("trust_remote_code", False)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        semantic_special_tokens = build_semantic_special_tokens(config["vocab_sizes"])
        num_added_tokens = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": semantic_special_tokens}
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        if num_added_tokens > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.code_len = config["code_len"]
        self.beam_size = config["evaluation_params"]["beam_size"]
        self.item_to_code_map = item_to_code_map
        self.semantic_prefix_trie = self._build_semantic_prefix_trie()

        logger.info(
            "LCRec initialized with %d items and %d semantic special tokens.",
            len(self.item_to_code_map),
            len(semantic_special_tokens),
        )

    @property
    def task_type(self) -> str:
        return "generative"

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"# Total trainable parameters: {total_params:,}"

    def _offset_token_to_special_token(self, token_id: int, level: int) -> str:
        semantic_value = token_id - self.config["bases"][level] - 1
        return f"<SID_{level}_{semantic_value}>"

    def _item_code_to_token_ids(self, code_tokens: List[int]) -> Tuple[int, ...]:
        semantic_tokens = [
            self._offset_token_to_special_token(token_id, level)
            for level, token_id in enumerate(code_tokens)
        ]
        return tuple(
            self.tokenizer.convert_tokens_to_ids(token) for token in semantic_tokens
        )

    def _build_semantic_prefix_trie(self) -> Dict[Tuple[int, ...], List[int]]:
        trie: Dict[Tuple[int, ...], set] = {}
        for code_tokens in self.item_to_code_map.values():
            token_ids = self._item_code_to_token_ids(code_tokens)
            prefix: Tuple[int, ...] = ()
            for token_id in token_ids:
                trie.setdefault(prefix, set()).add(token_id)
                prefix = prefix + (token_id,)
            trie.setdefault(prefix, set()).add(self.tokenizer.eos_token_id)
        return {key: sorted(value) for key, value in trie.items()}

    def _prefix_allowed_tokens_fn(self, prompt_lengths: List[int]):
        batch_size = len(prompt_lengths)
        eos_token_id = self.tokenizer.eos_token_id

        def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> List[int]:
            prompt_len = prompt_lengths[batch_id % batch_size]
            generated_prefix = tuple(input_ids.tolist()[prompt_len:])
            return self.semantic_prefix_trie.get(generated_prefix, [eos_token_id])

        return prefix_allowed_tokens_fn

    def forward(self, batch: Dict[str, torch.Tensor]):
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

    def generate(self, **kwargs: Any) -> torch.Tensor:
        kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        return self.model.generate(**kwargs)

    def save_pretrained(self, save_dir: str) -> None:
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

    def load_pretrained(self, load_dir: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(load_dir)
        self.model = AutoModelForCausalLM.from_pretrained(load_dir)
        self.semantic_prefix_trie = self._build_semantic_prefix_trie()

    def evaluate_step(self, batch: Dict[str, torch.Tensor], topk_list: List[int]) -> Dict[str, float]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        target_token_ids = batch["target_token_ids"]
        batch_size = input_ids.size(0)

        prompt_lengths = attention_mask.sum(dim=1).tolist()
        prefix_allowed_tokens_fn = self._prefix_allowed_tokens_fn(prompt_lengths)

        outputs = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=self.beam_size,
            num_return_sequences=self.beam_size,
            max_new_tokens=self.code_len + 1,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        generated = outputs[:, input_ids.shape[1]:]
        if generated.size(1) < self.code_len:
            padding = torch.full(
                (generated.size(0), self.code_len - generated.size(1)),
                fill_value=-1,
                dtype=generated.dtype,
                device=generated.device,
            )
            generated = torch.cat([generated, padding], dim=1)
        generated = generated[:, : self.code_len]
        generated = generated.view(batch_size, self.beam_size, self.code_len)

        pos_index = self._calculate_pos_index(generated, target_token_ids, self.beam_size)
        metrics: Dict[str, float] = {"count": float(batch_size)}
        for k in topk_list:
            metrics[f"Recall@{k}"] = recall_at_k(pos_index, k).sum().item()
            metrics[f"NDCG@{k}"] = ndcg_at_k(pos_index, k).sum().item()
        return metrics

    @staticmethod
    def _calculate_pos_index(preds: torch.Tensor, labels: torch.Tensor, maxk: int) -> torch.Tensor:
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        batch_size, beam_size, _ = preds.shape
        pos_index = torch.zeros((batch_size, maxk), dtype=torch.bool)

        for batch_idx in range(batch_size):
            target = labels[batch_idx].tolist()
            for beam_idx in range(min(beam_size, maxk)):
                if preds[batch_idx, beam_idx].tolist() == target:
                    pos_index[batch_idx, beam_idx] = True
                    break
        return pos_index
