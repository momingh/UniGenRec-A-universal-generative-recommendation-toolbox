from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from transformers.generation.logits_process import LogitsProcessor


class TIGERLogitsProcessor(LogitsProcessor):
    """
    在生成时屏蔽非法 token，保证每个 beam 都沿着合法 item code 路径前进。

    这里仍然使用 HuggingFace 的 beam search，只是在每个解码步把非法
    next-token 的 score 置为 -inf。
    """

    def __init__(self, item_to_code_map: Dict[int, Sequence[int]], config: Dict):
        self.code_len = int(config["code_len"])
        self.eos_token_id = config["token_params"].get("eos_token_id")
        self.decoder_start_token_id = 0
        self.prefix_to_allowed = self._build_prefix_table(item_to_code_map.values())
        self.layer_token_ids = self._build_layer_token_ids(config)

    def _build_prefix_table(
        self,
        code_sequences: Iterable[Sequence[int]],
    ) -> Dict[Tuple[int, ...], List[int]]:
        table: Dict[Tuple[int, ...], set[int]] = {}
        for code in code_sequences:
            seq = tuple(int(token) for token in code)
            if len(seq) != self.code_len:
                continue

            for pos, token in enumerate(seq):
                table.setdefault(seq[:pos], set()).add(token)

            if self.eos_token_id is not None:
                table.setdefault(seq, set()).add(int(self.eos_token_id))

        return {prefix: sorted(tokens) for prefix, tokens in table.items()}

    @staticmethod
    def _build_layer_token_ids(config: Dict) -> List[List[int]]:
        layer_token_ids: List[List[int]] = []
        for base, size in zip(config["bases"], config["vocab_sizes"]):
            start = int(base) + 1
            stop = start + int(size)
            layer_token_ids.append(list(range(start, stop)))
        return layer_token_ids

    def _allowed_tokens_for_row(self, row_input_ids: torch.Tensor) -> List[int]:
        generated = row_input_ids.tolist()
        prefix = tuple(int(token) for token in generated[1:])

        if len(prefix) >= self.code_len:
            if self.eos_token_id is not None:
                return [int(self.eos_token_id)]
            return []

        allowed = self.prefix_to_allowed.get(prefix)
        if allowed:
            return allowed

        return self.layer_token_ids[len(prefix) % self.code_len]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        masked_scores = scores.new_full(scores.shape, float("-inf"))

        for row_idx in range(input_ids.shape[0]):
            allowed = self._allowed_tokens_for_row(input_ids[row_idx])
            if allowed:
                masked_scores[row_idx, allowed] = scores[row_idx, allowed]

        return masked_scores
