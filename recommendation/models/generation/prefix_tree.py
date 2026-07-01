from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


class Trie:
    """Prefix constraint table for HuggingFace `prefix_allowed_tokens_fn`."""

    def __init__(
        self,
        eos_token_id: Optional[int] = None,
        pad_token_id: int = 0,
        skip_bos: int = 1,
    ) -> None:
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.skip_bos = skip_bos
        self._table: dict[Tuple[int, ...], List[int]] = {}
        self._fallback = [eos_token_id] if eos_token_id is not None else []

    def bulk_insert(self, sequences: Iterable[Sequence[int]]) -> None:
        table: dict[Tuple[int, ...], set[int]] = {}
        n_seq = 0

        for raw_seq in sequences:
            seq = [int(token) for token in raw_seq if token != self.pad_token_id]
            if not seq:
                continue

            n_seq += 1
            if self.eos_token_id is not None:
                seq.append(self.eos_token_id)

            for idx, token in enumerate(seq):
                table.setdefault(tuple(seq[:idx]), set()).add(token)

        self._table = {prefix: sorted(tokens) for prefix, tokens in table.items()}
        logger.info(
            "[Trie] Built from %d sequences, unique prefixes: %d",
            n_seq,
            len(self._table),
        )

    def get_allowed_next_tokens(self, batch_id: int, input_ids) -> List[int]:
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()

        prefix = tuple(int(token) for token in input_ids[self.skip_bos:])
        return self._table.get(prefix, self._fallback)


def build_trie_from_codebook(
    token_sequences: Iterable[Sequence[int]],
    eos_token_id: Optional[int] = None,
) -> Trie:
    trie = Trie(eos_token_id=eos_token_id, pad_token_id=0, skip_bos=1)
    trie.bulk_insert(token_sequences)
    return trie
