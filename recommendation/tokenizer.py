import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)

class BaseTokenizer:
    """Tokenizer 基类 (保持不变)"""
    def __init__(self, config: Dict[str, Any], item_to_code_map: Dict[int, List[int]]):
        self.config = config
        self.item_to_code_map = item_to_code_map
        self.pad_token_id = config['token_params']['pad_token_id']
        self.mask_token_id = config['token_params'].get('mask_token_id')
        self.cls_token_id = config['token_params'].get('cls_token_id')
        self.sep_token_id = config['token_params'].get('sep_token_id')
        self.code_len = config['code_len']
        self.item_pad_id = 0
        self.code_pad_list = [self.pad_token_id] * self.code_len
        self.max_len = config['model_params']['max_len']

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def _pad_sequences(self, sequences: List[torch.Tensor], padding_side: str, padding_value: int) -> torch.Tensor:
        """
        辅助函数：执行 Padding (使用更可靠的 left-padding)。
        """
        # sequences: list of tensors, where each tensor is a flattened sequence of valid codes

        # 获取每个序列的实际长度
        lengths = [len(s) for s in sequences]
        max_len = max(lengths) if lengths else 0

        if padding_side == 'right':
            padded_sequences = []
            for s in sequences:
                pad_len = max_len - len(s)
                # torch.nn.functional.pad only supports same padding on both sides easily
                # Manual padding is clearer here
                if pad_len > 0:
                     padding = torch.full((pad_len,), padding_value, dtype=s.dtype)
                     padded_sequences.append(torch.cat((s, padding)))
                else:
                     padded_sequences.append(s)
            if padded_sequences:
                 return torch.stack(padded_sequences)
            else:
                 return torch.empty((0, max_len), dtype=torch.long) # Handle empty batch

        elif padding_side == 'left':
            padded_sequences = []
            for s in sequences:
                pad_len = max_len - len(s)
                if pad_len > 0:
                     padding = torch.full((pad_len,), padding_value, dtype=s.dtype)
                     padded_sequences.append(torch.cat((padding, s)))
                else:
                     padded_sequences.append(s)
            if padded_sequences:
                 return torch.stack(padded_sequences)
            else:
                 return torch.empty((0, max_len), dtype=torch.long) # Handle empty batch
        else:
            raise ValueError(f"不支持的 padding_side: {padding_side}")

class GenerativeTokenizer(BaseTokenizer):
    """
    为 TIGER 和 GPT-2 准备数据。
    核心：
    1. JSONL 中的 item ID 是 0-based，0 是合法 item。
    2. 查 codebook 前统一转换为 1-based item ID。
    3. 压平 code token 后再用 code PAD token 填充。
    """
    def __init__(self, config: Dict[str, Any], item_to_code_map: Dict[int, List[int]], padding_side: str):
        super().__init__(config, item_to_code_map)
        if padding_side not in ('left', 'right'):
            raise ValueError("padding_side 必須是 'left' 或 'right'")
        self.padding_side = padding_side
        logger.info(f"GenerativeTokenizer 初始化, padding_side='{padding_side}'")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        batch_sequences_flat_valid = [] # 存储每个样本压平后的 *有效* code token
        target_codes = []

        logger.debug(f"--- Tokenizer Start Batch (Size: {len(batch)}) ---")

        for i, item in enumerate(batch):
            hist_ids_0based = item['history']
            tgt_id_0based = item['target']

            logger.debug(f"Sample {i} Raw History Item IDs: {hist_ids_0based}")

            # 0 is a valid raw item id. Padding only exists after shifting item ids by +1.
            shifted_hist_ids = [int(x) + 1 for x in hist_ids_0based]
            logger.debug(f"Sample {i} Shifted History Item IDs: {shifted_hist_ids}")

            # 2. 将有效 Item IDs 转换为 Code 列表
            valid_hist_codes = []
            if shifted_hist_ids:
                valid_hist_codes = [
                    self.item_to_code_map.get(item_id, self.code_pad_list)
                    for item_id in shifted_hist_ids
                ]

            # 3. 压平 *有效* Code Token 序列
            seq_flat_valid = [code for item_codes in valid_hist_codes for code in item_codes]

            logger.debug(f"Sample {i} Flattened Valid Codes (len={len(seq_flat_valid)}): {seq_flat_valid[:20]}...{seq_flat_valid[-20:]}")

            batch_sequences_flat_valid.append(torch.tensor(seq_flat_valid, dtype=torch.long))

            t_code = self.item_to_code_map.get(int(tgt_id_0based) + 1, self.code_pad_list)
            target_codes.append(t_code)

        # 4. ✅ 对压平后的 *有效* 序列进行 Padding (使用修正后的函数)
        padded_histories = self._pad_sequences(
            batch_sequences_flat_valid,
            padding_side=self.padding_side,
            padding_value=self.pad_token_id
        )
        logger.debug(f"Padded Histories Shape: {padded_histories.shape}")
        if len(batch) > 0 and padded_histories.numel() > 0: # Add check for empty tensor
            logger.debug(f"Padded Histories[0] (first 30): {padded_histories[0][:30].tolist()}")
            logger.debug(f"Padded Histories[0] (last 30): {padded_histories[0][-30:].tolist()}")

        # 5. 生成 Attention Mask
        attention_masks = (padded_histories != self.pad_token_id).long()
        if len(batch) > 0 and attention_masks.numel() > 0: # Add check for empty tensor
            logger.debug(f"Attention Mask[0] (first 30): {attention_masks[0][:30].tolist()}")
            logger.debug(f"Attention Mask[0] (last 30): {attention_masks[0][-30:].tolist()}")

        # 6. Target Tensor (保持不变)
        target_codes_tensor = torch.tensor(target_codes, dtype=torch.long)

        logger.debug(f"--- Tokenizer End Batch ---")

        # 预期输出:
        # GPT-2 (left): input_ids=[0,...,0, C6.., C11..] mask=[0,...,0, 1.., 1..]
        # TIGER (right): input_ids=[C6.., C11.., 0,...,0] mask=[1.., 1.., 0,...,0]

        return {
            'input_ids': padded_histories,
            'attention_mask': attention_masks,
            'labels': target_codes_tensor,
        }

# --- RetrievalTokenizer 和 get_tokenizer 保持不变 ---
# --- RetrievalTokenizer (✅ 已补全) ---
class RetrievalTokenizer(BaseTokenizer):
    """
    为 RPG 模型准备数据。
    RPG 数据集返回完整 item_seq，并在 collate 时按原 RPG 逻辑展开训练窗口。
    RPG (forward) 期望:
        - 'input_ids': 1-based item IDs，右侧用 0 padding
        - 'attention_mask': item-level mask
        - 'labels_seq': 0-based target item IDs，无目标位置为 -100
    RPG (evaluate_step) 期望:
        - 'target_ids': 0-based target item IDs
    """
    def __init__(self, config: Dict[str, Any], item_to_code_map: Dict[int, List[int]]):
        super().__init__(config, item_to_code_map)
        logger.info(f"RetrievalTokenizer (RPG) 初始化, max_len={self.max_len}")

    def _build_example(self, item_seq: List[int], label_all_positions: bool) -> Dict[str, List[int] | int]:
        input_items = item_seq[:-1]
        target_items = item_seq[1:]
        seq_len = len(input_items)
        pad_len = self.max_len - seq_len

        input_ids = [item_id + 1 for item_id in input_items] + [self.item_pad_id] * pad_len
        attention_mask = [1] * seq_len + [0] * pad_len
        labels = [-100] * self.max_len
        if label_all_positions:
            labels[:seq_len] = target_items
        elif seq_len > 0:
            labels[seq_len - 1] = item_seq[-1]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels_seq": labels,
            "target_id": item_seq[-1],
        }

    def _tokenize_item_seq(self, item_seq: List[int], mode: str) -> List[Dict[str, List[int] | int]]:
        if len(item_seq) < 2:
            return []

        if mode == "train":
            n_return_examples = max(len(item_seq) - self.max_len, 1)
            first_window = item_seq[: min(len(item_seq), self.max_len + 1)]
            examples = [self._build_example(first_window, label_all_positions=True)]

            for start in range(1, n_return_examples):
                window = item_seq[start : start + self.max_len + 1]
                examples.append(self._build_example(window, label_all_positions=False))
            return examples

        eval_window = item_seq[-(self.max_len + 1):]
        return [self._build_example(eval_window, label_all_positions=False)]

    def _tokenize_history_target(self, item: Dict[str, Any]) -> Dict[str, List[int] | int]:
        history = [int(x) for x in item["history"]]
        item_seq = history[-self.max_len:] + [int(item["target"])]
        return self._build_example(item_seq, label_all_positions=False)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        examples = []
        for item in batch:
            if "item_seq" in item:
                item_seq = [int(x) for x in item["item_seq"]]
                if "label_all_positions" in item:
                    examples.append(
                        self._build_example(
                            item_seq,
                            label_all_positions=bool(item["label_all_positions"]),
                        )
                    )
                else:
                    examples.extend(
                        self._tokenize_item_seq(
                            item_seq,
                            item.get("mode", "train"),
                        )
                    )
            else:
                examples.append(self._tokenize_history_target(item))

        if not examples:
            raise ValueError("RPG tokenizer received an empty batch after tokenization.")

        return {
            "input_ids": torch.tensor([example["input_ids"] for example in examples], dtype=torch.long),
            "attention_mask": torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long),
            "labels_seq": torch.tensor([example["labels_seq"] for example in examples], dtype=torch.long),
            "target_ids": torch.tensor([example["target_id"] for example in examples], dtype=torch.long),
        }


def get_tokenizer(model_name: str, config: Dict[str, Any], item_to_code_map: Dict[int, List[int]]) -> Callable:
    """
    (保持之前的版本不变)
    """
    model_name = model_name.upper()
    if model_name == 'TIGER':
        return GenerativeTokenizer(config, item_to_code_map, padding_side='right')
    elif 'GPT2' in model_name or 'LLM' in model_name:
        return GenerativeTokenizer(config, item_to_code_map, padding_side='left')
    elif model_name == 'RPG':
        return RetrievalTokenizer(config, item_to_code_map)
    else:
        logger.warning(f"未知模型: {model_name}，使用 left-padding GenerativeTokenizer")
        return GenerativeTokenizer(config, item_to_code_map, padding_side='left')
