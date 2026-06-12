from typing import Any, Dict, List, Optional
import torch
import logging
logger = logging.getLogger(__name__)
import transformers

import sys
from pathlib import Path
root = Path(__file__).resolve().parents[3]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from recommendation.metrics import recall_at_k, ndcg_at_k
from recommendation.models.generation.prefix_tree import Trie
from recommendation.models.generation.logits_processor import TIGERLogitsProcessor
from recommendation.models.abstract_model import AbstractModel
from transformers.generation.logits_process import LogitsProcessorList



T5ForConditionalGeneration = transformers.T5ForConditionalGeneration
T5Config = transformers.T5Config


class TIGER(AbstractModel):
  
  def __init__(
      self,
      config: Dict[str, Any],
      prefix_trie: Optional[Trie] = None,
      item_to_code_map: Optional[Dict[int, List[int]]] = None
  ):
    super().__init__(config)
    model_params = config['model_params']
    token_params = config['token_params']
    t5_model_params = {
        k: v for k, v in model_params.items()
        if k != 'num_epochs'
    }
    t5config = T5Config(
        **t5_model_params,
        **token_params,
        decoder_start_token_id=0,
        use_cache=True  # 生成时缓存 key/value,避免重复计算 attention
    )

    self.t5 = T5ForConditionalGeneration(config=t5config)
    self.t5.resize_token_embeddings(config['token_params']['vocab_size'])
    self.n_params_str = self._calculate_n_parameters()

    self._item_to_code_map = item_to_code_map
    if item_to_code_map:
        logger.info(f"TIGER 已接收 item_to_code_map,包含 {len(item_to_code_map)} 个 item。")

    eval_params = config.get('evaluation_params', {})
    self.use_logits_processor = bool(
        eval_params.get('use_logits_processor', eval_params.get('use_prefix_trie', False))
    )
    self.logits_processor = None
    if self.use_logits_processor and item_to_code_map:
        self.logits_processor = TIGERLogitsProcessor(item_to_code_map, config)
        logger.info("TIGER 已启用 LogitsProcessor 约束解码。")
    elif self.use_logits_processor:
        logger.warning("TIGER 请求启用 LogitsProcessor,但未收到 item_to_code_map。")
    else:
        logger.info("TIGER 未启用约束解码。")

    if prefix_trie is not None:
        logger.info("TIGER 收到 prefix_trie,但当前实现使用 LogitsProcessor 约束解码。")

  @property
  def task_type(self) -> str:
        return 'generative'

  @property
  def n_parameters(self) -> str:
    return self.n_params_str

  def _calculate_n_parameters(self) -> str:
    num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
    total_params = num_params(self.parameters())
    emb_params = num_params(self.t5.get_input_embeddings().parameters())
    return (
        f'# Embedding parameters: {emb_params:,}\n'
        f'# Non-embedding parameters: {total_params - emb_params:,}\n'
        f'# Total trainable parameters: {total_params:,}\n'
    )
  
  def forward(self, batch: Dict) -> Dict:
        t5_known_args = {'input_ids', 'attention_mask', 'labels'}
        t5_inputs = {key: value for key, value in batch.items() if key in t5_known_args}
        return self.t5(**t5_inputs)

  def generate(self, **kwargs: Any) -> torch.Tensor:
    if self.logits_processor is not None:
        existing_processors = kwargs.pop('logits_processor', None)
        if existing_processors is None:
            processors = LogitsProcessorList()
        else:
            processors = LogitsProcessorList(list(existing_processors))
        processors.append(self.logits_processor)
        kwargs['logits_processor'] = processors

    return self.t5.generate(**kwargs)

  def evaluate_step(self, batch: Dict[str, torch.Tensor], topk_list: List[int]) -> Dict[str, float]:
    """
    封装 TIGER 的评估逻辑:生成 -> 匹配 -> 计算指标。
    返回本批次指标的总和(sum)和样本数(count),供 trainer 聚合。
    """
    beam_size = self.config['evaluation_params']['beam_size']
    code_len = self.config['code_len']
    max_k = max(topk_list)

    # 边界检查:topk 不能超过 beam_size,否则 recall@k 会静默截断导致指标偏低
    assert max_k <= beam_size, \
        f"topk_list 最大值 {max_k} 超过 beam_size {beam_size}"

    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']  # (batch_size, code_len)
    device = input_ids.device
    batch_size = input_ids.shape[0]

    # 1. 生成:beam search 输出 (batch_size * beam_size, 1 + code_len)
    preds = self.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_beams=beam_size,
        num_return_sequences=beam_size,
        max_new_tokens=code_len,
        early_stopping=False
    )

    # 2. 跳过 decoder_start_token(位置0),取 code_len 个生成 token 后 reshape
    #    -> (batch_size, beam_size, code_len)
    preds = preds[:, 1:1 + code_len].view(batch_size, beam_size, code_len)

    # 3. 命中矩阵 (batch_size, beam_size)
    pos_index = self._calculate_pos_index(preds, labels, maxk=beam_size).to(device)

    # 4. 累加指标(返回 sum,由 trainer 统一除以总样本数)
    batch_metrics = {'count': batch_size}
    for k in topk_list:
        batch_metrics[f'Recall@{k}'] = recall_at_k(pos_index, k).sum().item()
        batch_metrics[f'NDCG@{k}'] = ndcg_at_k(pos_index, k).sum().item()

    return batch_metrics
  
  @staticmethod
  def _calculate_pos_index(preds: torch.Tensor, labels: torch.Tensor, maxk: int) -> torch.Tensor:
        """
        计算命中矩阵:对每个样本,标记哪个 beam 完整命中了 ground-truth code。

        Args:
            preds:  (B, maxk, L_pred) 每个样本 maxk 个 beam 的预测 code
            labels: (B, L_label)      每个样本的 ground-truth code
            maxk:   beam 数量

        Returns:
            (B, maxk) 的 bool 张量。每行最多一个 True(取排名最靠前的命中 beam),
            因为单正例场景下命中即停。
        """
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        L_pred = preds.shape[2]
        L_label = labels.shape[1]

        # 对齐长度:生成不足则补 0,过长则截断(正常 max_new_tokens=code_len 时两者相等)
        if L_pred < L_label:
            padding = torch.zeros((preds.shape[0], maxk, L_label - L_pred), dtype=preds.dtype)
            preds = torch.cat([preds, padding], dim=2)
        elif L_pred > L_label:
            preds = preds[:, :, :L_label]

        # 整段逐 token 比较:(B, maxk, L) == (B, 1, L) -> 全部 token 相等才算命中
        # full_match: (B, maxk) bool
        full_match = (preds == labels.unsqueeze(1)).all(dim=2)

        # 只保留每行排名最靠前的那个命中(单正例:命中即停)
        # cumsum 后第一个 True 处累计值为 1,之后的命中被置 False
        first_hit = full_match & (full_match.cumsum(dim=1) == 1)
        return first_hit
