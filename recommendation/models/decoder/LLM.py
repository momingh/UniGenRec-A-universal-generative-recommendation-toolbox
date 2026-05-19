# 文件路径: recommendation/models/generative/LLM_REC.py (直接数字 ID 版)

import torch
import torch.nn as nn
from typing import Any, Dict, List
import transformers
import logging
from transformers import AutoModelForCausalLM, AutoConfig # 只需要 Config 和 Model

from ..abstract_model import AbstractModel
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from metrics import recall_at_k, ndcg_at_k

logger = logging.getLogger(__name__)

class LLM(AbstractModel):
    """
    【直接数字 ID 版】
    使用预训练 Decoder-Only LLM 架构 (如 Llama, Qwen)，但抛弃其原始 Embedding，
    直接使用 Code Token Offset ID 作为输入，训练一个新的 Embedding 层。
    """
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config)
        
        model_params = config['model_params']
        token_params = config['token_params']
        
        # 1. 获取预训练模型名称或路径
        model_name_or_path = model_params.get('model_name_or_path')
        if not model_name_or_path:
            raise ValueError("設定檔 'model_params' 中必須提供 'model_name_or_path'")
        logger.info(f"載入預訓練模型架構: {model_name_or_path}")

        # 2. 载入预训练模型的 Config (不需要 Tokenizer)
        llm_config = AutoConfig.from_pretrained(model_name_or_path)
        
        # (可选) 根据您的配置文件覆盖 LLM Config 的某些参数
        # 例如，如果您想调整层数或隐藏维度 (但不推荐，失去了预训练的意义)
        # llm_config.n_layer = model_params.get('n_layer', llm_config.n_layer)
        # llm_config.n_embd = model_params.get('n_embd', llm_config.hidden_size) # 注意名称可能不同

        # 3. 载入预训练模型 (不包含 LM Head，因为我们要重置 Embedding)
        # 我们只使用其 Transformer 骨架
        # 注意：不同的模型架构可能需要不同的 from_pretrained 方式，
        # AutoModelForCausalLM 通常適用
        # 为了安全起見，我们先载入完整模型
        self.llm = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=llm_config)

        # 4. ✅ 关键：徹底重置 Embedding 和 LM Head
        # 获取 Code Token 的总词表大小
        code_vocab_size = token_params['vocab_size']
        logger.info(f"將重置模型 Embedding 層和 LM Head 大小至: {code_vocab_size}")
        self.llm.resize_token_embeddings(code_vocab_size) 
        # 注意：resize_token_embeddings 会自动处理 Embedding 层和输出层 (LM Head)
        # 新的 Embedding 会被随机初始化

        # 5. 存储 PAD 和 EOS ID (来自我们的 Code Token 定义)
        self._pad_id = token_params['pad_token_id']
        self._eos_id = token_params['eos_token_id']

        self.n_params_str = self._calculate_n_parameters()
        self.code_len = config['code_len']

    @property
    def task_type(self) -> str:
        return 'generative'

    @property
    def n_parameters(self) -> str:
        # 现在的 Embedding 层会小得多
        return self._calculate_n_parameters()

    def _calculate_n_parameters(self) -> str:
        num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
        total_params = num_params(self.parameters())
        # 注意：现在 get_input_embeddings() 返回的是我们 resize 后的新 embedding 层
        emb_params = num_params(self.llm.get_input_embeddings().parameters())
        return (f'# Embedding parameters: {emb_params:,}\n' f'# Non-embedding parameters: {total_params - emb_params:,}\n' f'# Total trainable parameters: {total_params:,}\n')

    def forward(self, batch: Dict) -> Dict:
        """
        处理方式与 SIMPLE_GPT 完全相同。
        """
        history_ids = batch['input_ids']      # (B, L_hist_flat) - Offset IDs
        target_ids = batch['labels']        # (B, L_target) - Offset IDs
        history_mask = batch['attention_mask'] # (B, L_hist_flat) - Token-level mask
        
        # 1. 拼接输入序列
        combined_ids = torch.cat([history_ids, target_ids], dim=1)
        
        # 2. 创建拼接后的 attention mask
        target_mask = torch.ones_like(target_ids)
        combined_mask = torch.cat([history_mask, target_mask], dim=1)

        # 3. 创建用於计算 loss 的 labels
        history_labels = torch.full_like(history_ids, -100)
        combined_labels = torch.cat([history_labels, target_ids], dim=1)

        # 4. 传給 LLM 模型 (现在它接收的是 Offset IDs)
        outputs = self.llm(
            input_ids=combined_ids,
            attention_mask=combined_mask,
            labels=combined_labels
        )
        return outputs

    def generate(self, **kwargs: Any) -> torch.Tensor:
        """执行 LLM 的标準生成 (使用 Code Token IDs)"""
        # 使用我们 Code Token 的 PAD/EOS ID
        kwargs.setdefault("pad_token_id", self._pad_id)
        kwargs.setdefault("eos_token_id", self._eos_id)
        # input_ids 应該是 history_ids (Offset IDs)
        return self.llm.generate(**kwargs)

    def evaluate_step(self, batch: Dict[str, torch.Tensor], topk_list: List[int]) -> Dict[str, float]:
        """
        评估逻辑与 SIMPLE_GPT 完全相同。
        """
        beam_size = self.config['evaluation_params']['beam_size']
        code_len = self.code_len

        input_ids = batch['input_ids']         # History Offset IDs
        attention_mask = batch['attention_mask'] # History Token-level Mask
        labels = batch['labels']               # Target Offset IDs
        device = input_ids.device

        # 1. 生成 (输入是 History Offset IDs)
        preds = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=beam_size,
            num_return_sequences=beam_size,
            max_new_tokens=code_len,
            early_stopping=False,
        )
        
        # 2. 后处理 (切掉 prompt 部分)
        generated_part = preds[:, input_ids.shape[1]:] # 得到生成的 Offset IDs
        preds_reshaped = generated_part.view(input_ids.shape[0], beam_size, -1)
        
        # 3. 计算命中 (直接使用 Offset IDs)
        pos_index = self._calculate_pos_index(preds_reshaped, labels, maxk=beam_size)
        pos_index = pos_index.to(device)
        
        # 4. 计算指标
        batch_metrics = {}
        for k in topk_list:
            recall = recall_at_k(pos_index, k).mean().item()
            ndcg = ndcg_at_k(pos_index, k).mean().item()
            batch_metrics[f'Recall@{k}'] = recall
            batch_metrics[f'NDCG@{k}'] = ndcg
            
        return batch_metrics
  
    # _calculate_pos_index 可以保持与 TIGER 一致 (假设有 dup 层)
    @staticmethod
    def _calculate_pos_index(preds: torch.Tensor, labels: torch.Tensor, maxk: int) -> torch.Tensor:
        # ... (与 TIGER 版本相同) ...
        preds = preds.detach().cpu(); labels = labels.detach().cpu()
        B, K, L_pred = preds.shape; L_label = labels.shape[1]
        if L_pred < L_label: padding = torch.full((B, K, L_label - L_pred), 0, dtype=preds.dtype); preds = torch.cat([preds, padding], dim=2)
        elif L_pred > L_label: preds = preds[:, :, :L_label]
        L = L_label
        pos_index = torch.zeros((B, maxk), dtype=torch.bool)
        has_dup_layer = True 
        for i in range(B):
            gt = labels[i]
            if L == 0: continue
            if has_dup_layer and L > 1: gt_semantic, gt_dup = gt[:-1].tolist(), int(gt[-1].item())
            else: gt_semantic, gt_dup = gt.tolist(), 0
            for j in range(min(K, maxk)):
                pj = preds[i, j]
                if has_dup_layer and L > 1: pj_semantic, pj_dup = pj[:-1].tolist(), int(pj[-1].item())
                else: pj_semantic, pj_dup = pj.tolist(), float('inf')
                if pj_semantic == gt_semantic and pj_dup >= gt_dup: pos_index[i, j] = True; break
        return pos_index
