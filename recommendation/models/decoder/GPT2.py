import torch
import torch.nn as nn
from typing import Any, Dict, List
import transformers

from ..abstract_model import AbstractModel
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from metrics import recall_at_k, ndcg_at_k

GPT2Config = transformers.GPT2Config
GPT2LMHeadModel = transformers.GPT2LMHeadModel

class GPT2(AbstractModel):
    """
    一个仿照 TIGER 接口的 Decoder-Only 生成式模型。
    它使用 GPT-2 架构从零开始训练，专用於序列推荐任务。
    """
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config)
        
        model_params = config['model_params']
        token_params = config['token_params']

        gpt2config = GPT2Config(
            vocab_size=token_params['vocab_size'],
            n_positions=model_params['max_len'] * config['code_len'] + config['code_len'],
            n_embd=model_params['n_embd'],
            n_layer=model_params['n_layer'],
            n_head=model_params['n_head'],
            n_inner=model_params.get('n_inner', model_params.get('d_ff', 2048)),
            activation_function=model_params.get('activation_function', 'gelu_new'),
            resid_pdrop=model_params.get('resid_pdrop', 0.1),
            embd_pdrop=model_params.get('embd_pdrop', 0.1),
            attn_pdrop=model_params.get('attn_pdrop', 0.1),
            layer_norm_epsilon=float(model_params.get('layer_norm_epsilon', 1e-5)),
            initializer_range=model_params.get('initializer_range', 0.02),
            eos_token_id=token_params['eos_token_id'],
            pad_token_id=token_params['pad_token_id'],
        )

        self.gpt2 = GPT2LMHeadModel(config=gpt2config)
        self.n_params_str = self._calculate_n_parameters()

    @property
    def task_type(self) -> str:
        return 'generative'

    @property
    def n_parameters(self) -> str:
        return self.n_params_str

    def _calculate_n_parameters(self) -> str:
        num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
        total_params = num_params(self.parameters())
        emb_params = num_params(self.gpt2.get_input_embeddings().parameters())
        return (
            f'# Embedding parameters: {emb_params:,}\n'
            f'# Non-embedding parameters: {total_params - emb_params:,}\n'
            f'# Total trainable parameters: {total_params:,}\n'
        )
    
    def forward(self, batch: Dict) -> Dict:
        """
        将 history 和 target code 拼接成一个序列进行自回归训练。
        """
        history_ids = batch['input_ids']
        target_ids = batch['labels']
        history_mask = batch['attention_mask']
        
        combined_ids = torch.cat([history_ids, target_ids], dim=1)
        
        target_mask = torch.ones_like(target_ids)
        combined_mask = torch.cat([history_mask, target_mask], dim=1)

        combined_labels = combined_ids.clone() 
        combined_labels[combined_mask == 0] = -100

        outputs = self.gpt2(
            input_ids=combined_ids,
            attention_mask=combined_mask,
            labels=combined_labels
        )
        return outputs

    def generate(self, **kwargs: Any) -> torch.Tensor:
        return self.gpt2.generate(**kwargs)

    def evaluate_step(self, batch: Dict[str, torch.Tensor], topk_list: List[int]) -> Dict[str, float]:
        """
        返回本批次指标总和和样本数，供 trainer 聚合。
        """
        beam_size = self.config['evaluation_params']['beam_size']
        code_len = self.config['code_len']

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        batch_size = labels.shape[0] 

        preds = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=beam_size,
            num_return_sequences=beam_size,
            max_new_tokens=code_len,
            early_stopping=False,
            pad_token_id=self.config['token_params']['pad_token_id'],
            eos_token_id=None
        )
        
        generated_part = preds[:, input_ids.shape[1]:]
        preds_reshaped = generated_part.view(batch_size, beam_size, -1)
        
        pos_index = self._calculate_pos_index(preds_reshaped, labels, maxk=beam_size)
        
        batch_metrics = {}
        for k in topk_list:
            recall_sum = recall_at_k(pos_index, k).sum().item() 
            ndcg_sum = ndcg_at_k(pos_index, k).sum().item()
            
            batch_metrics[f'Recall@{k}'] = recall_sum 
            batch_metrics[f'NDCG@{k}'] = ndcg_sum
            
        batch_metrics['count'] = float(batch_size) 
          
        return batch_metrics
  
    @staticmethod
    def _calculate_pos_index(preds: torch.Tensor, labels: torch.Tensor, maxk: int) -> torch.Tensor:
        """
        假设 code 总是包含 L-1 个语义层和最后 1 个重复层。
        """
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        B, _, L_pred = preds.shape
        L_label = labels.shape[1]

        # 如果生成长度不足（例如提前遇到 EOS），用 padding 补齐
        if L_pred < L_label:
            padding = torch.zeros((B, maxk, L_label - L_pred), dtype=preds.dtype)
            preds = torch.cat([preds, padding], dim=2)
        # 如果生成长度过长，截断
        elif L_pred > L_label:
            preds = preds[:, :, :L_label]
        
        pos_index = torch.zeros((B, maxk), dtype=torch.bool)
        for i in range(B):
            gt = labels[i]
            gt_semantic = gt[:-1].tolist()
            gt_dup  = int(gt[-1].item())

            for j in range(maxk):
                pj = preds[i, j]
                pj_semantic = pj[:-1].tolist()
                pj_dup  = int(pj[-1].item())

                if pj_semantic == gt_semantic and pj_dup == gt_dup:
                    pos_index[i, j] = True
                    break
        return pos_index
