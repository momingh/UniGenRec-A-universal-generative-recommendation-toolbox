import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import BertConfig, BertForMaskedLM

from recommendation.metrics import (
    recall_at_k,
    ndcg_at_k,
    calculate_diversity_at_n,
    calculate_alpha_ndcg_at_k,
    compute_scd,
    compute_weighted_scd,
)
from recommendation.models.abstract_model import AbstractModel
from recommendation.models.generation.prefix_tree import Trie


logger = logging.getLogger(__name__)


class AdaDiff(AbstractModel):
    """
    基於離散擴散的可控生成推薦模型。
    - Backbone: BERT (雙向 Encoder)
    - 訓練: 掩碼語言模型損失 (只計算 labels != -100)
    - 推理: 迭代去噪 + 多樣性制導 + 前缀樹約束
    """

    def __init__(
        self,
        config: Dict[str, Any],
        item_to_code_map: Optional[Dict[int, List[int]]] = None,
        code_to_item_map: Optional[Dict[Tuple[int, ...], int]] = None,
        item_to_cate_map: Optional[Dict[int, Any]] = None,
        prefix_trie: Optional[Trie] = None,
        **kwargs,
    ):
        super().__init__(config)
        model_params = config["model_params"]
        token_params = config["token_params"]

        bert_config = BertConfig(
            vocab_size=token_params["vocab_size"],
            hidden_size=model_params.get("hidden_size", 256),
            num_hidden_layers=model_params.get("num_hidden_layers", 4),
            num_attention_heads=model_params.get("num_attention_heads", 8),
            intermediate_size=model_params.get("intermediate_size", 1024),
            hidden_dropout_prob=model_params.get("dropout_rate", 0.2),
            attention_probs_dropout_prob=model_params.get("dropout_rate", 0.2),
            max_position_embeddings=model_params.get("max_position_embeddings", 512),
            is_decoder=False,
            add_pooling_layer=False,
        )

        self.backbone = BertForMaskedLM(config=bert_config)
        # 確保完全隨機初始化，禁止使用預訓練權重
        self.backbone.apply(self._init_weights)

        # 保存一些常用的配置
        self.code_len = config["code_len"]
        self.pad_token_id = token_params["pad_token_id"]
        self.mask_token_id = token_params["mask_token_id"]
        self.cls_token_id = token_params["cls_token_id"]
        self.sep_token_id = token_params["sep_token_id"]
        self.vocab_size = token_params["vocab_size"]
        self.vocab_sizes = config["vocab_sizes"]
        self.bases = config["bases"]
        
        # level 范圍：用於 prefix 判斷
        level_starts = [b + 1 for b in self.bases]
        level_ends = [self.bases[i] + self.vocab_sizes[i] for i in range(len(self.vocab_sizes))]
        self.register_buffer("level_starts", torch.tensor(level_starts, dtype=torch.long), persistent=False)
        self.register_buffer("level_ends", torch.tensor(level_ends, dtype=torch.long), persistent=False)
        
        self.code_to_item_map = code_to_item_map or {}
        self.item_to_cate_map, self.item_to_cates_map = self._build_item_cate_maps(
            item_to_cate_map, item_to_code_map
        )
        self.num_semantic_categories = len(set(self.item_to_cate_map.values())) if self.item_to_cate_map else 0
        
        # 構建類別映射 Tensor
        max_item_id = max(self.item_to_cate_map.keys()) if self.item_to_cate_map else 0
        cate_tensor = torch.full((max_item_id + 1,), -1, dtype=torch.long)
        for iid, cate in self.item_to_cate_map.items():
            if iid <= max_item_id:
                cate_tensor[iid] = int(cate)
        self.register_buffer("item_category_tensor", cate_tensor, persistent=False)
        
        # 構建 Level Token Mask
        target_len = self.code_len
        allowed = torch.zeros(target_len, self.vocab_size, dtype=torch.bool)
        for i in range(target_len):
            start = self.bases[i] + 1
            end = self.bases[i] + self.vocab_sizes[i] + 1
            allowed[i, start:end] = True
        self.register_buffer("level_token_mask", allowed, persistent=False)
        
        # 前缀樹約束（核心）
        self.prefix_trie = prefix_trie or kwargs.get("prefix_trie")
        if self.prefix_trie is not None:
            logger.info("AdaDiff: Prefix trie loaded, logits will be masked by valid prefixes during decoding.")

        eval_params = config.get("evaluation_params", {})
        self.beam_size = int(eval_params.get("beam_size", 20))
        self.lambda_div = float(
            eval_params.get("lambda_div", model_params.get("lambda_div", 0.0))
        )
        # 採樣階段的獨立制導係數
        self.lambda_div_sampling = float(
            eval_params.get(
                "lambda_div_sampling",
                0.0 if self.lambda_div > 0 else self.lambda_div,
            )
        )
        self.temperature = float(
            eval_params.get("temperature", eval_params.get("sampling_temperature", 0.3))
        )
        
        # 參數自適應調整
        if self.lambda_div > 0 and self.temperature < 0.5:
            logger.info(f"[Init] Auto-adjusting temperature from {self.temperature} to 0.5 for diversity guidance.")
            self.temperature = 0.5
            
        self.top_k = int(eval_params.get("top_k", eval_params.get("topk_sampling", 10)))
        min_topk = max(self.beam_size * 2, 32)
        if self.top_k < min_topk:
            self.top_k = min_topk
            
        self.num_inference_steps = int(
            eval_params.get(
                "num_inference_steps",
                eval_params.get("diffusion_steps", model_params.get("diffusion_steps", 4)),
            )
        )
        self.alpha_diversity = float(eval_params.get("alpha_diversity", 0.5))
        self.diversity_topk = int(eval_params.get("diversity_topk", min(10, self.beam_size)))

    @property
    def task_type(self) -> str:
        return "generative"

    def _init_weights(self, module: nn.Module):
        """BERT 標準初始化"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))

        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def evaluate_step(
        self, batch: Dict[str, torch.Tensor], topk_list: List[int]
    ) -> Dict[str, float]:
        """
        迭代去噪 + 多樣性制導評估
        """
        device = batch["input_ids"].device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_codes = batch["target_codes"].to(device)

        batch_size, seq_len = input_ids.shape
        target_len = self.code_len
        target_start = seq_len - target_len
        mask_value = -1e9

        # 1) 多樣性懲罰計算
        diversity_penalty = self._calculate_diversity_penalty(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_start=target_start,
        )

        # 2) 構造 prefix mask (Target 區域的前兩級)
        prefix_mask = torch.zeros(seq_len, device=device, dtype=torch.float)
        penalty_offset = 1 if self.lambda_div > 0 else 0
        penalty_len = min(2 if penalty_offset == 0 else 1, max(0, target_len - penalty_offset))
        if penalty_len > 0:
            start = target_start + penalty_offset
            prefix_mask[start : start + penalty_len] = 1.0
        prefix_mask = prefix_mask.unsqueeze(0)

        # 3) 擴展 Batch 進行 Beam Search
        expanded_ids = input_ids.clone()
        expanded_ids[:, target_start:] = self.mask_token_id
        expanded_ids = (
            expanded_ids.unsqueeze(1)
            .repeat(1, self.beam_size, 1)
            .view(batch_size * self.beam_size, seq_len)
        )
        expanded_mask = (
            attention_mask.unsqueeze(1)
            .repeat(1, self.beam_size, 1)
            .view(batch_size * self.beam_size, seq_len)
        )
        expanded_penalty = diversity_penalty.unsqueeze(1).repeat(1, self.beam_size, 1).view(
            batch_size * self.beam_size, -1
        )
        expanded_prefix_mask = prefix_mask.repeat(batch_size * self.beam_size, 1)
        dead_beam_mask = torch.zeros(
            batch_size * self.beam_size, device=device, dtype=torch.bool
        )

        # 4) 迭代去噪循環
        for t in range(self.num_inference_steps):
            current_target = expanded_ids[:, target_start:]
            prefix_lens: Optional[torch.Tensor] = None
            allowed_next: List[List[int]] = []
            prefix_dead = torch.zeros_like(dead_beam_mask)
            
            if self.prefix_trie is not None:
                prefix_lens, allowed_next, prefix_dead = self._build_prefix_state(current_target)
                dead_beam_mask |= prefix_dead

            outputs = self.backbone(
                input_ids=expanded_ids,
                attention_mask=expanded_mask,
                return_dict=True,
            )
            logits = outputs.logits

            # Step B: 注入制導
            guided_logits = logits - self.lambda_div_sampling * expanded_penalty.unsqueeze(1) * expanded_prefix_mask.unsqueeze(-1)

            # Step C: 採樣 (使用 guided logits)
            target_logits = guided_logits[:, target_start:, :] / max(self.temperature, 1e-5)
            target_logits = target_logits.masked_fill(~self.level_token_mask, mask_value)
            
            if self.prefix_trie is not None:
                dead_beam_mask |= self._mask_logits_with_trie(
                    target_logits, allowed_next, prefix_lens, mask_value
                )

            topk_vals, topk_ids = torch.topk(target_logits, k=self.top_k, dim=-1)
            probs = torch.softmax(topk_vals, dim=-1)
            sampled = torch.multinomial(
                probs.view(-1, self.top_k), num_samples=1
            ).view(-1, target_len, 1)
            sampled_tokens = topk_ids.gather(-1, sampled).squeeze(-1)

            # Step D: 自適應重掩碼
            conf_logits = guided_logits[:, target_start:, :].masked_fill(~self.level_token_mask, mask_value)
            if self.prefix_trie is not None:
                dead_beam_mask |= self._mask_logits_with_trie(
                    conf_logits, allowed_next, prefix_lens, mask_value
                )
            target_conf = torch.softmax(conf_logits, dim=-1).max(dim=-1).values
            
            threshold = min(0.05, float(t + 1) / float(self.num_inference_steps + 1))
            confident = target_conf > threshold
            
            if self.prefix_trie is not None and dead_beam_mask.any():
                confident = confident & (~dead_beam_mask.unsqueeze(1))

            # 兜底：如果全都不置信，放寬首個位置
            confident_view = confident.view(batch_size, self.beam_size, target_len)
            prefix_lens_view = (prefix_lens.view(batch_size, self.beam_size) if prefix_lens is not None else None)
            
            for b in range(batch_size):
                for k in range(self.beam_size):
                    pos = int(prefix_lens_view[b, k].item()) if prefix_lens_view is not None else 0
                    pos = min(pos, target_len - 1)
                    confident_view[b, k, pos] = True
            
            no_conf_mask = ~confident_view.any(dim=2)
            if no_conf_mask.any():
                confident_view[no_conf_mask, 0] = True
            confident = confident_view.view(batch_size * self.beam_size, target_len)

            if self.prefix_trie is not None:
                new_target = self._remask_with_trie(
                    sampled_tokens=sampled_tokens,
                    confident=confident,
                    current_target=current_target,
                    prefix_lens=prefix_lens,
                    inactive_mask=dead_beam_mask,
                )
            else:
                new_target = torch.where(
                    confident, sampled_tokens, torch.full_like(sampled_tokens, self.mask_token_id)
                )

            expanded_ids[:, target_start:] = new_target

        # 5) 最終打分 (Refinement + Reranking)
        final_outputs = self.backbone(
            input_ids=expanded_ids,
            attention_mask=expanded_mask,
            return_dict=True,
        )
        final_target_logits = final_outputs.logits[:, target_start:, :]
        
        prefix_target = prefix_mask[:, target_start:target_start + target_len].repeat(batch_size * self.beam_size, 1)
        guided_final_logits = final_target_logits - self.lambda_div * expanded_penalty.unsqueeze(1) * prefix_target.unsqueeze(-1)
        guided_final_logits = guided_final_logits.masked_fill(~self.level_token_mask, mask_value)

        # 最終解碼 (帶 Trie 約束)
        decode_dead_mask = dead_beam_mask.clone()
        if self.prefix_trie is not None:
            final_prefix_lens, _allowed_next, prefix_dead = self._build_prefix_state(expanded_ids[:, target_start:])
            decode_dead_mask |= prefix_dead
            decoded_flat, guided_final_logits, decode_dead_mask = self._decode_with_trie(
                logits=guided_final_logits,
                base_tokens=expanded_ids[:, target_start:],
                prefix_lens=final_prefix_lens,
                inactive_mask=decode_dead_mask,
                mask_value=mask_value,
            )
            final_preds = decoded_flat.view(batch_size, self.beam_size, target_len)
            guided_final_logits = guided_final_logits.view(batch_size, self.beam_size, target_len, self.vocab_size)
            dead_beam_mask = decode_dead_mask
        else:
            guided_final_logits = guided_final_logits.view(batch_size, self.beam_size, target_len, self.vocab_size)
            final_preds = guided_final_logits.argmax(dim=-1)

        # 計算基礎分數
        log_probs = torch.log_softmax(guided_final_logits, dim=-1)
        token_log_probs = torch.gather(log_probs, dim=-1, index=final_preds.unsqueeze(-1)).squeeze(-1)
        beam_scores = token_log_probs.sum(dim=-1)

        # 初始化 invalid_mask
        invalid_mask = torch.zeros_like(beam_scores, dtype=torch.bool)
        if dead_beam_mask.any():
            invalid_mask |= dead_beam_mask.view(batch_size, self.beam_size)

        # 6) Rerank: 內部重複懲罰 & 類別覆蓋懲罰
        if self.lambda_div > 0:
            # 序列內部重複懲罰
            rep_penalty = torch.zeros_like(beam_scores)
            for b in range(batch_size):
                for k in range(self.beam_size):
                    seq_tokens = final_preds[b, k].tolist()
                    uniq = len(set(seq_tokens))
                    rep_ratio = 1.0 - (uniq / float(len(seq_tokens) + 1e-6))
                    rep_penalty[b, k] = rep_ratio if rep_ratio > 0.5 else 0.0
            beam_scores = beam_scores - (self.lambda_div * 0.25) * rep_penalty

            # 類別覆蓋懲罰 (Beam 間)
            if self.code_to_item_map:
                beam_cates = self._codes_to_categories(final_preds)
                cate_penalty = torch.zeros_like(beam_scores)
                penalty_scale = 1.0 / float(max(1, self.beam_size))
                for b in range(batch_size):
                    cats = beam_cates[b]
                    valid_mask_b = ~invalid_mask[b]
                    for cate in cats.unique():
                        if cate < 0: continue
                        mask = (cats == cate) & valid_mask_b
                        repeats = int(mask.sum().item()) - 1
                        if repeats > 0:
                            cate_penalty[b, mask] = 1.0
                beam_scores = beam_scores - (self.lambda_div * penalty_scale) * cate_penalty

        # 7) 標記無效 Beam
        if self.code_to_item_map:
            for b in range(batch_size):
                for k in range(self.beam_size):
                    code_tuple = tuple(final_preds[b, k].tolist())
                    invalid = self.code_to_item_map.get(code_tuple, -1) == -1
                    if not invalid and self.prefix_trie is not None:
                        invalid = not self._is_valid_code_by_trie(list(code_tuple))
                    invalid_mask[b, k] |= invalid
        elif self.prefix_trie is not None:
            for b in range(batch_size):
                for k in range(self.beam_size):
                    invalid_mask[b, k] |= not self._is_valid_code_by_trie(final_preds[b, k].tolist())
        
        if invalid_mask.any():
            beam_scores = torch.where(invalid_mask, torch.full_like(beam_scores, mask_value), beam_scores)

        # 8) 最終排序 (類別優先策略)
        sort_indices = None
        if self.lambda_div > 0 and self.code_to_item_map:
            beam_cates = self._codes_to_categories(final_preds)
            new_orders = []
            for b in range(batch_size):
                scores_b = beam_scores[b]
                cates_b = beam_cates[b]
                valid_mask_b = ~invalid_mask[b]

                # 策略: 每個類別先取最高分的一個
                best_per_cate = []
                for cate in cates_b.unique():
                    if cate < 0: continue
                    cate_mask = (cates_b == cate) & valid_mask_b
                    idxs = torch.nonzero(cate_mask, as_tuple=False).squeeze(-1)
                    if idxs.numel() == 0: continue
                    best_idx = idxs[scores_b[idxs].argmax()]
                    best_per_cate.append(best_idx)

                if best_per_cate:
                    best_per_cate = torch.stack(best_per_cate)
                    best_scores = scores_b[best_per_cate]
                    _, order = best_scores.sort(descending=True)
                    primary = best_per_cate[order]
                else:
                    primary = torch.tensor([], device=device, dtype=torch.long)

                # 剩餘的按分數補齊
                mask_keep = torch.zeros_like(scores_b, dtype=torch.bool)
                mask_keep[primary] = True
                mask_keep = mask_keep | invalid_mask[b]
                remaining = torch.nonzero(~mask_keep, as_tuple=False).squeeze(-1)
                
                final_order = primary
                if remaining.numel() > 0:
                    _, rem_order = scores_b[remaining].sort(descending=True)
                    secondary = remaining[rem_order]
                    final_order = torch.cat([primary, secondary], dim=0)
                
                if final_order.numel() < self.beam_size:
                    pad = torch.arange(self.beam_size - final_order.numel(), device=device, dtype=torch.long)
                    final_order = torch.cat([final_order, pad], dim=0)[: self.beam_size]
                new_orders.append(final_order)
            sort_indices = torch.stack(new_orders, dim=0)
            
            # 強制前 diversity_topk 不重複類別
            if self.diversity_topk > 0:
                enforce_orders = []
                for b in range(batch_size):
                    order_b = sort_indices[b]
                    cates_b = self._codes_to_categories(final_preds[b:b+1, :, :])[0]
                    seen = set()
                    keep = []
                    drop = []
                    for idx in order_b.tolist():
                        cate = int(cates_b[idx].item())
                        if cate < 0: 
                            drop.append(idx)
                            continue
                        if len(keep) < self.diversity_topk and cate in seen:
                            drop.append(idx)
                        else:
                            keep.append(idx)
                            seen.add(cate)
                    final_order = torch.tensor(keep + drop, device=device, dtype=torch.long)[: self.beam_size]
                    if final_order.numel() < self.beam_size:
                        pad = torch.arange(self.beam_size - final_order.numel(), device=device, dtype=torch.long)
                        final_order = torch.cat([final_order, pad], dim=0)[: self.beam_size]
                    enforce_orders.append(final_order)
                sort_indices = torch.stack(enforce_orders, dim=0)
        else:
            sort_indices = beam_scores.sort(dim=1, descending=True).indices

        final_preds = final_preds.gather(1, sort_indices.unsqueeze(-1).expand(-1, -1, target_len))

        # 9) 計算指標
        pos_index = self._calculate_pos_index(preds=final_preds, labels=target_codes)
        metrics = {"count": batch_size}
        for k in topk_list:
            metrics[f"Recall@{k}"] = recall_at_k(pos_index, k).sum().item()
            metrics[f"NDCG@{k}"] = ndcg_at_k(pos_index, k).sum().item()

        # 計算多樣性指標
        if self.code_to_item_map and self.item_category_tensor.numel() > 0:
            top_beams = max(1, min(self.diversity_topk, self.beam_size))
            base_map = self.item_category_tensor.to(device)

            beam_items = torch.full((batch_size, self.beam_size), -1, device=device, dtype=torch.long)
            for b in range(batch_size):
                for idx in range(self.beam_size):
                    code_tuple = tuple(final_preds[b, idx].tolist())
                    beam_items[b, idx] = self.code_to_item_map.get(code_tuple, -1)

            cand_items = beam_items[:, :top_beams]
            # 確保索引安全
            safe_items = torch.clamp(cand_items, min=0, max=max(int(base_map.numel() - 1), 0))
            cand_cates = base_map[safe_items]
            valid_mask = (cand_items > 0) & (cand_items < base_map.numel()) & (cand_cates >= 0)

            gt_items = torch.full((batch_size, 1), -1, device=device, dtype=torch.long)
            for b in range(batch_size):
                code_tuple = tuple(target_codes[b].tolist())
                gt_item = self.code_to_item_map.get(code_tuple, -1)
                gt_items[b, 0] = gt_item if gt_item > 0 else -1

            div_score, div_cnt = calculate_diversity_at_n(cand_items, base_map, valid_mask=valid_mask)
            alpha_score, alpha_cnt = calculate_alpha_ndcg_at_k(
                cand_items, gt_items, base_map, k=top_beams, alpha=self.alpha_diversity, valid_mask=valid_mask
            )
            
            metrics[f"Diversity@{top_beams}"] = div_score
            metrics[f"_valid_Diversity@{top_beams}"] = div_cnt
            metrics[f"AlphaNDCG@{top_beams}"] = alpha_score
            metrics[f"_valid_AlphaNDCG@{top_beams}"] = alpha_cnt

            if self.num_semantic_categories > 0:
                scd_sum = 0.0
                scd_cnt = 0
                wscd_sum = 0.0
                for b in range(batch_size):
                    rec_items = [
                        int(cand_items[b, idx].item()) 
                        for idx in range(top_beams) 
                        if cand_items[b, idx] > 0 and int(cand_items[b, idx]) in self.item_to_cate_map
                    ]
                    if not rec_items: continue
                    try:
                        scd_sum += compute_scd(rec_items, self.item_to_cate_map, self.num_semantic_categories)
                        wscd_sum += compute_weighted_scd(rec_items, self.item_to_cate_map)
                        scd_cnt += 1
                    except Exception:
                        continue
                if scd_cnt > 0:
                    metrics[f"SCD@{top_beams}"] = scd_sum
                    metrics[f"WSCD@{top_beams}"] = wscd_sum
                    metrics[f"_valid_SCD@{top_beams}"] = scd_cnt

        return metrics

    def _get_allowed_tokens_from_trie(self, prefix: List[int]) -> List[int]:
        if self.prefix_trie is None: return []
        try:
            return self.prefix_trie._lookup(tuple(prefix))
        except AttributeError:
            try: return self.prefix_trie.get_allowed_next_tokens(0, prefix)
            except Exception: return []

    def _build_prefix_state(self, targets: torch.Tensor) -> Tuple[torch.Tensor, List[List[int]], torch.Tensor]:
        batch_beam, target_len = targets.shape
        prefix_lens = torch.zeros(batch_beam, device=targets.device, dtype=torch.long)
        allowed_next: List[List[int]] = [[] for _ in range(batch_beam)]
        dead_mask = torch.zeros(batch_beam, device=targets.device, dtype=torch.bool)
        if self.prefix_trie is None: return prefix_lens, allowed_next, dead_mask

        for idx in range(batch_beam):
            prefix: List[int] = []
            valid_len = 0
            for pos in range(target_len):
                token = int(targets[idx, pos].item())
                if token == self.mask_token_id or token == self.pad_token_id: break
                allowed = self._get_allowed_tokens_from_trie(prefix)
                if not allowed:
                    dead_mask[idx] = True
                    break
                if token not in allowed: break
                prefix.append(token)
                valid_len += 1
            prefix_lens[idx] = valid_len
            allowed_next[idx] = self._get_allowed_tokens_from_trie(prefix)
            if not allowed_next[idx]: dead_mask[idx] = True
        return prefix_lens, allowed_next, dead_mask

    def _mask_logits_with_trie(self, logits: torch.Tensor, allowed_next: List[List[int]], prefix_lens: Optional[torch.Tensor], mask_value: float) -> torch.Tensor:
        if self.prefix_trie is None or prefix_lens is None:
            return torch.zeros(logits.size(0), device=logits.device, dtype=torch.bool)
        dead_mask = torch.zeros(logits.size(0), device=logits.device, dtype=torch.bool)
        target_len = logits.size(1)
        for i, allowed in enumerate(allowed_next):
            pos = int(prefix_lens[i].item())
            if pos >= target_len: continue
            if not allowed:
                dead_mask[i] = True
                continue
            allowed_tensor = torch.as_tensor(allowed, device=logits.device, dtype=torch.long)
            disallow = torch.ones(self.vocab_size, device=logits.device, dtype=torch.bool)
            disallow[allowed_tensor] = False
            logits[i, pos] = logits[i, pos].masked_fill(disallow, mask_value)
        return dead_mask

    def _remask_with_trie(self, sampled_tokens: torch.Tensor, confident: torch.Tensor, current_target: torch.Tensor, prefix_lens: Optional[torch.Tensor], inactive_mask: torch.Tensor) -> torch.Tensor:
        batch_beam, target_len = sampled_tokens.shape
        new_target = torch.full_like(sampled_tokens, self.mask_token_id)
        for i in range(batch_beam):
            if inactive_mask is not None and inactive_mask[i]: continue
            keep_len = int(prefix_lens[i].item()) if prefix_lens is not None else 0
            prefix_tokens: List[int] = []
            if keep_len > 0:
                prefix_slice = current_target[i, :keep_len]
                new_target[i, :keep_len] = prefix_slice
                prefix_tokens = [int(x) for x in prefix_slice.tolist()]
            for pos in range(keep_len, target_len):
                if not bool(confident[i, pos].item()): break
                allowed = self._get_allowed_tokens_from_trie(prefix_tokens)
                if not allowed: break
                cand = int(sampled_tokens[i, pos].item())
                if cand not in allowed: break
                new_target[i, pos] = cand
                prefix_tokens.append(cand)
        return new_target

    def _decode_with_trie(self, logits: torch.Tensor, base_tokens: torch.Tensor, prefix_lens: Optional[torch.Tensor], inactive_mask: torch.Tensor, mask_value: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_beam, target_len, _ = logits.shape
        decoded = torch.full((batch_beam, target_len), self.mask_token_id, device=logits.device, dtype=torch.long)
        masked_logits = logits.clone()
        for i in range(batch_beam):
            if inactive_mask is not None and inactive_mask[i]: continue
            keep_len = int(prefix_lens[i].item()) if prefix_lens is not None else 0
            prefix_tokens: List[int] = []
            if keep_len > 0:
                prefix_slice = base_tokens[i, :keep_len]
                decoded[i, :keep_len] = prefix_slice
                prefix_tokens = [int(x) for x in prefix_slice.tolist()]
            for pos in range(keep_len, target_len):
                allowed = self._get_allowed_tokens_from_trie(prefix_tokens)
                if not allowed:
                    inactive_mask[i] = True
                    break
                disallow = torch.ones(self.vocab_size, device=logits.device, dtype=torch.bool)
                disallow[torch.as_tensor(allowed, device=logits.device, dtype=torch.long)] = False
                masked_logits[i, pos] = masked_logits[i, pos].masked_fill(disallow, mask_value)
                token = int(masked_logits[i, pos].argmax(dim=-1).item())
                decoded[i, pos] = token
                prefix_tokens.append(token)
        return decoded, masked_logits, inactive_mask

    def _is_valid_code_by_trie(self, code_seq: List[int]) -> bool:
        if self.prefix_trie is None: return True
        prefix: List[int] = []
        for token in code_seq:
            if token == self.mask_token_id or token == self.pad_token_id: return False
            allowed = self._get_allowed_tokens_from_trie(prefix)
            if not allowed or token not in allowed: return False
            prefix.append(token)
        return True

    def _codes_to_categories(self, preds: torch.Tensor) -> torch.Tensor:
        if not self.code_to_item_map or self.item_category_tensor.numel() == 0:
            return torch.full((preds.size(0), preds.size(1)), -1, device=preds.device, dtype=torch.long)
        base_map = self.item_category_tensor.to(preds.device)
        bsz, num_beam, _ = preds.shape
        cates = torch.full((bsz, num_beam), -1, device=preds.device, dtype=torch.long)
        for b in range(bsz):
            for k in range(num_beam):
                code_tuple = tuple(preds[b, k].tolist())
                item_id = self.code_to_item_map.get(code_tuple, -1)
                if 0 <= item_id < base_map.numel():
                    cates[b, k] = base_map[item_id]
                else:
                    cates[b, k] = preds[b, k, 0]
        return cates

    def _calculate_diversity_penalty(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, target_start: int) -> torch.Tensor:
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        special_ids = torch.tensor([self.pad_token_id, self.mask_token_id, self.cls_token_id, self.sep_token_id], device=device)
        
        sep_mask = (input_ids == self.sep_token_id) & attention_mask.bool()
        default_sep = torch.full((batch_size,), target_start, device=device)
        sep_pos = torch.where(sep_mask.any(dim=1), sep_mask.float().argmax(dim=1), default_sep)
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        hist_mask = (positions < sep_pos.unsqueeze(1)) & attention_mask.bool()
        valid_mask = hist_mask & (~torch.isin(input_ids, special_ids))
        
        starts = self.level_starts.to(device)
        ends = self.level_ends.to(device)
        max_prefix_lvl = min(1, len(starts) - 1)
        in_prefix_levels = (input_ids >= starts[0]) & (input_ids <= ends[max_prefix_lvl])
        valid_mask = valid_mask & in_prefix_levels
        
        penalty = torch.zeros(batch_size, self.vocab_size, device=device, dtype=torch.float)
        non_zero = torch.nonzero(valid_mask, as_tuple=False)
        if non_zero.numel() > 0:
            batch_indices = non_zero[:, 0]
            token_indices = input_ids[valid_mask]
            penalty.index_put_((batch_indices, token_indices), torch.ones_like(token_indices, dtype=penalty.dtype), accumulate=True)
        
        total = penalty.sum(dim=1, keepdim=True).clamp(min=1.0)
        return penalty / total

    @staticmethod
    def _calculate_pos_index(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if labels.dim() == 2: labels = labels.unsqueeze(1)
        label_len = labels.shape[-1]
        preds = preds[..., :label_len]
        labels_expanded = labels.expand(-1, preds.shape[1], -1)
        return (preds == labels_expanded).all(dim=-1)

    @staticmethod
    def _build_item_cate_maps(item_to_cate_map: Optional[Dict[int, Any]], item_to_code_map: Optional[Dict[int, List[int]]]) -> Tuple[Dict[int, Any], Dict[int, List[Any]]]:
        cate_map: Dict[int, Any] = {}
        if item_to_cate_map: cate_map.update(item_to_cate_map)
        if item_to_code_map:
            for iid, codes in item_to_code_map.items():
                if iid not in cate_map and len(codes) > 0:
                    cate_map[iid] = codes[0]
        cates_map = {iid: (v if isinstance(v, list) else [v]) for iid, v in cate_map.items()}
        return cate_map, cates_map