from collections import defaultdict
from typing import Dict, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData

EdgeType = Tuple[str, str, str]


class LightGCNLinkPredictor(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        hidden_dim: int,
        num_layers: int,
        eval_top_k,
        target_edge_type: EdgeType = ("user", "interacts", "item"),
    ):
        super().__init__()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.eval_top_k = [int(k) for k in eval_top_k]
        self.target_edge_type = target_edge_type

        self.user_embedding = nn.Embedding(self.num_users, self.hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.user_embedding.weight, std=0.1)

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        z_dict = self.encode(data)
        src_index, pos_dst_index, neg_dst_index = self._get_triplet_indices(data)
        pos_logits, neg_logits = self.decode_triplets(
            z_dict,
            src_index,
            pos_dst_index,
            neg_dst_index,
        )
        return {
            "loss": self.bpr_loss(pos_logits, neg_logits),
            "pos_logits": pos_logits,
            "neg_logits": neg_logits,
            "z_dict": z_dict,
        }

    def encode(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        user_z = self._initial_user_embeddings(data)
        item_z = self._initial_item_embeddings(data)
        user_outputs = [user_z]
        item_outputs = [item_z]

        edge_index = data[self.target_edge_type].edge_index
        num_users = int(user_z.size(0))
        num_items = int(item_z.size(0))

        for _ in range(self.num_layers):
            user_z, item_z = self._propagate_once(
                user_z,
                item_z,
                edge_index,
                num_users,
                num_items,
            )
            user_outputs.append(user_z)
            item_outputs.append(item_z)

        return {
            "user": torch.stack(user_outputs, dim=0).mean(dim=0),
            "item": torch.stack(item_outputs, dim=0).mean(dim=0),
        }

    def _initial_user_embeddings(self, data: HeteroData) -> torch.Tensor:
        user_store = data["user"]
        user_ids = getattr(user_store, "n_id", None)
        if user_ids is None:
            user_ids = torch.arange(
                int(user_store.num_nodes),
                device=self.user_embedding.weight.device,
            )
        else:
            user_ids = user_ids.to(self.user_embedding.weight.device)
        return F.normalize(self.user_embedding(user_ids), p=2, dim=-1)

    def _initial_item_embeddings(self, data: HeteroData) -> torch.Tensor:
        return F.normalize(data["item"].x.float(), p=2, dim=-1)

    @staticmethod
    def _propagate_once(
        user_z: torch.Tensor,
        item_z: torch.Tensor,
        edge_index: torch.Tensor,
        num_users: int,
        num_items: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_index.numel() == 0:
            return torch.zeros_like(user_z), torch.zeros_like(item_z)

        src, dst = edge_index
        user_degree = torch.bincount(src, minlength=num_users).to(user_z.dtype)
        item_degree = torch.bincount(dst, minlength=num_items).to(item_z.dtype)
        norm = (
            user_degree[src].clamp_min(1).rsqrt()
            * item_degree[dst].clamp_min(1).rsqrt()
        )

        next_user_z = torch.zeros_like(user_z)
        next_item_z = torch.zeros_like(item_z)
        next_item_z.index_add_(0, dst, user_z[src] * norm.view(-1, 1))
        next_user_z.index_add_(0, src, item_z[dst] * norm.view(-1, 1))
        return next_user_z, next_item_z

    def decode_triplets(
        self,
        z_dict,
        src_index: torch.Tensor,
        pos_dst_index: torch.Tensor,
        neg_dst_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src_z = z_dict["user"][src_index]
        pos_z = z_dict["item"][pos_dst_index]
        pos_logits = (src_z * pos_z).sum(dim=-1)

        if neg_dst_index.dim() == 1:
            neg_z = z_dict["item"][neg_dst_index]
            neg_logits = (src_z * neg_z).sum(dim=-1)
        else:
            neg_z = z_dict["item"][neg_dst_index]
            neg_logits = (src_z.unsqueeze(1) * neg_z).sum(dim=-1)

        return pos_logits, neg_logits

    @staticmethod
    def bpr_loss(
        pos_logits: torch.Tensor,
        neg_logits: torch.Tensor,
    ) -> torch.Tensor:
        if neg_logits.dim() == 1:
            diff = pos_logits - neg_logits
        else:
            diff = pos_logits.unsqueeze(1) - neg_logits
        return -F.logsigmoid(diff).mean()

    @torch.no_grad()
    def evaluate_full_sort(
        self,
        data: HeteroData,
        split: str,
        batch_size: int,
    ) -> Dict[str, float]:
        self.eval()
        z_dict = self.encode(data)
        user_z = z_dict["user"]
        item_z = z_dict["item"]

        edge_store = data[self.target_edge_type]
        target_edge_index = getattr(edge_store, f"{split}_edge_label_index")
        exclude_items = self._build_full_sort_exclusions(edge_store, split)
        max_k = min(max(self.eval_top_k), int(item_z.size(0)))

        totals: Dict[str, float] = {}
        num_targets = int(target_edge_index.size(1))
        for k in self.eval_top_k:
            totals[f"Recall@{k}"] = 0.0
            totals[f"NDCG@{k}"] = 0.0

        for start in range(0, num_targets, int(batch_size)):
            end = min(start + int(batch_size), num_targets)
            users = target_edge_index[0, start:end]
            target_items = target_edge_index[1, start:end]
            scores = user_z[users] @ item_z.t()

            for row, user in enumerate(users.tolist()):
                excluded = exclude_items.get(int(user), set())
                if excluded:
                    scores[row, list(excluded)] = -torch.inf

            top_items = torch.topk(scores, k=max_k, dim=1).indices
            hits = top_items == target_items.view(-1, 1)
            for k in self.eval_top_k:
                topk_hits = hits[:, :k]
                hit_rows = topk_hits.any(dim=1)
                totals[f"Recall@{k}"] += float(hit_rows.float().sum().item())

                hit_positions = topk_hits.float().argmax(dim=1)
                ndcg = torch.zeros_like(hit_positions, dtype=torch.float32)
                if hit_rows.any():
                    ranks = hit_positions[hit_rows].float() + 2.0
                    ndcg[hit_rows] = 1.0 / torch.log2(ranks)
                totals[f"NDCG@{k}"] += float(ndcg.sum().item())

        return {
            metric: value / max(1, num_targets)
            for metric, value in totals.items()
        }

    @staticmethod
    def _build_full_sort_exclusions(edge_store, split: str) -> Dict[int, Set[int]]:
        excluded = defaultdict(set)
        attrs = ["train_edge_label_index"]
        if split == "test":
            attrs.append("valid_edge_label_index")

        for attr in attrs:
            edge_index = getattr(edge_store, attr)
            for user, item in edge_index.t().tolist():
                excluded[int(user)].add(int(item))
        return excluded

    def _get_triplet_indices(self, data: HeteroData):
        src_type, _, dst_type = self.target_edge_type
        src_store = data[src_type]
        dst_store = data[dst_type]
        return (
            src_store.src_index,
            dst_store.dst_pos_index,
            dst_store.dst_neg_index,
        )


def build_lightgcn_link_predictor(
    data: HeteroData,
    hidden_dim: int,
    num_layers: int,
    eval_top_k,
    target_edge_type: EdgeType = ("user", "interacts", "item"),
    **_,
) -> LightGCNLinkPredictor:
    return LightGCNLinkPredictor(
        num_users=int(data["user"].num_nodes),
        num_items=int(data["item"].num_nodes),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        eval_top_k=eval_top_k,
        target_edge_type=target_edge_type,
    )
