from collections import defaultdict
from typing import Dict, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv, Linear
from tqdm import tqdm

EdgeType = Tuple[str, str, str]
TQDM_KWARGS = {"dynamic_ncols": True, "leave": False}


class HANEncoder(nn.Module):
    def __init__(
        self,
        metadata,
        hidden_dim: int,
        num_layers: int,
        heads: int,
        negative_slope: float,
        dropout: float,
    ):
        super().__init__()
        self.dropout = float(dropout)
        self.convs = nn.ModuleList(
            [
                HANConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=metadata,
                    heads=heads,
                    negative_slope=negative_slope,
                    dropout=dropout,
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            next_x_dict = conv(x_dict, edge_index_dict)
            x_dict = {
                node_type: F.dropout(
                    F.relu(next_x if next_x is not None else x_dict[node_type]),
                    p=self.dropout,
                    training=self.training,
                )
                for node_type, next_x in next_x_dict.items()
            }
        return x_dict


class HANLinkPredictor(nn.Module):
    def __init__(
        self,
        metadata,
        num_nodes_dict: Dict[str, int],
        input_dims: Dict[str, int],
        hidden_dim: int,
        num_layers: int,
        heads: int,
        negative_slope: float,
        dropout: float,
        eval_top_k,
        target_edge_type: EdgeType = ("user", "interacts", "item"),
    ):
        super().__init__()
        self.metadata = metadata
        self.num_nodes_dict = dict(num_nodes_dict)
        self.input_dims = dict(input_dims)
        self.hidden_dim = int(hidden_dim)
        self.eval_top_k = [int(k) for k in eval_top_k]
        self.target_edge_type = target_edge_type

        node_types, _ = metadata
        self.input_linears = nn.ModuleDict()
        self.node_embeddings = nn.ModuleDict()

        for node_type in node_types:
            if node_type in self.input_dims:
                self.input_linears[node_type] = Linear(
                    self.input_dims[node_type],
                    self.hidden_dim,
                )
            else:
                self.node_embeddings[node_type] = nn.Embedding(
                    self.num_nodes_dict[node_type],
                    self.hidden_dim,
                )

        self.encoder = HANEncoder(
            metadata=metadata,
            hidden_dim=self.hidden_dim,
            num_layers=num_layers,
            heads=heads,
            negative_slope=negative_slope,
            dropout=dropout,
        )

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

    def encode(self, data: HeteroData):
        x_dict = self._build_x_dict(data)
        return self.encoder(x_dict, data.edge_index_dict)

    def decode_triplets(
        self,
        z_dict,
        src_index: torch.Tensor,
        pos_dst_index: torch.Tensor,
        neg_dst_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src_type, _, dst_type = self.target_edge_type
        src_z = z_dict[src_type][src_index]
        pos_z = z_dict[dst_type][pos_dst_index]

        if neg_dst_index.dim() == 1:
            neg_z = z_dict[dst_type][neg_dst_index]
            neg_logits = (src_z * neg_z).sum(dim=-1)
        else:
            neg_z = z_dict[dst_type][neg_dst_index]
            neg_logits = (src_z.unsqueeze(1) * neg_z).sum(dim=-1)

        pos_logits = (src_z * pos_z).sum(dim=-1)
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
        if split not in {"valid", "test"}:
            raise ValueError(f"Full-sort evaluation only supports valid/test: {split}")

        self.eval()
        z_dict = self.encode(data)
        src_type, _, dst_type = self.target_edge_type
        user_z = z_dict[src_type]
        item_z = z_dict[dst_type]

        edge_store = data[self.target_edge_type]
        target_edge_index = getattr(edge_store, f"{split}_edge_label_index")
        exclude_items = self._build_full_sort_exclusions(edge_store, split)
        max_k = min(max(self.eval_top_k), int(item_z.size(0)))

        totals: Dict[str, float] = {}
        num_targets = int(target_edge_index.size(1))
        for k in self.eval_top_k:
            totals[f"Recall@{k}"] = 0.0
            totals[f"NDCG@{k}"] = 0.0

        for start in tqdm(
            range(0, num_targets, int(batch_size)),
            desc="Evaluating",
            **TQDM_KWARGS,
        ):
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

    def _build_x_dict(self, data: HeteroData):
        x_dict = {}
        node_types, _ = self.metadata

        for node_type in node_types:
            store = data[node_type]
            if node_type in self.input_linears:
                if not hasattr(store, "x") or store.x is None:
                    raise KeyError(f"Missing node features for node type: {node_type}")
                x_dict[node_type] = F.normalize(
                    self.input_linears[node_type](store.x.float()),
                    p=2,
                    dim=-1,
                )
                continue

            node_ids = getattr(store, "n_id", None)
            if node_ids is None:
                num_nodes = store.num_nodes
                if num_nodes is None:
                    num_nodes = self.num_nodes_dict[node_type]
                node_ids = torch.arange(
                    num_nodes,
                    device=self.node_embeddings[node_type].weight.device,
                )
            else:
                node_ids = node_ids.to(self.node_embeddings[node_type].weight.device)

            x_dict[node_type] = F.normalize(
                self.node_embeddings[node_type](node_ids),
                p=2,
                dim=-1,
            )

        return x_dict

    def _get_triplet_indices(self, data: HeteroData):
        src_type, _, dst_type = self.target_edge_type
        src_store = data[src_type]
        dst_store = data[dst_type]
        return (
            src_store.src_index,
            dst_store.dst_pos_index,
            dst_store.dst_neg_index,
        )


def build_han_link_predictor(
    data: HeteroData,
    hidden_dim: int,
    num_layers: int,
    heads: int,
    negative_slope: float,
    dropout: float,
    eval_top_k,
    target_edge_type: EdgeType = ("user", "interacts", "item"),
) -> HANLinkPredictor:
    num_nodes_dict = {
        node_type: int(data[node_type].num_nodes)
        for node_type in data.node_types
    }
    input_dims = {
        node_type: int(data[node_type].x.size(-1))
        for node_type in data.node_types
        if hasattr(data[node_type], "x") and data[node_type].x is not None
    }
    return HANLinkPredictor(
        metadata=data.metadata(),
        num_nodes_dict=num_nodes_dict,
        input_dims=input_dims,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
        negative_slope=negative_slope,
        dropout=dropout,
        eval_top_k=eval_top_k,
        target_edge_type=target_edge_type,
    )
