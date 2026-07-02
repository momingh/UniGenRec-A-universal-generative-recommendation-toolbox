from collections import defaultdict
from typing import Dict, List, Mapping, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from tqdm import tqdm

EdgeType = Tuple[str, str, str]
TQDM_KWARGS = {"dynamic_ncols": True, "leave": False}


class MetaPath2VecLinkPredictor(nn.Module):
    """MetaPath2Vec-style skip-gram model with full-sort recommendation eval."""

    def __init__(
        self,
        edge_index_dict: Mapping[EdgeType, torch.Tensor],
        num_nodes_dict: Mapping[str, int],
        hidden_dim: int,
        metapath: Sequence[EdgeType],
        walk_length: int,
        context_size: int,
        walks_per_node: int,
        num_negative_samples: int,
        bpr_weight: float,
        eval_top_k,
        target_edge_type: EdgeType = ("user", "interacts", "item"),
    ):
        super().__init__()
        self.num_nodes_dict = {
            str(node_type): int(num_nodes)
            for node_type, num_nodes in num_nodes_dict.items()
        }
        self.hidden_dim = int(hidden_dim)
        self.metapath = [tuple(edge_type) for edge_type in metapath]
        self.walk_length = int(walk_length)
        self.context_size = int(context_size)
        self.walks_per_node = int(walks_per_node)
        self.num_negative_samples = int(num_negative_samples)
        self.bpr_weight = float(bpr_weight)
        self.eval_top_k = [int(k) for k in eval_top_k]
        self.target_edge_type = target_edge_type

        if not self.metapath:
            raise ValueError("metapath must contain at least one edge type.")
        if self.walk_length < 1:
            raise ValueError("walk_length must be positive.")
        if self.context_size < 2:
            raise ValueError("context_size must be at least 2.")
        if self.walk_length + 1 < self.context_size:
            raise ValueError("walk_length + 1 must be >= context_size.")
        if self.walks_per_node < 1:
            raise ValueError("walks_per_node must be positive.")
        if self.num_negative_samples < 1:
            raise ValueError("num_negative_samples must be positive.")
        if self.bpr_weight < 0.0:
            raise ValueError("bpr_weight must be non-negative.")

        self.node_embeddings = nn.ModuleDict()
        self._init_node_representations()
        self._edge_buffer_index: Dict[EdgeType, int] = {}
        for edge_idx, edge_type in enumerate(self.metapath):
            if edge_type not in edge_index_dict:
                raise KeyError(f"Metapath edge type is missing from graph: {edge_type}")
            self._register_edge_buffers(
                edge_idx=edge_idx,
                edge_type=edge_type,
                edge_index=edge_index_dict[edge_type],
            )

        self._walk_edge_types = [
            self.metapath[step % len(self.metapath)]
            for step in range(self.walk_length)
        ]
        self._walk_node_types = self._build_walk_node_types()
        self.reset_parameters()

    def reset_parameters(self):
        for embedding in self.node_embeddings.values():
            nn.init.normal_(embedding.weight, std=0.1)

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        start_nodes = self._batch_start_nodes(data)
        skipgram_loss = self.skipgram_loss(start_nodes)
        if self.bpr_weight > 0.0:
            bpr_loss = self.bpr_loss_from_batch(data)
        else:
            bpr_loss = self._zero_loss()
        loss = skipgram_loss + self.bpr_weight * bpr_loss
        return {
            "loss": loss,
            "skipgram_loss": skipgram_loss,
            "bpr_loss": bpr_loss,
        }

    def encode(self, data: HeteroData | None = None) -> Dict[str, torch.Tensor]:
        z_dict = {}
        for node_type in self.num_nodes_dict:
            z_dict[node_type] = self._node_matrix(node_type, data=data)
        return z_dict

    def skipgram_loss(self, start_nodes: torch.Tensor) -> torch.Tensor:
        if start_nodes.numel() == 0:
            return self._zero_loss()

        walks = self._pos_sample(start_nodes)
        num_windows = self.walk_length + 1 - self.context_size
        if num_windows < 0:
            return self._zero_loss()

        losses = []
        for offset in range(num_windows + 1):
            window = walks[:, offset : offset + self.context_size]
            window_types = self._walk_node_types[offset : offset + self.context_size]
            start_type = window_types[0]
            start_ids = window[:, 0]
            start_z = self._lookup(start_type, start_ids)

            for context_pos, context_type in enumerate(window_types[1:], start=1):
                context_ids = window[:, context_pos]
                context_z = self._lookup(context_type, context_ids)
                pos_logits = (start_z * context_z).sum(dim=-1)
                losses.append(-F.logsigmoid(pos_logits).mean())

                neg_ids = torch.randint(
                    0,
                    self.num_nodes_dict[context_type],
                    (window.size(0), self.num_negative_samples),
                    device=window.device,
                )
                neg_z = self._lookup(context_type, neg_ids)
                neg_logits = (start_z.unsqueeze(1) * neg_z).sum(dim=-1)
                losses.append(-F.logsigmoid(-neg_logits).mean())

        if not losses:
            return self._zero_loss()
        return torch.stack(losses).mean()

    def bpr_loss_from_batch(self, data: HeteroData) -> torch.Tensor:
        src_ids, pos_dst_ids, neg_dst_ids = self._get_triplet_global_indices(data)
        src_type, _, dst_type = self.target_edge_type
        src_z = self._lookup(src_type, src_ids)
        pos_z = self._lookup(dst_type, pos_dst_ids)
        pos_logits = (src_z * pos_z).sum(dim=-1)

        neg_z = self._lookup(dst_type, neg_dst_ids)
        if neg_dst_ids.dim() == 1:
            neg_logits = (src_z * neg_z).sum(dim=-1)
        else:
            neg_logits = (src_z.unsqueeze(1) * neg_z).sum(dim=-1)
        return self.bpr_loss(pos_logits, neg_logits)

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

    def _pos_sample(self, start_nodes: torch.Tensor) -> torch.Tensor:
        device = self._node_device(self._walk_node_types[0])
        batch = start_nodes.to(device).repeat(self.walks_per_node)
        walks = [batch]

        for edge_type in self._walk_edge_types:
            batch = self._sample_neighbors(edge_type, batch)
            walks.append(batch)

        return torch.stack(walks, dim=-1)

    def _sample_neighbors(
        self,
        edge_type: EdgeType,
        subset: torch.Tensor,
    ) -> torch.Tensor:
        edge_idx = self._edge_buffer_index[edge_type]
        rowptr = getattr(self, f"rowptr_{edge_idx}")
        col = getattr(self, f"col_{edge_idx}")
        rowcount = getattr(self, f"rowcount_{edge_idx}")
        dst_type = edge_type[-1]
        num_dst = self.num_nodes_dict[dst_type]

        fallback = torch.randint(0, num_dst, subset.shape, device=subset.device)
        if col.numel() == 0:
            return fallback

        num_src = rowcount.numel()
        safe_subset = subset.clamp(min=0, max=max(num_src - 1, 0))
        count = rowcount[safe_subset]
        valid = (subset >= 0) & (subset < num_src) & (count > 0)
        if not valid.any():
            return fallback

        rand = torch.rand(int(valid.sum().item()), device=subset.device)
        offsets = (rand * count[valid].to(rand.dtype)).to(torch.long)
        edge_positions = rowptr[safe_subset[valid]] + offsets
        out = fallback
        out[valid] = col[edge_positions]
        return out

    def _lookup(self, node_type: str, ids: torch.Tensor) -> torch.Tensor:
        return self.node_embeddings[node_type](ids)

    def _node_matrix(
        self,
        node_type: str,
        data: HeteroData | None = None,
    ) -> torch.Tensor:
        return self.node_embeddings[node_type].weight

    def _node_device(self, node_type: str) -> torch.device:
        return self._node_matrix(node_type).device

    def _init_node_representations(self) -> None:
        for node_type, num_nodes in self.num_nodes_dict.items():
            self.node_embeddings[node_type] = nn.Embedding(
                num_nodes,
                self.hidden_dim,
            )

    def _batch_start_nodes(self, data: HeteroData) -> torch.Tensor:
        src_type, _, _ = self.target_edge_type
        src_store = data[src_type]
        src_index = getattr(src_store, "src_index", None)

        if src_index is None:
            edge_store = data[self.target_edge_type]
            edge_label_index = getattr(edge_store, "edge_label_index", None)
            if edge_label_index is None:
                edge_label_index = getattr(edge_store, "train_edge_label_index")
            src_index = edge_label_index[0]

        node_ids = getattr(src_store, "n_id", None)
        if node_ids is not None:
            return node_ids[src_index].to(torch.long)
        return src_index.to(torch.long)

    def _get_triplet_global_indices(self, data: HeteroData):
        src_type, _, dst_type = self.target_edge_type
        src_store = data[src_type]
        dst_store = data[dst_type]
        return (
            self._local_to_global(src_store, src_store.src_index),
            self._local_to_global(dst_store, dst_store.dst_pos_index),
            self._local_to_global(dst_store, dst_store.dst_neg_index),
        )

    @staticmethod
    def _local_to_global(store, index: torch.Tensor) -> torch.Tensor:
        node_ids = getattr(store, "n_id", None)
        if node_ids is None:
            return index.to(torch.long)
        return node_ids.to(index.device)[index].to(torch.long)

    def _register_edge_buffers(
        self,
        edge_idx: int,
        edge_type: EdgeType,
        edge_index: torch.Tensor,
    ) -> None:
        src_type, _, dst_type = edge_type
        num_src = self.num_nodes_dict[src_type]
        num_dst = self.num_nodes_dict[dst_type]
        rowptr, col = self._build_csr(edge_index, num_src=num_src, num_dst=num_dst)
        rowcount = rowptr[1:] - rowptr[:-1]

        self._edge_buffer_index[edge_type] = edge_idx
        self.register_buffer(f"rowptr_{edge_idx}", rowptr)
        self.register_buffer(f"col_{edge_idx}", col)
        self.register_buffer(f"rowcount_{edge_idx}", rowcount)

    @staticmethod
    def _build_csr(
        edge_index: torch.Tensor,
        num_src: int,
        num_dst: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_index = edge_index.detach().to(torch.long).cpu()
        if edge_index.numel() == 0:
            return torch.zeros(num_src + 1, dtype=torch.long), torch.empty(0, dtype=torch.long)

        row, col = edge_index
        valid = (row >= 0) & (row < num_src) & (col >= 0) & (col < num_dst)
        row = row[valid]
        col = col[valid]
        if row.numel() == 0:
            return torch.zeros(num_src + 1, dtype=torch.long), torch.empty(0, dtype=torch.long)

        perm = (row * int(num_dst) + col).argsort()
        row = row[perm]
        col = col[perm]

        counts = torch.bincount(row, minlength=num_src)
        rowptr = torch.zeros(num_src + 1, dtype=torch.long)
        rowptr[1:] = counts.cumsum(dim=0)
        return rowptr, col

    def _build_walk_node_types(self) -> List[str]:
        node_types = [self.metapath[0][0]]
        for edge_type in self._walk_edge_types:
            if node_types[-1] != edge_type[0]:
                raise ValueError(
                    "Invalid metapath continuity: "
                    f"expected edge from {node_types[-1]}, got {edge_type}."
                )
            node_types.append(edge_type[-1])
        return node_types

    def _zero_loss(self) -> torch.Tensor:
        zero = None
        for param in self.parameters():
            value = param.sum() * 0.0
            zero = value if zero is None else zero + value
        if zero is not None:
            return zero
        for buffer in self.buffers():
            return buffer.sum() * 0.0
        return torch.zeros(())


def build_metapath2vec_link_predictor(
    data: HeteroData,
    hidden_dim: int,
    walk_length: int,
    context_size: int,
    walks_per_node: int,
    num_negative_samples: int,
    eval_top_k,
    bpr_weight: float = 1.0,
    metapath=None,
    target_edge_type: EdgeType = ("user", "interacts", "item"),
    **_,
) -> MetaPath2VecLinkPredictor:
    if metapath is None:
        metapath = [
            target_edge_type,
            (target_edge_type[-1], "rev_interacts", target_edge_type[0]),
        ]
    metapath = [tuple(edge_type) for edge_type in metapath]
    num_nodes_dict = {
        node_type: int(data[node_type].num_nodes)
        for node_type in data.node_types
    }
    edge_index_dict = {
        edge_type: data[edge_type].edge_index
        for edge_type in data.edge_types
    }
    return MetaPath2VecLinkPredictor(
        edge_index_dict=edge_index_dict,
        num_nodes_dict=num_nodes_dict,
        hidden_dim=hidden_dim,
        metapath=metapath,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=num_negative_samples,
        bpr_weight=bpr_weight,
        eval_top_k=eval_top_k,
        target_edge_type=target_edge_type,
    )
