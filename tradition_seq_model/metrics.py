import torch


def recall_at_k(pos_index: torch.Tensor, k: int) -> torch.Tensor:
    return pos_index[:, :k].sum(dim=1).cpu().float()


def ndcg_at_k(pos_index: torch.Tensor, k: int) -> torch.Tensor:
    ranks = torch.arange(1, pos_index.shape[-1] + 1, device=pos_index.device, dtype=torch.float)
    discounts = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(pos_index, discounts, torch.tensor(0.0, device=pos_index.device))
    return dcg[:, :k].sum(dim=1).cpu().float()
