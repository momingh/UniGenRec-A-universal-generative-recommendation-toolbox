import logging

import numpy as np
import torch


def as_float32_numpy(data):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    return np.ascontiguousarray(data.astype(np.float32, copy=False))


def codebook_sizes_to_nbits(codebook_sizes):
    nbits = [int(np.log2(size)) for size in codebook_sizes]
    if any(2 ** bit != size for bit, size in zip(nbits, codebook_sizes)):
        raise ValueError("All FAISS RQ codebook sizes must be powers of 2.")
    return nbits


def train_faiss_rq(data, codebook_sizes, verbose=True):
    import faiss

    codebook_sizes = [int(size) for size in codebook_sizes]
    nbits = codebook_sizes_to_nbits(codebook_sizes)
    num_levels = len(codebook_sizes)
    data = as_float32_numpy(data)

    n_items, dim = data.shape
    if verbose:
        print("Training FAISS ResidualQuantizer")
        print(
            f"  data={n_items}  dim={dim}  levels={num_levels}  "
            f"codebooks={codebook_sizes}  total_codes={np.prod(codebook_sizes):,}"
        )

    if len(set(nbits)) == 1:
        rq = faiss.ResidualQuantizer(dim, num_levels, nbits[0])
    else:
        rq = faiss.ResidualQuantizer(dim, nbits)
    rq.max_beam_size = 1

    rq.train(data)
    if verbose:
        print("  training completed\n")
    return rq


def get_faiss_rq_codebooks(data, codebook_sizes, verbose=True):
    from faiss.contrib.inspect_tools import get_additive_quantizer_codebooks

    faiss_rq = train_faiss_rq(
        data,
        codebook_sizes=codebook_sizes,
        verbose=verbose,
    )
    return get_additive_quantizer_codebooks(faiss_rq)


def copy_codebooks_to_embeddings(codebooks, embeddings, label):
    if len(codebooks) != len(embeddings):
        raise ValueError(
            f"{label} codebook count mismatch: got {len(codebooks)}, "
            f"expected {len(embeddings)}."
        )

    for layer_idx, (codebook, embedding) in enumerate(zip(codebooks, embeddings)):
        expected_shape = tuple(embedding.weight.shape)
        codebook = np.asarray(codebook, dtype=np.float32)
        if codebook.shape != expected_shape:
            raise ValueError(
                f"FAISS codebook shape mismatch at {label} layer {layer_idx + 1}: "
                f"got {codebook.shape}, expected {expected_shape}."
            )
        embedding.weight.data.copy_(
            torch.from_numpy(codebook).to(embedding.weight.device)
        )
        logging.info(
            "FAISS 初始化完成: %s 第 %d/%d 层 codebook",
            label,
            layer_idx + 1,
            len(embeddings),
        )


def assign_to_codebook(x, codebook):
    distances = (
        x.pow(2).sum(dim=1, keepdim=True)
        + codebook.pow(2).sum(dim=1).unsqueeze(0)
        - 2 * x @ codebook.t()
    )
    return torch.argmin(distances, dim=1)


def assign_batch_multiple(x, candidates):
    distances = (candidates - x.unsqueeze(1)).pow(2).sum(dim=2)
    codes = torch.argmin(distances, dim=1)
    gather_index = codes.view(-1, 1, 1).expand(-1, 1, candidates.shape[-1])
    xhat_next = torch.gather(candidates, dim=1, index=gather_index).squeeze(1)
    return codes, xhat_next
