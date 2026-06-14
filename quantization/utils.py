import os
import logging
import importlib
import random
from datetime import datetime
from collections import defaultdict
import numpy as np
import torch
import yaml

def get_model(model_name: str):
    """
    模型工厂函数。
    """
    if model_name.lower().startswith('mm_'):
        raise ValueError("当前量化入口仅支持单文本模态模型。")

    try:
        module_path = f'models.{model_name}'
        model_module = importlib.import_module(module_path)
        class_name = model_name.upper()
        model_class = getattr(model_module, class_name)
    except (ImportError, AttributeError) as e:
        print(f"ERROR: 尝试加载模型 '{model_name}' 时失败。 异常: {e}")
        class_name_upper = model_name.upper()
        raise ValueError(
            f'Model "{model_name}" not found. '
            f'请检查:\n'
            f'1. "models/" 中是否存在 "{model_name}.py"。\n'
            f'2. 该文件中是否定义了类 "{class_name_upper}"。'
        )
        
    return model_class


def load_yaml_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Set random seeds before model initialization and training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
def setup_paths(args):
    """根据输入参数构建文本模态量化路径。"""
    emb_dir = os.path.join(args.data_base_path, args.dataset_name, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)

    embedding_modality = getattr(args, 'embedding_modality', 'text') or 'text'
    if embedding_modality != 'text':
        raise ValueError("当前仅支持文本模态 embedding_modality=text。")
    if not args.embedding_model:
        raise ValueError("必须提供 '--embedding_model' 参数。")

    embedding_filename = f"{args.dataset_name}.emb-text-{args.embedding_model}.npy"
    embedding_path = os.path.join(emb_dir, embedding_filename)
    output_base_dir = f"{args.model_name}/text-{args.embedding_model}"

    log_dir = os.path.join(args.log_base_path, args.dataset_name, output_base_dir)
    ckpt_dir = os.path.join(args.ckpt_base_path, args.dataset_name, output_base_dir)
    codebook_base_dir = os.path.join(args.codebook_base_path, args.dataset_name, "codebooks")

    for d in [log_dir, ckpt_dir, codebook_base_dir]:
        os.makedirs(d, exist_ok=True)

    print("--- 自动构建路径 ---")
    print(f"输入文本嵌入文件: {embedding_path}")
    print(f"日志目录: {log_dir}")
    print(f"模型目录: {ckpt_dir}")
    print(f"码本根目录: {codebook_base_dir}")
    print("------------------------------------\n")

    return embedding_path, log_dir, ckpt_dir, codebook_base_dir

def setup_logging(log_dir):
    """配置日志记录器"""
    log_filename = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    logging.info("Logging setup complete.")

def build_dedup_layer(base_codes_np: np.ndarray, vocab_size: int):
    """
    为基础码本添加一个去重层。
    对基础码完全相同的条目，在各自簇内部分配 0..k-1 的ID。
    这是一个通用逻辑，可以被任何产生分层码本的模型复用。
    """
    logging.info("构建去重层...")
    N = base_codes_np.shape[0]
    groups = defaultdict(list)
    for idx, key in enumerate(map(tuple, base_codes_np)):
        groups[key].append(idx)

    dedup_layer = np.zeros((N, 1), dtype=np.int64)
    max_dup, overflow_count = 0, 0
    for idx_list in groups.values():
        k = len(idx_list)
        max_dup = max(max_dup, k)
        if k > vocab_size:
            logging.warning(f"一个簇内重复数 {k} > 码本大小 {vocab_size}。去重ID将取模，可能导致碰撞。")
            local_ids = np.arange(k, dtype=np.int64) % vocab_size
            overflow_count += 1
        else:
            local_ids = np.arange(k, dtype=np.int64)
        dedup_layer[np.array(idx_list), 0] = local_ids
    
    logging.info(f"去重层构建完成。最大簇内重复数: {max_dup}。发生取模的簇数量: {overflow_count}。")
    return dedup_layer


def _get_codebook_sizes_for_codes(model_params: dict, num_layers: int):
    codebook_sizes = model_params.get("codebook_sizes")
    if codebook_sizes is None:
        codebook_size = model_params.get("codebook_size")
        if codebook_size is None:
            raise ValueError("缺少 codebook_size/codebook_sizes，无法计算码本利用率。")
        codebook_sizes = [codebook_size] * num_layers

    codebook_sizes = [int(size) for size in codebook_sizes]
    if len(codebook_sizes) != num_layers:
        raise ValueError(
            f"codebook_sizes 长度 ({len(codebook_sizes)}) 与 codes 层数 ({num_layers}) 不一致。"
        )
    if any(size <= 0 for size in codebook_sizes):
        raise ValueError("codebook_size 必须为正数。")
    return codebook_sizes


def calculate_codebook_metrics(codes_np: np.ndarray, model_params: dict):
    """Calculate per-layer codebook usage, SID duplication, and token entropy."""
    codes_np = np.asarray(codes_np)
    if codes_np.ndim != 2:
        raise ValueError(f"codes 必须是二维数组，实际 shape={codes_np.shape}。")

    num_items, num_layers = codes_np.shape
    if num_items == 0:
        raise ValueError("codes 为空，无法计算码本指标。")

    codes_np = codes_np.astype(np.int64, copy=False)
    codebook_sizes = _get_codebook_sizes_for_codes(model_params, num_layers)

    unique_sid_count = int(np.unique(codes_np, axis=0).shape[0])
    duplicate_count = int(num_items - unique_sid_count)
    sid_duplicate_rate = duplicate_count / num_items

    layer_metrics = []
    for layer_idx, codebook_size in enumerate(codebook_sizes):
        layer_codes = codes_np[:, layer_idx]
        if np.any(layer_codes < 0):
            raise ValueError(f"第 {layer_idx} 层 codes 中存在负数 token id。")

        max_token_id = int(layer_codes.max())
        if max_token_id >= codebook_size:
            raise ValueError(
                f"第 {layer_idx} 层 token id 最大值 {max_token_id} "
                f">= codebook_size {codebook_size}。"
            )

        counts = np.bincount(layer_codes, minlength=codebook_size)
        used_tokens = int(np.count_nonzero(counts))
        probs = counts[counts > 0].astype(np.float64) / float(num_items)
        entropy = float(-(probs * np.log(probs)).sum())
        normalized_entropy = (
            entropy / float(np.log(codebook_size)) if codebook_size > 1 else 0.0
        )

        layer_metrics.append({
            "layer": layer_idx,
            "codebook_size": int(codebook_size),
            "used_tokens": used_tokens,
            "unused_tokens": int(codebook_size - used_tokens),
            "utilization": float(used_tokens / codebook_size),
            "entropy": entropy,
            "normalized_entropy": float(normalized_entropy),
        })

    return {
        "num_items": int(num_items),
        "num_layers": int(num_layers),
        "sid": {
            "unique_count": unique_sid_count,
            "duplicate_count": duplicate_count,
            "duplicate_rate": float(sid_duplicate_rate),
        },
        "layers": layer_metrics,
        "averages": {
            "utilization": float(np.mean([m["utilization"] for m in layer_metrics])),
            "entropy": float(np.mean([m["entropy"] for m in layer_metrics])),
            "normalized_entropy": float(
                np.mean([m["normalized_entropy"] for m in layer_metrics])
            ),
        },
    }


def log_codebook_metrics(metrics: dict, prefix: str = "码本指标"):
    sid = metrics["sid"]
    summary_message = (
        f"{prefix}: items={metrics['num_items']}, layers={metrics['num_layers']}, "
        f"unique_sid={sid['unique_count']}, duplicate_count={sid['duplicate_count']}, "
        f"duplicate_rate={sid['duplicate_rate']:.6f}"
    )
    if "mean_prefix_cosine" in metrics.get("averages", {}):
        summary_message += (
            f", avg_mean_cos={_format_optional_float(metrics['averages']['mean_prefix_cosine'])}"
        )
    logging.info(summary_message)

    for layer in metrics["layers"]:
        layer_message = (
            f"{prefix} L{layer['layer']}: "
            f"used={layer['used_tokens']}/{layer['codebook_size']}, "
            f"utilization={layer['utilization']:.6f}, "
            f"entropy={layer['entropy']:.6f}, "
            f"normalized_entropy={layer['normalized_entropy']:.6f}"
        )
        if "mean_prefix_cosine" in layer:
            layer_message += (
                f", groups={layer['prefix_groups']}, "
                f"multi_groups={layer['prefix_multi_groups']}, "
                f"max_group={layer['prefix_max_group_size']}, "
                f"mean_cos={_format_optional_float(layer['mean_prefix_cosine'])}"
            )
        logging.info(layer_message)


def calculate_prefix_cosine_metrics(
    codes_np: np.ndarray,
    embeddings_np: np.ndarray,
    max_items_per_group: int = 1000,
    random_seed: int = 42,
):
    """
    Calculate cosine similarity among item embeddings sharing the same prefix code.

    For layer L, the prefix is codes[:, :L+1]. The reported cosine is averaged
    over prefix groups with at least two items. Large groups are sampled to keep
    periodic training evaluation bounded.
    """
    codes_np = np.asarray(codes_np)
    embeddings_np = np.asarray(embeddings_np)

    if codes_np.ndim != 2:
        raise ValueError(f"codes 必须是二维数组，实际 shape={codes_np.shape}。")
    if embeddings_np.ndim < 2:
        raise ValueError(f"embeddings 必须至少是二维数组，实际 shape={embeddings_np.shape}。")
    if embeddings_np.shape[0] != codes_np.shape[0]:
        raise ValueError(
            f"codes 行数 ({codes_np.shape[0]}) 与 embeddings 行数 ({embeddings_np.shape[0]}) 不一致。"
        )

    num_items, num_layers = codes_np.shape
    embeddings_np = embeddings_np.reshape(num_items, -1).astype(np.float64, copy=False)
    norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    normalized_embeddings = embeddings_np / np.maximum(norms, 1e-12)

    rng = np.random.default_rng(random_seed)
    layer_metrics = []
    max_items_per_group = int(max_items_per_group) if max_items_per_group else 0

    for layer_idx in range(num_layers):
        prefix_codes = codes_np[:, :layer_idx + 1].astype(np.int64, copy=False)
        _, inverse = np.unique(prefix_codes, axis=0, return_inverse=True)

        order = np.argsort(inverse, kind="stable")
        sorted_inverse = inverse[order]
        group_starts = np.r_[0, np.flatnonzero(np.diff(sorted_inverse)) + 1]
        group_ends = np.r_[group_starts[1:], len(order)]

        group_sizes = group_ends - group_starts
        num_groups = int(len(group_sizes))
        multi_group_mask = group_sizes > 1
        multi_item_groups = int(np.count_nonzero(multi_group_mask))
        singleton_groups = int(num_groups - multi_item_groups)
        total_items_in_multi_groups = int(group_sizes[multi_group_mask].sum())
        max_group_size = int(group_sizes.max()) if num_groups > 0 else 0

        group_means = []
        sampled_groups = 0

        for start, end in zip(group_starts[multi_group_mask], group_ends[multi_group_mask]):
            item_indices = order[start:end]
            group_size = int(item_indices.shape[0])
            if max_items_per_group > 1 and group_size > max_items_per_group:
                item_indices = rng.choice(item_indices, size=max_items_per_group, replace=False)
                sampled_groups += 1

            group_embeddings = normalized_embeddings[item_indices]
            sampled_size = int(group_embeddings.shape[0])
            if sampled_size < 2:
                continue

            summed = group_embeddings.sum(axis=0)
            diagonal_sum = float(np.einsum("ij,ij->i", group_embeddings, group_embeddings).sum())
            pair_sum_twice = float(np.dot(summed, summed) - diagonal_sum)
            mean_cosine = pair_sum_twice / float(sampled_size * (sampled_size - 1))

            group_means.append(mean_cosine)

        mean_prefix_cosine = None
        if group_means:
            group_means = np.asarray(group_means, dtype=np.float64)
            mean_prefix_cosine = float(group_means.mean())

        layer_metrics.append({
            "layer": int(layer_idx),
            "prefix_length": int(layer_idx + 1),
            "num_groups": num_groups,
            "singleton_groups": singleton_groups,
            "multi_item_groups": multi_item_groups,
            "avg_group_size": float(num_items / num_groups) if num_groups > 0 else 0.0,
            "avg_multi_item_group_size": (
                float(total_items_in_multi_groups / multi_item_groups)
                if multi_item_groups > 0 else 0.0
            ),
            "max_group_size": max_group_size,
            "sampled_groups": int(sampled_groups),
            "mean_prefix_cosine": mean_prefix_cosine,
        })

    valid_means = [
        layer["mean_prefix_cosine"]
        for layer in layer_metrics
        if layer["mean_prefix_cosine"] is not None
    ]

    return {
        "num_items": int(num_items),
        "num_layers": int(num_layers),
        "max_items_per_group": int(max_items_per_group),
        "layers": layer_metrics,
        "averages": {
            "mean_prefix_cosine": float(np.mean(valid_means)) if valid_means else None,
        },
    }


def merge_prefix_cosine_into_codebook_metrics(codebook_metrics: dict, prefix_cosine_metrics: dict):
    if codebook_metrics["num_layers"] != prefix_cosine_metrics["num_layers"]:
        raise ValueError(
            "codebook 指标层数与 prefix cosine 指标层数不一致，"
            f"{codebook_metrics['num_layers']} != {prefix_cosine_metrics['num_layers']}。"
        )

    codebook_metrics["averages"]["mean_prefix_cosine"] = (
        prefix_cosine_metrics["averages"]["mean_prefix_cosine"]
    )
    codebook_metrics["prefix_cosine_max_items_per_group"] = (
        prefix_cosine_metrics["max_items_per_group"]
    )

    for codebook_layer, prefix_layer in zip(
        codebook_metrics["layers"],
        prefix_cosine_metrics["layers"],
    ):
        if codebook_layer["layer"] != prefix_layer["layer"]:
            raise ValueError(
                "codebook 指标层号与 prefix cosine 指标层号不一致，"
                f"{codebook_layer['layer']} != {prefix_layer['layer']}。"
            )
        codebook_layer["prefix_groups"] = prefix_layer["num_groups"]
        codebook_layer["prefix_multi_groups"] = prefix_layer["multi_item_groups"]
        codebook_layer["prefix_max_group_size"] = prefix_layer["max_group_size"]
        codebook_layer["mean_prefix_cosine"] = prefix_layer["mean_prefix_cosine"]

    return codebook_metrics


def _format_optional_float(value):
    return "NA" if value is None else f"{value:.6f}"


def calc_cos_sim(model, data, config):
    if len(data.shape) > 2:
        data = data[:, 0, :]
    ids = model.get_codes(data).cpu().numpy()
    max_item_calculate = 1000
    data_np = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else np.asarray(data)
    metrics = calculate_prefix_cosine_metrics(
        ids,
        data_np,
        max_items_per_group=max_item_calculate,
        random_seed=42,
    )
    return np.array([
        layer["mean_prefix_cosine"] if layer["mean_prefix_cosine"] is not None else 0.0
        for layer in metrics["layers"][:config["num_levels"]]
    ])


def process_embeddings(config, device, id2meta_file=None, embedding_save_path=None):
    category = config["dataset"]["name"]
    type = config["dataset"]["type"]
    final_output_path = os.path.join("cache", type, category, "processed", "final_pca_embeddings.npy")

    if not os.path.exists(final_output_path):
        raise FileNotFoundError(f"Embedding file not found: {final_output_path}")

    np_array = np.load(final_output_path)
    tensor = torch.from_numpy(np_array).to(device, dtype=torch.float32)
    print(f"[QUANTIZATION] Loaded embeddings from '{final_output_path}', shape={tensor.shape}, dtype={tensor.dtype}")
    return tensor


def set_weight_decay(optimizer, weight_decay):
    for param_group in optimizer.param_groups:
        param_group["weight_decay"] = weight_decay

def build_codebook_path(codebook_base_path: str, dataset_name: str, 
                        model_name: str, 
                        embedding_model: str = None,
                        embedding_modality: str = 'text') -> str:
    """
    构建文本模态码本路径，例如 Baby.text.rqvae.npy。
    """
    ds = str(dataset_name)
    model_tag = str(model_name).lower()
    mod_tag = str(embedding_modality or 'text').lower()
    if mod_tag != 'text':
        raise ValueError("当前仅支持文本模态码本路径。")

    dir_path = os.path.join(codebook_base_path, ds, "codebooks")
    os.makedirs(dir_path, exist_ok=True)

    filename = f"{ds}.text.{model_tag}.npy"

    return os.path.join(dir_path, filename)
