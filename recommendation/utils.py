# utils.py
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import numpy as np
import random
import torch
import logging
from pathlib import Path
import importlib
from collections.abc import Mapping
from numbers import Real

VALID_QUANT_METHODS = {"rqvae", "rqvae_faiss", "opq", "qinco", "qinco_aux", "qinco_v2", "rqkmeans", "rqkmeans_plus"}


def _load_yaml_file(path: Path | str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _ensure_dir_exists(dir_path: Path):
    """确保目录存在"""
    if dir_path:
        dir_path.mkdir(parents=True, exist_ok=True)

def _recursive_update(base_dict: dict, new_dict: dict) -> dict:
    """
    递归地更新字典。
    如果 new_dict 中的键在 base_dict 中也存在且对应的值都是字典，
    则递归地合并它们，否则直接用 new_dict 的值覆盖 base_dict 的值。
    """
    for key, value in new_dict.items():
        if isinstance(value, Mapping) and key in base_dict and isinstance(base_dict[key], Mapping):
            base_dict[key] = _recursive_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def format_metric_value(value, digits: int = 8) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, Real) and not isinstance(value, bool):
        return f"{float(value):.{digits}f}"
    return repr(value)


def _metric_sort_key(metric: str):
    metric_order = {
        "Recall": 0,
        "NDCG": 1,
        "Precision": 2,
        "Hit": 3,
        "MRR": 4,
    }
    if "@" not in metric:
        return (1, metric, 0, "")

    name, k_text = metric.rsplit("@", 1)
    try:
        k_value = int(k_text)
    except ValueError:
        k_value = 10**9
    return (0, k_value, metric_order.get(name, 99), name)


def _ordered_metric_items(metrics: Mapping):
    return sorted(metrics.items(), key=lambda item: _metric_sort_key(str(item[0])))


def format_metrics(metrics: Mapping | None, digits: int = 8) -> str:
    """Format metric dictionaries in a stable, compact order for logs."""
    if metrics is None:
        return "None"

    return " | ".join(
        f"{key}={format_metric_value(value, digits)}"
        for key, value in _ordered_metric_items(metrics)
    )


def format_metrics_line(label: str, metrics: Mapping | None, digits: int = 8, label_width: int = 18) -> str:
    return f"{label:<{label_width}} | {format_metrics(metrics, digits)}"


def _load_quant_details(path: str, quant_method: str) -> dict:
    """
    从指定的 YAML 文件中，根据 quant_method 载入对应的参数。
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"[Config] 根據約定，未找到量化設定檔: {path}")
    
    cfg = _load_yaml_file(path)
        
    if quant_method not in cfg or 'model_params' not in cfg[quant_method]:
        raise ValueError(f"[Config] 在 {path} 中缺少 '{quant_method}.model_params' 節點")
    
    mp = dict(cfg[quant_method]['model_params'])
    if 'num_levels' not in mp and 'n_codebook' in mp:
        mp['num_levels'] = mp['n_codebook']
    if 'has_dup_layer' not in mp:
        mp['has_dup_layer'] = False

    required_keys = ['codebook_size', 'num_levels']
    if not all(key in mp for key in required_keys):
         raise ValueError(f"[Config] 在 {path} 的 model_params 中缺少 'codebook_size' 和 'num_levels/n_codebook'")

    return mp


def load_and_process_config(model_name: str, dataset_name: str, quant_method: str, embedding_modality: str = 'text') -> dict:
    """
    通用配置加载器 (V6 - 支援 base.yaml 继承与覆盖)。
    """
    # === 1. ✅ 关键改动 2：依序载入 base 和 model-specific 配置文件 ===
    # 载入基础配置文件
    base_config_path = Path("configs/base.yaml")
    if not base_config_path.is_file():
        raise FileNotFoundError(f"基礎設定檔未找到: {base_config_path}")
    config = _load_yaml_file(base_config_path)

    # 载入特定模型配置文件
    model_config_path = Path(f"configs/{model_name}.yaml")
    if not model_config_path.is_file():
        raise FileNotFoundError(f"模型配置文件未找到: {model_config_path}")
    model_config = _load_yaml_file(model_config_path)
        
    # ✨ 使用递归更新，让 model_config 覆盖 base_config
    config = _recursive_update(config, model_config)

    # === 后续流程完全不变，它们现在操作的是已经合并好的 config ===
    
    config['model_name'] = model_name
    config['dataset_name'] = dataset_name
    config['quant_method'] = quant_method.lower()
    config['embedding_modality'] = embedding_modality.lower()
    
    if config['quant_method'] not in VALID_QUANT_METHODS:
        raise ValueError(f"不支持的量化方法: {quant_method}。可选: {VALID_QUANT_METHODS}")

    # 2. 独立地载入量化配置文件
    quant_config_path = Path(f"../quantization/configs/{quant_method}_config.yaml")
    quant_details = _load_quant_details(quant_config_path, config['quant_method'])
    
    # === 3. 格式化和派生「数据」与「输出」的路径 ===
    # 这部分的路径模板仍然来自 TIGER.yaml，是合理的，因为它定义了数据存放格式
    # 3. 格式化路径
    paths = config['paths']
    # ✅ 将 model_name 加入字典
    format_args = {
        'dataset_name': dataset_name, 
        'quant_method': config['quant_method'], 
        'model_name': model_name
    }
    dataset_root = Path(paths['dataset_root'].format(**format_args))
    output_root = Path(paths['output_root'].format(**format_args))

        # === 4. 自动构造 codebook 路径 ===
    dataset_root = Path(f"../datasets/{dataset_name}")
    codebook_dir = dataset_root / "codebooks"

    mod_tag = embedding_modality.lower()
    quant_tag = config['quant_method'].lower()

    # 严格匹配指定模态和量化方法
    codebook_path = codebook_dir / f"{dataset_name}.{mod_tag}.{quant_tag}.npy"

    if not codebook_path.exists():
        raise FileNotFoundError(
            f"[FATAL] 未找到指定模态 '{mod_tag}' 的 codebook 文件！\n"
            f"期望路径: {codebook_path}\n"
            f"请确认路径及文件名与保存时一致。"
        )

    config['code_path'] = str(codebook_path)
    logging.info(f"📦 [Config] 成功加载 Codebook: {config['code_path']}")

    config['log_path'] = output_root / "training.log"
    config['save_path'] = output_root / "best_model.pth"
    config['dataset_root'] = dataset_root
    config['inter_json'] = dataset_root / f"{dataset_name}.inter.json"
    _ensure_dir_exists(output_root)

    # === 4. 根据载入的量化细节，计算词表参数 ===
    K = int(quant_details['codebook_size'])
    num_semantic_levels = int(quant_details['num_levels'])
    has_dup_layer = quant_details.get('has_dup_layer', True) 
    
    config['codebook_size'] = K
    config['num_semantic_levels'] = num_semantic_levels

    # === 5. 校验 codebook 文件 ===
    if not Path(config['code_path']).is_file():
        raise FileNotFoundError(f"[FATAL] 未找到 codebook: {config['code_path']}")
    codes_arr = np.load(config['code_path'], allow_pickle=True)
    codes_mat = np.vstack(codes_arr) if codes_arr.dtype == object else codes_arr
    
    expected_code_len = num_semantic_levels + 1 if has_dup_layer else num_semantic_levels
    config['code_len'] = expected_code_len

    if codes_mat.ndim != 2 or codes_mat.shape[1] != expected_code_len:
        raise ValueError(f"[FATAL] Codebook {config['code_path']} 的期望形状為 (N, {expected_code_len})，實際為 {codes_mat.shape}")

    # === 6. 计算最终词表参数 ===
    if has_dup_layer:
        dup_max = int(codes_mat[:, -1].max()) if codes_mat.size > 0 else 0
        dup_vocab_size = dup_max + 1
        config['dup_vocab_size'] = dup_vocab_size
        vocab_sizes = [K] * num_semantic_levels + [dup_vocab_size]
    else:
        vocab_sizes = [K] * num_semantic_levels
    config['vocab_sizes'] = vocab_sizes

    bases = np.cumsum([0] + vocab_sizes[:-1]).tolist()
    config['bases'] = [int(b) for b in bases]

    # === 6. 定义特殊 Token (统一放置在词表的尾端，避免与 Code 冲突) ===
    base_vocab = sum(config['vocab_sizes'])  # 语义 token 数量
    # 规范：PAD 固定为 0，语义 token 全部偏移 +1，特殊 token 追加在词表尾部
    pad_id = 0
    mask_id = base_vocab + 1
    cls_id = base_vocab + 2
    sep_id = base_vocab + 3
    eos_id = base_vocab + 4  # 保留 EOS
    vocab_size = eos_id + 1  # ID 范圍 0..eos_id

    # === 6.1 (可选) 用户 Token 区间 (对齐 NonameUntitled/tiger) ===
    # 在词表尾部、特殊 token 之后再划出 user_ids_count 个槽位给 user token。
    # user id 经哈希取模落入 0..user_ids_count-1，再加 user_token_base 偏移成最终 token id。
    # 仅当 config 顶层声明了 user_ids_count(且 > 0) 时启用；GPT2/RPG 等不声明则不受影响。
    user_ids_count = int(config.get('user_ids_count', 0) or 0)
    if user_ids_count > 0:
        user_token_base = vocab_size  # 紧接在 eos_id 之后
        vocab_size = user_token_base + user_ids_count
        config['user_ids_count'] = user_ids_count
        config['user_token_base'] = user_token_base
        logging.info(
            f"[Config] 启用 user token: user_ids_count={user_ids_count}, "
            f"user_token_base={user_token_base}, 扩展后 vocab_size={vocab_size}"
        )

    config['token_params'] = {
        'pad_token_id': pad_id,
        'cls_token_id': cls_id,
        'sep_token_id': sep_id,
        'mask_token_id': mask_id,
        'eos_token_id': eos_id,
        'vocab_size': vocab_size
    }

    return config


# --- 日志和随机种子函数保持不变 ---
def setup_logging(log_path: Path):
    """配置日志记录器"""
    class NewBestFormatter(logging.Formatter):
        red = "\033[31m"
        reset = "\033[0m"

        def format(self, record):
            if "New best |" not in record.getMessage():
                return super().format(record)

            original_msg = record.msg
            original_args = record.args
            try:
                record.msg = f"{self.red}{record.getMessage()}{self.reset}"
                record.args = ()
                return super().format(record)
            finally:
                record.msg = original_msg
                record.args = original_args

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(NewBestFormatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(sh)

    fh = logging.FileHandler(str(log_path), mode='a', encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

def get_model_class(model_name: str):
    """
    增强版：根据模型名称字符串，动态地从 'models' 目录及其所有子目录中
    搜索并导入对应的模型类。

    Args:
        model_name (str): 模型的名称 (e.g., "TIGER")。
                          约定：文件名和类名都应与 model_name 一致 (TIGER.py, class TIGER)。
    """
    models_root_dir = "models"
    model_file_name = f"{model_name}.py"
    model_module_path = None

    # os.walk 会遍历指定目录下的所有文件夹和文件
    for root, dirs, files in os.walk(models_root_dir):
        if model_file_name in files:
            # 找到了文件！现在构建 Python 的导入路径
            # 例如, root = "models/encoder_decoder"
            
            # 1. 将文件系统路径 ('/') 替换为 Python 导入路径 ('.')
            # "models/encoder_decoder" -> "models.encoder_decoder"
            base_path = root.replace(os.sep, '.')
            
            # 2. 拼接成最终的模块路径
            # "models.encoder_decoder.TIGER"
            model_module_path = f"{base_path}.{model_name}"
            break # 找到后立刻停止搜索

    # 如果遍历完都没找到文件
    if not model_module_path:
        raise ImportError(
            f"错误：无法在 '{models_root_dir}' 目录或其任何子目录中找到模型文件 '{model_file_name}'。\n"
            f"请检查你的文件结构和 --model 参数是否正确。"
        )

    try:
        # 使用动态构建的路径来导入模块
        logging.info(f"Found model module at: {model_module_path}")
        model_module = importlib.import_module(model_module_path)
        # 从模块中获取与模型同名的类
        model_class = getattr(model_module, model_name)
        return model_class
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"错误：成功找到文件，但在从 '{model_module_path}' 导入类 '{model_name}' 时失败。\n"
            f"请确保你的 Python 文件内 class 的名称 ({model_name}) 与文件名和 --model 参数完全一致。\n"
            f"原始错误: {e}"
        )

def set_seed(seed: int):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
