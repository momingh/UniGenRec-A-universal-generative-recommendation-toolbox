import argparse
import json
import logging
import os

import numpy as np
import torch
from sklearn.decomposition import PCA

import utils
from trainer import Trainer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="通用量化器训练脚本")

    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        choices=['rqvae', 'rqvae_faiss', 'opq', 'qinco', 'qinco_aux', 'qinco_v2', 'rqkmeans', 'rqkmeans_plus'],
        help='要使用的量化器模型名称。'
    )
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称 (e.g., Baby)')
    parser.add_argument(
        '--embedding_source',
        type=str,
        default='text',
        choices=['text', 'graph'],
        help='输入 embedding 来源：text=预处理文本 embedding，graph=图模型输出的 item embedding。'
    )
    parser.add_argument(
        '--embedding_modality',
        type=str,
        default='text',
        choices=['text'],
        help='嵌入模态；当前仅支持 text。'
    )
    parser.add_argument(
        '--embedding_model',
        type=str,
        default=None,
        help='文本嵌入来源模型名称 (e.g., sentence-t5-base)。'
    )
    parser.add_argument(
        '--graph_model',
        type=str,
        default=None,
        help='当 embedding_source=graph 时使用的图模型名称，例如 lightgcn。'
    )
    parser.add_argument('--config_path', type=str, default=None, help='配置文件路径。')
    parser.add_argument('--data_base_path', type=str, default=os.path.abspath(os.path.join(BASE_DIR, '../datasets')), help='数据集根目录')
    parser.add_argument('--log_base_path', type=str, default=os.path.abspath(os.path.join(BASE_DIR, '../logs/quantization')), help='日志根目录')
    parser.add_argument('--ckpt_base_path', type=str, default=os.path.abspath(os.path.join(BASE_DIR, '../ckpt/quantization')), help='模型根目录')
    parser.add_argument('--graph_ckpt_base_path', type=str, default=os.path.abspath(os.path.join(BASE_DIR, '../ckpt/graph')), help='图模型输出根目录')
    parser.add_argument('--codebook_base_path', type=str, default=os.path.abspath(os.path.join(BASE_DIR, '../datasets')), help='码本根目录')
    parser.add_argument('--pca_dim', type=int, default=None, help='可选 PCA 降维目标维度；<=0 表示不降维；未传时读取配置文件。')
    parser.add_argument(
        '--checkpoint_selection',
        type=str,
        default=None,
        choices=['final', 'best_val'],
        help='生成 code 时使用的 checkpoint：final=最后一次训练模型，best_val=验证集 MSE 最优模型。未传时读取配置文件。'
    )
    parser.add_argument(
        '--early_stop_patience',
        type=int,
        default=None,
        help='checkpoint_selection=best_val 时，验证集 MSE 连续多少个 epoch 未提升后早停。未传时读取配置文件。'
    )

    args = parser.parse_args()

    embedding_path, log_dir, ckpt_dir, _codebook_base_dir = utils.setup_paths(args)

    utils.setup_logging(log_dir)

    config_path = args.config_path or os.path.join(BASE_DIR, "configs", f"{args.model_name}_config.yaml")

    config = utils.load_yaml_file(config_path)
    config['model_name'] = args.model_name
    config['dataset_name'] = args.dataset_name

    seed = int(config.get('common', {}).get('seed', 42))
    utils.set_seed(seed)

    model_cfg = config.get(args.model_name.lower(), {})
    train_cfg = model_cfg.get("training_params", {})
    if args.checkpoint_selection is not None:
        train_cfg["checkpoint_selection"] = args.checkpoint_selection
    if args.early_stop_patience is not None:
        train_cfg["early_stop_patience"] = args.early_stop_patience
    pca_dim = args.pca_dim
    if pca_dim is None:
        pca_dim = train_cfg.get("pca_dim", 0)
    pca_dim = int(pca_dim or 0)

    item_embeddings = np.load(embedding_path)
    mapping_records = None
    mapping_path = resolve_embedding_mapping_path(args, embedding_path)
    if mapping_path is not None:
        item_embeddings, mapping_records = align_embeddings_by_item_mapping(
            item_embeddings,
            mapping_path,
        )
        logging.info(f"已按 item_id 对齐 embedding mapping: {mapping_path}")
    if pca_dim > 0:
        original_dim = item_embeddings.shape[1]
        if pca_dim >= original_dim:
            logging.warning(
                f"pca_dim={pca_dim} >= 原始维度 {original_dim}，跳过 PCA 降维。"
            )
        elif pca_dim > min(item_embeddings.shape):
            raise ValueError(
                f"pca_dim={pca_dim} 超过 PCA 允许的最大维度 "
                f"{min(item_embeddings.shape)}，embedding shape={item_embeddings.shape}。"
            )
        else:
            logging.info(
                f"对 item_embeddings 执行 PCA 降维: {item_embeddings.shape} -> "
                f"({item_embeddings.shape[0]}, {pca_dim})"
            )
            logging.info(f"执行 PCA (n_components={pca_dim}, whiten=True)...")
            pca = PCA(n_components=pca_dim, whiten=True, random_state=42)
            item_embeddings = pca.fit_transform(item_embeddings).astype(np.float32)
            kept_variance = float(np.sum(pca.explained_variance_ratio_))
            logging.info(f"PCA 完成，保留方差比例: {kept_variance:.6f}")

    config['total_item_count'] = len(item_embeddings)
    input_size = item_embeddings.shape[1]

    device = torch.device(config.get('common', {}).get('device', 'cuda:0'))
    logging.info(
        f"任务: model={args.model_name}, dataset={args.dataset_name}, "
        f"embedding_source={args.embedding_source}, shape={item_embeddings.shape}, device={device}"
    )

    ModelClass = utils.get_model(args.model_name)
    model = ModelClass(config=config, input_size=input_size, item_embeddings=item_embeddings).to(device)

    trainer = Trainer(config=config, model=model, device=device)
    model_path = trainer.fit(
        embeddings_data=item_embeddings,
        ckpt_dir=ckpt_dir
    )

    if model_path and os.path.exists(model_path):
        if model_path.endswith((".pth", ".pt")):
            model.load_state_dict(torch.load(model_path, map_location=device))
            logging.info(f"已加载用于生成 code 的 checkpoint: {model_path}")
        else:
            logging.info(f"One-shot 拟合信号已生成: {model_path}。跳过 PyTorch checkpoint 加载。")
    else:
        logging.warning(f"未找到或无效的模型路径: {model_path}。使用当前内存中的模型。")

    final_codebook_path = utils.build_codebook_path(
        codebook_base_path=args.codebook_base_path,
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        embedding_model=args.embedding_model,
        embedding_modality=resolve_codebook_modality(args)
    )
    logging.info(f"最终码本将保存到: {final_codebook_path}")

    trainer.predict(
        embeddings_data=item_embeddings,
        output_path=final_codebook_path
    )
    if mapping_records is not None:
        mapping_output_path = final_codebook_path.replace(".npy", ".mapping.jsonl")
        write_codebook_mapping(mapping_output_path, mapping_records)
        logging.info(f"已保存 codebook row 到 item 的映射: {mapping_output_path}")

    logging.info("\n--- 所有任务完成 ---")


def resolve_codebook_modality(args) -> str:
    if args.embedding_source == "graph":
        return f"graph-{args.graph_model}"
    return args.embedding_modality


def resolve_embedding_mapping_path(args, embedding_path: str):
    if args.embedding_source != "graph":
        return None

    graph_dir = os.path.dirname(os.path.abspath(embedding_path))
    mapping_path = os.path.join(graph_dir, "final_item_embedding_mapping.jsonl")
    if not os.path.isfile(mapping_path):
        raise FileNotFoundError(
            f"Graph embedding mapping not found: {mapping_path}"
        )
    return mapping_path


def align_embeddings_by_item_mapping(item_embeddings: np.ndarray, mapping_path: str):
    if item_embeddings.ndim != 2:
        raise ValueError(f"item_embeddings 必须是二维数组，实际 shape={item_embeddings.shape}")

    num_items = int(item_embeddings.shape[0])
    aligned = np.empty_like(item_embeddings)
    records = [None] * num_items
    seen_rows = set()
    seen_item_ids = set()

    with open(mapping_path, "r", encoding="utf-8") as fp:
        for line_num, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            source_row = int(obj.get("row_index", line_num - 1))
            item_id = int(obj["item_id"])
            if not 0 <= source_row < num_items:
                raise ValueError(
                    f"{mapping_path}:{line_num} row_index out of range: {source_row}"
                )
            if not 0 <= item_id < num_items:
                raise ValueError(
                    f"{mapping_path}:{line_num} item_id out of range: {item_id}"
                )
            if source_row in seen_rows:
                raise ValueError(f"{mapping_path}:{line_num} duplicate row_index: {source_row}")
            if item_id in seen_item_ids:
                raise ValueError(f"{mapping_path}:{line_num} duplicate item_id: {item_id}")

            seen_rows.add(source_row)
            seen_item_ids.add(item_id)
            aligned[item_id] = item_embeddings[source_row]
            records[item_id] = {
                "row_index": item_id,
                "item_id": item_id,
                "raw_item_id": obj.get("raw_item_id"),
            }

    missing_rows = sorted(set(range(num_items)) - seen_rows)
    missing_item_ids = sorted(set(range(num_items)) - seen_item_ids)
    if missing_rows:
        raise ValueError(f"{mapping_path} missing row mappings: {missing_rows[:10]}")
    if missing_item_ids:
        raise ValueError(f"{mapping_path} missing item_id mappings: {missing_item_ids[:10]}")

    return aligned, records


def write_codebook_mapping(path: str, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    main()
