import argparse
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
        '--embedding_modality',
        type=str,
        default='text',
        choices=['text'],
        help='嵌入模态；当前仅支持 text。'
    )
    parser.add_argument(
        '--embedding_model',
        type=str,
        required=True,
        help='文本嵌入来源模型名称 (e.g., sentence-t5-base)。'
    )
    parser.add_argument('--config_path', type=str, default=None, help='配置文件路径。')
    parser.add_argument('--data_base_path', type=str, default=os.path.abspath(os.path.join(BASE_DIR, '../datasets')), help='数据集根目录')
    parser.add_argument('--log_base_path', type=str, default=os.path.abspath(os.path.join(BASE_DIR, '../logs/quantization')), help='日志根目录')
    parser.add_argument('--ckpt_base_path', type=str, default=os.path.abspath(os.path.join(BASE_DIR, '../ckpt/quantization')), help='模型根目录')
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
        f"embedding={args.embedding_model}, shape={item_embeddings.shape}, device={device}"
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
        embedding_modality=args.embedding_modality
    )
    logging.info(f"最终码本将保存到: {final_codebook_path}")

    trainer.predict(
        embeddings_data=item_embeddings,
        output_path=final_codebook_path
    )

    logging.info("\n--- 所有任务完成 ---")


if __name__ == '__main__':
    main()
