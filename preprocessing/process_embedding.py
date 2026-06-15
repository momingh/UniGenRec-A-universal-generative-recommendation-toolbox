import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOCAL_TEXT_MODEL = str(SCRIPT_DIR / "emb_llm" / "Qwen3-Embedding-8B")
OPENAI_API_KEY = 'sk-poTkiwhHfjdaP83BK8lAHHUxGKRjxuLcJAv7fhfhY3sGVf8c'
DEFAULT_OPENAI_BASE_URL = "https://yunwu.ai/v1"

sys.path.append(str(SCRIPT_DIR))
sys.path.append(str(SCRIPT_DIR / "generate_embedding"))

try:
    from utils import (
        apply_pca_and_save,
        build_output_path,
        build_text_map,
        get_id2item_dict,
        load_json,
        set_device,
    )
except ImportError as exc:
    print(f"导入 preprocessing/utils.py 失败: {exc}")
    sys.exit(1)

try:
    import cf_encoder
    import image_encoder
    import text_encoder
    import vlm_encoder
except ImportError as exc:
    print(f"导入 encoder 模块失败: {exc}")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="统一的 embedding 生成脚本")

    parser.add_argument("--embedding_type", required=True, choices=["text_local", "text_api", "image_clip", "cf_sasrec", "vlm_fused"], help="要生成的 embedding 类型")
    parser.add_argument("--dataset", required=True, help="数据集名称")
    parser.add_argument("--dataset_type", default="amazon", choices=["amazon", "movielens"], help="数据集类型")
    parser.add_argument("--data_version", default="14", choices=["14", "18"], help="Amazon 版本")
    parser.add_argument("--save_root", default="../datasets", help="预处理数据根目录")
    parser.add_argument("--image_root", default="../datasets", help="包含 amazonXX/Images 的根目录")

    parser.add_argument("--model_name_or_path", default=DEFAULT_LOCAL_TEXT_MODEL, help="[text_local] 本地 Transformer 模型")
    parser.add_argument("--max_sent_len", type=int, default=1024, help="[text_local] 文本最大长度")
    parser.add_argument("--sent_emb_model", default="text-embedding-3-large", help="[text_api] OpenAI 模型 ID")
    parser.add_argument("--api_emb_dim", type=int, default=0, help="[text_api] API 输出维度，0 为自动")
    parser.add_argument("--openai_api_key", default=os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY), help="OpenAI API key")
    parser.add_argument("--openai_base_url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL), help="OpenAI base URL")
    parser.add_argument("--clip_model_name", default="openai/clip-vit-base-patch32", help="[image_clip] CLIP 模型 ID")
    parser.add_argument("--sasrec_hidden_dim", type=int, default=64, help="[cf_sasrec] 隐藏维度")
    parser.add_argument("--sasrec_max_seq_len", type=int, default=50, help="[cf_sasrec] 序列长度")
    parser.add_argument("--sasrec_n_layers", type=int, default=2, help="[cf_sasrec] 层数")
    parser.add_argument("--sasrec_n_heads", type=int, default=2, help="[cf_sasrec] 头数")
    parser.add_argument("--sasrec_dropout", type=float, default=0.2, help="[cf_sasrec] dropout")
    parser.add_argument("--sasrec_epochs", type=int, default=30, help="[cf_sasrec] 训练轮数")
    parser.add_argument("--sasrec_lr", type=float, default=0.001, help="[cf_sasrec] 学习率")
    parser.add_argument("--sasrec_weight_decay", type=float, default=0.0, help="[cf_sasrec] 权重衰减")
    parser.add_argument("--vlm_model_name_or_path", default="Qwen/Qwen3-VL-7B-Instruct", help="[vlm_fused] VLM 模型 ID")
    parser.add_argument("--vlm_prompt_template", default="Represent this item for recommendation: {}", help="[vlm_fused] prompt 模板")

    parser.add_argument("--model_cache_dir", default=None, help="Hugging Face 缓存目录")
    parser.add_argument("--batch_size", type=int, default=100, help="批处理大小")
    parser.add_argument("--pca_dim", type=int, default=512, help="PCA 目标维度，<=0 不降维")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID，<0 使用 CPU")

    return parser.parse_args()


def load_common_data(args):
    data_dir = Path(args.save_root) / args.dataset
    item2id_path = data_dir / f"{args.dataset}.item2id"
    item_meta_path = data_dir / f"{args.dataset}.item.json"

    print(f"\n--- 加载数据: {args.dataset} ---")
    print(f"item2id: {item2id_path}")
    id2item = get_id2item_dict(str(item2id_path))

    print(f"item meta: {item_meta_path}")
    item_meta = load_json(str(item_meta_path))
    if not item_meta:
        raise FileNotFoundError(f"item.json 文件加载失败或为空: {item_meta_path}")

    images_info, image_dir = load_image_data(args)
    return id2item, item_meta, images_info, image_dir


def load_image_data(args):
    if args.embedding_type not in {"image_clip", "vlm_fused"}:
        return None, None

    image_base_path = Path(args.image_root) / f"amazon{args.data_version}" / "Images"
    if not image_base_path.is_dir():
        print(f"[WARN] 图像基础路径不存在: {image_base_path}")
        return {}, None

    images_info_path = image_base_path / f"{args.dataset}_images_info.json"
    image_dir = image_base_path / args.dataset

    print(f"image info: {images_info_path}")
    images_info = load_json(str(images_info_path)) or {}
    if not images_info:
        print(f"[WARN] image info 文件缺失或为空: {images_info_path}")

    if not image_dir.is_dir():
        print(f"[WARN] 图像目录不存在: {image_dir}")
        return images_info, None

    return images_info, str(image_dir)


def build_item_text_list(id2item, text_map):
    return [
        [int(new_id), text_map.get(orig_id, "N/A")]
        for new_id, orig_id in sorted(id2item.items(), key=lambda pair: int(pair[0]))
    ]


def generate_embeddings(args, id2item, item_meta, images_info, image_dir):
    text_map = {}
    if args.embedding_type in {"text_local", "text_api", "vlm_fused"}:
        text_map = build_text_map(args, id2item, item_meta)
        if not text_map:
            print("[WARN] text_map 为空。")

    if args.embedding_type == "text_local":
        return (
            text_encoder.generate_local_text(args, build_item_text_list(id2item, text_map)),
            "text",
            args.model_name_or_path,
        )

    if args.embedding_type == "text_api":
        return (
            text_encoder.generate_api_text(args, build_item_text_list(id2item, text_map)),
            "text",
            args.sent_emb_model,
        )

    if args.embedding_type == "image_clip":
        if not image_dir:
            raise ValueError("图像目录未找到或无效，无法生成图像嵌入。")
        return (
            image_encoder.generate_clip_image(args, id2item, images_info, image_dir),
            "image",
            args.clip_model_name,
        )

    if args.embedding_type == "cf_sasrec":
        return (
            cf_encoder.train_and_extract_sasrec(args, len(id2item)),
            "cf",
            "sasrec",
        )

    if args.embedding_type == "vlm_fused":
        if not image_dir:
            raise ValueError("图像目录未找到或无效，无法生成 VLM 融合嵌入。")
        return (
            vlm_encoder.generate_vlm_fused(args, id2item, text_map, images_info, image_dir),
            "vlm-fused",
            args.vlm_model_name_or_path,
        )

    raise ValueError(f"未知 embedding_type: {args.embedding_type}")


def validate_embeddings(embeddings, expected_rows):
    if embeddings is None or not isinstance(embeddings, np.ndarray) or embeddings.size == 0:
        raise RuntimeError("未能生成有效 embedding。")

    if embeddings.shape[0] != expected_rows:
        print(f"[WARN] embedding 数量 ({embeddings.shape[0]}) 与 item 数量 ({expected_rows}) 不匹配。")
        embeddings = embeddings[:expected_rows]
        if embeddings.shape[0] != expected_rows:
            raise RuntimeError("截断后数量仍不匹配。")

    return embeddings


def main():
    args = parse_args()
    args.device = set_device(args.gpu_id)

    try:
        id2item, item_meta, images_info, image_dir = load_common_data(args)
        print(f"\n--- 生成 embedding: {args.embedding_type} ---")
        embeddings, modality_tag, model_tag = generate_embeddings(
            args,
            id2item,
            item_meta,
            images_info,
            image_dir,
        )
        embeddings = validate_embeddings(embeddings, len(id2item))

        output_path = build_output_path(args, modality_tag, model_tag)
        final_output_path = apply_pca_and_save(embeddings, args, output_path)
        if not final_output_path:
            raise RuntimeError("保存 embedding 失败。")

        print(f"\n任务完成，最终 embedding 保存至: {final_output_path}")
    except Exception as exc:
        print(f"错误: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
