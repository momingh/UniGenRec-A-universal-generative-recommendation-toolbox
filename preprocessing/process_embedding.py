# preprocessing/generate_embeddings/main_generate.py

import argparse
import os
import sys
import numpy as np
import torch 
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOCAL_TEXT_MODEL = os.path.join(SCRIPT_DIR, 'emb_llm', 'Qwen3-Embedding-8B')
DEFAULT_OPENAI_BASE_URL = 'https://yunwu.ai/v1'

# ✅ (核心修改) 导入增强后的 utils.py 中的函数
try:
    # 添加父目录 (preprocessing/) 到 Python 路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import (
        load_json, 
        get_id2item_dict, 
        build_text_map, 
        find_first_image_path, 
        load_pil_image, 
        build_output_path, 
        apply_pca_and_save, 
        set_device,
        clean_text # 如果 encoder 模块需要
    )
    print("[INFO] 成功从父目录 utils.py 导入共享函数。")
except ImportError as e:
    print(f"导入错误: {e}")
    print("错误: 无法从父目录 (preprocessing/) 导入 utils.py。请检查文件结构和 Python 路径。")
    sys.exit(1)

# 添加路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generate_embedding'))

# ✅ (核心修改) 从当前目录导入各个 encoder 模块
try:
    import text_encoder 
    import image_encoder
    import cf_encoder
    import vlm_encoder
    print("[INFO] 成功导入 encoder 模块。")
except ImportError as e:
    print(f"导入错误: {e}")
    print("错误: 无法导入当前目录下的 encoder 模块。请确保 __init__.py 文件存在且 encoder 文件名正确。")
    sys.exit(1)

# 🚨 (移除) 不再需要 common_utils.py

# =============== 数据加载 (现在使用 utils.py 函数) ===============
def load_common_data(args):
    """加载所有 encoder 都需要的基础数据 (使用导入的 utils 函数)"""
    print(f"\n--- 加载通用数据 ({args.dataset}) ---")
    data_dir = os.path.join(args.save_root, args.dataset) # save_root 指向 ../datasets
    item2id_path = os.path.join(data_dir, f'{args.dataset}.item2id')
    item_meta_path = os.path.join(data_dir, f'{args.dataset}.item.json')
    
    print(f"加载 item2id: {item2id_path}")
    id2item = get_id2item_dict(item2id_path) # 使用 utils.get_id2item_dict
    
    print(f"加载 item meta: {item_meta_path}")
    item_meta = load_json(item_meta_path) # 使用 utils.load_json
    if not item_meta: raise FileNotFoundError(f"item.json 文件加载失败或为空: {item_meta_path}")

    images_info = None
    image_dir = None # 初始化
    if args.embedding_type in ['image_clip', 'vlm_fused']: 
         # ✅ (修改) 图像路径构建更健壮
         # image_root 指向 ../datasets/amazonXX/Images/
         image_base_path = os.path.join(args.image_root, f"amazon{args.data_version}", "Images")
         if not os.path.isdir(image_base_path): # 检查基础路径是否存在
              print(f"[WARN] 图像基础路径不存在: {image_base_path}。如果您不需要图像，请忽略。")
         else:
              images_info_path = os.path.join(image_base_path, f"{args.dataset}_images_info.json")
              image_dir = os.path.join(image_base_path, args.dataset) # 图片文件夹
              print(f"加载 image info: {images_info_path}")
              images_info = load_json(images_info_path) or {}
              if not images_info: print(f"[WARN] 未找到或加载 image info 文件失败: {images_info_path}")
              if not os.path.isdir(image_dir):
                  print(f"[WARN] 图像目录不存在: {image_dir}。将无法加载图像。")
                  image_dir = None # 设为 None
         
    print("通用数据加载完毕。")
    # 返回 image_dir 以便后续使用
    return id2item, item_meta, images_info, image_dir 

# =============== 主程序 (调度逻辑不变，调用导入的函数) ===============
def main():
    parser = argparse.ArgumentParser(description="统一的 Embedding 生成脚本")

    # --- 核心调度参数 ---
    parser.add_argument('--embedding_type', type=str, required=True, 
                        choices=['text_local', 'text_api', 'image_clip', 'cf_sasrec', 'vlm_fused'],
                        help='要生成的 Embedding 类型')

    # --- 数据参数 ---
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称')
    parser.add_argument('--dataset_type', type=str, default='amazon', choices=['amazon', 'movielens'], help='数据集类型')
    parser.add_argument('--data_version', type=str, default='14', choices=['14','18'], help='Amazon 版本')
    parser.add_argument('--save_root', type=str, default='../datasets', help='保存预处理数据的根目录')
    parser.add_argument('--text_format', type=str, default='rpg_sentence', choices=['rpg_sentence', 'unigenrec'], help='[text/vlm] item 文本构造格式')
    # ✅ (修改) image_root 指向 amazonXX/Images/ 或类似目录
    parser.add_argument('--image_root', type=str, default='../datasets', help='包含图像信息文件和图像文件夹的根目录 (e.g., ../datasets)') 

    # --- 模型参数 ---
    # (保持不变)
    parser.add_argument('--model_name_or_path', type=str, default=DEFAULT_LOCAL_TEXT_MODEL, help='[text_local] 本地 Transformer 模型')
    parser.add_argument('--max_sent_len', type=int, default=1024, help='[text_local] 文本最大长度')
    parser.add_argument('--sent_emb_model', type=str, default='text-embedding-3-large', help='[text_api] OpenAI 模型 ID')
    parser.add_argument('--api_emb_dim', type=int, default=0, help='[text_api] API 输出维度 (0 为自动)')
    parser.add_argument('--openai_api_key', type=str, default=os.environ.get('OPENAI_API_KEY'), help='OpenAI Key')
    parser.add_argument('--openai_base_url', type=str, default=os.environ.get('OPENAI_BASE_URL', DEFAULT_OPENAI_BASE_URL), help='OpenAI Base URL')
    parser.add_argument('--clip_model_name', type=str, default='openai/clip-vit-base-patch32', help='[image_clip] CLIP 模型 ID')
    parser.add_argument('--sasrec_hidden_dim', type=int, default=64, help='[cf_sasrec] 隐藏维度')
    parser.add_argument('--sasrec_max_seq_len', type=int, default=50, help='[cf_sasrec] 序列长度')
    parser.add_argument('--sasrec_n_layers', type=int, default=2, help='[cf_sasrec] 层数')
    parser.add_argument('--sasrec_n_heads', type=int, default=2, help='[cf_sasrec] 头数')
    parser.add_argument('--sasrec_dropout', type=float, default=0.2, help='[cf_sasrec] Dropout')
    parser.add_argument('--sasrec_epochs', type=int, default=30, help='[cf_sasrec] 训练轮数')
    parser.add_argument('--sasrec_lr', type=float, default=0.001, help='[cf_sasrec] 学习率')
    parser.add_argument('--sasrec_weight_decay', type=float, default=0.0, help='[cf_sasrec] 权重衰减')
    parser.add_argument('--vlm_model_name_or_path', type=str, default='Qwen/Qwen3-VL-7B-Instruct', help='[vlm_fused] VLM 模型 ID')
    parser.add_argument('--vlm_prompt_template', type=str, default="Represent this item for recommendation: {}", help='[vlm_fused] Prompt 模板')

    # --- 通用 ---
    parser.add_argument('--model_cache_dir', type=str, default=None, help='Hugging Face 缓存目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--pca_dim', type=int, default=512, help='PCA 目标维度 (<=0 不降维)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID (<0 使用 CPU)')
    
    args = parser.parse_args()
    args.device = set_device(args.gpu_id) # 使用 utils.set_device

    # --- 1. 加载通用数据 ---
    try:
        id2item, item_meta, images_info, image_dir = load_common_data(args) 
        text_map = {}
        if args.embedding_type in ['text_local', 'text_api', 'vlm_fused']:
             text_map = build_text_map(args, id2item, item_meta) # 使用 utils.build_text_map
             if not text_map: print("[WARN] text_map 为空。")

    except FileNotFoundError as e: print(f"错误：加载基础数据失败: {e}"); sys.exit(1)
    except Exception as e: print(f"错误：加载数据时发生未知错误: {e}"); sys.exit(1)


    # --- 2. 调度 Embedding 生成 ---
    embeddings = None
    modality_tag = "" 
    model_tag = ""    
    
    print(f"\n--- 开始生成 Embedding ({args.embedding_type}) ---")

    try:
        if args.embedding_type == 'text_local':
            item_text_list = []
            # 确保按新 ID 顺序生成
            sorted_new_ids = sorted(id2item.keys(), key=int)
            for new_id_str in sorted_new_ids:
                 orig_id = id2item[new_id_str]
                 # encoder 需要 int 类型的 ID
                 item_text_list.append([int(new_id_str), text_map.get(orig_id, "N/A")])
                 
            embeddings = text_encoder.generate_local_text(args, item_text_list)
            modality_tag = "text"
            model_tag = args.model_name_or_path

        elif args.embedding_type == 'text_api':
            item_text_list = []
            sorted_new_ids = sorted(id2item.keys(), key=int)
            for new_id_str in sorted_new_ids:
                 orig_id = id2item[new_id_str]
                 item_text_list.append([int(new_id_str), text_map.get(orig_id, "N/A")])
                 
            embeddings = text_encoder.generate_api_text(args, item_text_list)
            modality_tag = "text"
            model_tag = args.sent_emb_model

        elif args.embedding_type == 'image_clip':
            if not image_dir: raise ValueError("图像目录未找到或无效，无法生成图像嵌入。")
            embeddings = image_encoder.generate_clip_image(args, id2item, images_info, image_dir)
            modality_tag = "image"
            model_tag = args.clip_model_name

        elif args.embedding_type == 'cf_sasrec':
            embeddings = cf_encoder.train_and_extract_sasrec(args, len(id2item))
            modality_tag = "cf"
            model_tag = "sasrec"

        elif args.embedding_type == 'vlm_fused':
            if not image_dir: raise ValueError("图像目录未找到或无效，无法生成 VLM 融合嵌入。")
            embeddings = vlm_encoder.generate_vlm_fused(args, id2item, text_map, images_info, image_dir)
            modality_tag = "vlm-fused"
            model_tag = args.vlm_model_name_or_path

        else:
            raise ValueError(f"错误：未知的 embedding_type: {args.embedding_type}")

    except FileNotFoundError as e: print(f"错误：生成嵌入时缺少文件: {e}"); sys.exit(1)
    except ValueError as e: print(f"错误：生成嵌入时参数错误: {e}"); sys.exit(1)
    except Exception as e: print(f"错误：生成嵌入时发生未知错误: {e}"); sys.exit(1)

    # --- 3. 验证和保存 ---
    if embeddings is None or not isinstance(embeddings, np.ndarray) or embeddings.size == 0:
         print(f"错误：未能生成有效的 Embedding！")
         sys.exit(1)
         
    if embeddings.shape[0] != len(id2item):
         print(f"错误：生成的 Embedding 数量 ({embeddings.shape[0]}) 与 item 数量 ({len(id2item)}) 不匹配！")
         # 尝试修复（填充或截断） - 这里选择截断
         print("将尝试截断 Embedding 以匹配 item 数量...")
         embeddings = embeddings[:len(id2item)]
         if embeddings.shape[0] != len(id2item): # 如果截断后仍不匹配（例如生成数量为0）
              print("错误：修复后数量仍不匹配，程序终止。")
              sys.exit(1)
              
    # 构建输出路径 (使用 utils.build_output_path)
    output_path = build_output_path(args, modality_tag, model_tag)
    
    # 应用 PCA 并保存 (使用 utils.apply_pca_and_save)
    final_output_path = apply_pca_and_save(embeddings, args, output_path)

    if final_output_path: # 确保路径有效
        print(f"\n🎉 任务完成！最终 Embedding 已保存至: {final_output_path}")
    else:
        print("\n❌ 任务失败：未能成功保存 Embedding。")

if __name__ == '__main__':
    main()
