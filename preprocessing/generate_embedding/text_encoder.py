# preprocessing/generate_embeddings/text_encoder.py

import torch
import numpy as np
from tqdm import tqdm
import time
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI 
import os # 导入 os
import sys # 导入 sys
from sentence_transformers import SentenceTransformer
# ✅ (核心修改) 从父目录导入共享函数
try:
    # 添加父目录到路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # 从 utils 导入需要的函数 (可能不需要全部导入)
    # from utils import clean_text # 如果需要内部清理
except ImportError as e:
    print(f"导入错误: {e}")
    print("错误: 无法从父目录 (preprocessing/) 导入 utils.py。")
    sys.exit(1)

# 🚨 (移除) 不再需要在这里定义 load_json, clean_text, set_device 等

def _to_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(str(token) for token in value)
    if value is None:
        return ""
    return str(value)


def _order_texts(item_text_list):
    items, texts = zip(*item_text_list)
    max_item_id = max(items) if items else -1
    order_texts = [""] * (max_item_id + 1)
    for item, text in zip(items, texts):
        order_texts[item] = _to_text(text)
    return items, order_texts


def generate_local_text(args, item_text_list) -> np.ndarray:
    """使用本地 Transformer 模型生成文本嵌入"""
    print(f"🔹 使用本地模型生成文本嵌入: {args.model_name_or_path}")
    
    # 确保 device 来自 args
    device = getattr(args, 'device', torch.device('cpu')) 
    
    # 👉 SentenceTransformer 自带 tokenizer，不能再用 HF Tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, cache_dir=args.model_cache_dir)

    # 👉 加载 SentenceTransformer（这是 encoder-only）
    model = SentenceTransformer(args.model_name_or_path, device=str(device))
    model.eval()
    
    items, final_texts = _order_texts(item_text_list)

    embeddings = []

    # ✔ SentenceTransformer 自动做 batching，不需要 HF tokenizer
    with torch.no_grad():
        for i in tqdm(range(0, len(final_texts), args.batch_size), desc="Local Text Encoding"):
            batch_texts = final_texts[i : i + args.batch_size]
            batch_texts = [t if t.strip() else "N/A" for t in batch_texts]
            try:
                # 👉 使用 SentenceTransformer 的 encode（自动 tokenizer + pooling）
                batch_emb = model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    convert_to_numpy=True,
                    normalize_embeddings=False
                )

                # 转回 torch 再 append（保持你原逻辑）
                embeddings.append(torch.tensor(batch_emb))
            except Exception as e:
                print(f"\n[警告] 本地编码批次 {i//args.batch_size} 失败: {e}")
                emb_dim = model.get_sentence_embedding_dimension()
                embeddings.append(torch.zeros((len(batch_texts), emb_dim)))

    if not embeddings:
        raise RuntimeError("未能生成任何本地文本嵌入。")
         
    embeddings = torch.cat(embeddings, dim=0).numpy().astype(np.float32)
    
    # (验证数量逻辑 — 保持不变)
    if embeddings.shape[0] != len(final_texts):
        print(f"[警告] 本地文本嵌入数量 ({embeddings.shape[0]}) 与预期 ({len(final_texts)}) 不符！")
        target_len = len(final_texts)
        current_len = embeddings.shape[0]
        emb_dim = embeddings.shape[1]
        if current_len < target_len:
            print(" -> 将用零向量填充。")
            padding = np.zeros((target_len - current_len, emb_dim), dtype=np.float32)
            embeddings = np.concatenate([embeddings, padding], axis=0)
        else:
            print(" -> 将截断多余部分。")
            embeddings = embeddings[:target_len]

    print(f"本地文本嵌入维度: {embeddings.shape}")
    return embeddings



def generate_api_text(args, item_text_list) -> np.ndarray:
    """使用 OpenAI API 生成文本嵌入"""
    print(f"🔹 使用 API 模型生成文本嵌入: {args.sent_emb_model}")
    try:
        from openai import OpenAI
    except ImportError:
        print("错误: 'openai' 库未找到。请运行: pip install openai")
        raise # 重新抛出，让主脚本知道依赖缺失

    client = OpenAI(api_key=args.openai_api_key, base_url=args.openai_base_url)

    items, final_texts = _order_texts(item_text_list)

    sent_embs = []
    api_emb_dim = args.api_emb_dim 
    if api_emb_dim <= 0: # 尝试根据模型名猜测
        if 'large' in args.sent_emb_model: api_emb_dim = 3072
        elif 'small' in args.sent_emb_model: api_emb_dim = 1536
        else: api_emb_dim = 0 # 无法猜测时保持 0
        
    print(f"[INFO] 预期/猜测的 API 维度: {api_emb_dim if api_emb_dim > 0 else '自动检测'}")

    for i in tqdm(range(0, len(final_texts), args.batch_size), desc="API Text Encoding"):
        batch = final_texts[i : i + args.batch_size]
        batch = [t if t.strip() else "N/A" for t in batch]
        
        try:
            response = client.embeddings.create(model=args.sent_emb_model, input=batch)
            batch_embeddings = [np.array(d.embedding, dtype=np.float32) for d in response.data] # 直接转 numpy
            sent_embs.extend(batch_embeddings)
            
            if api_emb_dim <= 0 and batch_embeddings:
                api_emb_dim = len(batch_embeddings[0])
                print(f"\n[INFO] 实际检测到 API 嵌入维度为: {api_emb_dim}")
                
        except Exception as e:
            batch_id = i // args.batch_size
            message = str(e)
            raise RuntimeError(
                f"API 请求批次 {batch_id} 失败，已停止生成 embedding，避免用零向量污染结果。"
                f"原始错误: {message}"
            ) from e

    if not sent_embs: # 处理完全失败
         raise RuntimeError("未能生成任何 API 文本嵌入。")

    # 尝试将 list of numpy arrays 转换为单个 numpy array
    try:
        sent_embs = np.stack(sent_embs, axis=0)
    except ValueError as e:
         print(f"错误：无法将 API 返回的嵌入堆叠成数组 ({e})。可能维度不一致。")
         # 尝试找出不一致的维度
         dims = [emb.shape for emb in sent_embs if isinstance(emb, np.ndarray)]
         print(f"检测到的维度: {set(dims)}")
         # 选择填充或报错，这里报错
         raise RuntimeError("API 返回的嵌入维度不一致。") from e
         
    args.api_emb_dim = api_emb_dim # 更新 args

    # 验证数量
    if sent_embs.shape[0] != len(final_texts):
         print(f"[警告] API 输出嵌入数量 ({sent_embs.shape[0]}) 与预期 ({len(final_texts)}) 不符！")
         # 填充或截断
         target_len = len(final_texts)
         current_len = sent_embs.shape[0]
         emb_dim = sent_embs.shape[1]
         if current_len < target_len:
              print(" -> 将用零向量填充。")
              padding = np.zeros((target_len - current_len, emb_dim), dtype=np.float32)
              sent_embs = np.concatenate([sent_embs, padding], axis=0)
         else:
              print(" -> 将截断多余部分。")
              sent_embs = sent_embs[:target_len]

    print(f"API 文本嵌入维度: {sent_embs.shape}")
    return sent_embs
