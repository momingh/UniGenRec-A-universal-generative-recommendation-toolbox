import html
import json
import os
import pickle
import re
import time

import torch
from transformers import AutoModel, AutoTokenizer
import collections
from typing import Any, Dict

try:
    from metadata_text import build_metadata_sentence
except ImportError:
    from .metadata_text import build_metadata_sentence



def get_res_batch(model_name, prompt_list, max_tokens, api_info):

    while True:
        try:
            res = openai.Completion.create(
                model=model_name,
                prompt=prompt_list,
                temperature=0.4,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            output_list = []
            for choice in res['choices']:
                output = choice['text'].strip()
                output_list.append(output)

            return output_list

        except openai.error.AuthenticationError as e:
            print(e)
            openai.api_key = api_info["api_key_list"].pop()
            time.sleep(10)
        except openai.error.RateLimitError as e:
            print(e)
            if str(e) == "You exceeded your current quota, please check your plan and billing details.":
                openai.api_key = api_info["api_key_list"].pop()
                time.sleep(10)
            else:
                print('\nopenai.error.RateLimitError\nRetrying...')
                time.sleep(10)
        except openai.error.ServiceUnavailableError as e:
            print(e)
            print('\nopenai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(10)
        except openai.error.Timeout:
            print('\nopenai.error.Timeout\nRetrying...')
            time.sleep(10)
        except openai.error.APIError as e:
            print(e)
            print('\nopenai.error.APIError\nRetrying...')
            time.sleep(10)
        except openai.error.APIConnectionError as e:
            print(e)
            print('\nopenai.error.APIConnectionError\nRetrying...')
            time.sleep(10)
        except Exception as e:
            print(e)
            return None




def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_device(gpu_id):
    if gpu_id == -1:
        return torch.device('cpu')
    else:
        return torch.device(
            'cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')

def load_plm(model_path='bert-base-uncased', kwargs=None):

    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)

    print("Load Model:", model_path)

    model = AutoModel.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    return tokenizer, model

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def clean_text(raw_text):
    if isinstance(raw_text, list):
        new_raw_text=[]
        for raw in raw_text:
            raw = html.unescape(raw)
            raw = re.sub(r'</?\w+[^>]*>', '', raw)
            raw = re.sub(r'["\n\r]*', '', raw)
            new_raw_text.append(raw.strip())
        cleaned_text = ' '.join(new_raw_text)
    else:
        if isinstance(raw_text, dict):
            cleaned_text = str(raw_text)[1:-1].strip()
        else:
            cleaned_text = raw_text.strip()
        cleaned_text = html.unescape(cleaned_text)
        cleaned_text = re.sub(r'</?\w+[^>]*>', '', cleaned_text)
        cleaned_text = re.sub(r'["\n\r]*', '', cleaned_text)
    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == '.':
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + '.'
    else:
        cleaned_text = cleaned_text[:index] + '.'
    if len(cleaned_text) >= 2000:
        cleaned_text = ''
    return cleaned_text

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def make_inters_in_order(inters):
    user2inters, new_inters = collections.defaultdict(list), list()
    for inter in inters:
        user, item, rating, timestamp = inter
        user2inters[user].append((user, item, rating, timestamp))
    for user in user2inters:
        user_inters = user2inters[user]
        user_inters.sort(key=lambda d: d[3])
        for inter in user_inters:
            new_inters.append(inter)
    return new_inters

def write_json_file(dic, file):
    print('Writing json file: ',file)
    with open(file, 'w') as fp:
        json.dump(dic, fp, indent=4)

def write_remap_index(unit2index, file):
    print('Writing remap file: ',file)
    with open(file, 'w') as fp:
        for unit in unit2index:
            fp.write(unit + '\t' + str(unit2index[unit]) + '\n')

import html
import json
import os
import pickle
import re
import time
import argparse
from pathlib import Path
import joblib
import numpy as np
from sklearn.decomposition import PCA
import torch
import collections

def check_path(path):
    """确保目录存在。"""
    Path(path).mkdir(parents=True, exist_ok=True)

def set_device(gpu_id: int) -> torch.device:
    """设置 PyTorch 设备。"""
    if gpu_id < 0:
        print("[INFO] Using CPU.")
        return torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"[INFO] Using CUDA device: {torch.cuda.get_device_name(device)}")
        return device
    else:
        print("[WARN] CUDA not available, falling back to CPU.")
        return torch.device('cpu')

def load_json(file: str) -> Any:
    """更健壮地加载 JSON 文件。"""
    if not os.path.exists(file):
        print(f"[WARN] JSON file not found: {file}")
        return None
    try:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        print(f"[ERROR] Failed to decode JSON file: {file}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to load JSON file {file}: {e}")
        return None

def clean_text(raw_text: Any) -> str:
    """清理文本中的 HTML 标签、换行符等。"""
    text_to_clean = ""
    if isinstance(raw_text, list):
        text_to_clean = ' '.join(str(item) for item in raw_text)
    elif isinstance(raw_text, dict):
        text_to_clean = ", ".join(f"{k}: {v}" for k, v in raw_text.items())
    elif isinstance(raw_text, (str, int, float)):
         text_to_clean = str(raw_text)
    else:
         return ""

    try:
        cleaned_text = html.unescape(text_to_clean)
        cleaned_text = re.sub(r'<[^>]+>', '', cleaned_text)
        cleaned_text = re.sub(r'[\n\r]+', ' ', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = cleaned_text.replace('"', '').strip()
    except Exception as e:
        cleaned_text = ""

    return cleaned_text

def load_pickle(filename: str) -> Any:
    """加载 Pickle 文件。"""
    if not os.path.exists(filename):
         print(f"[WARN] Pickle file not found: {filename}")
         return None
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as e:
         print(f"[ERROR] Failed to load pickle file {filename}: {e}")
         return None

def write_json_file(dic: dict, file: str):
    """写入 JSON 文件。"""
    print(f'Writing json file: {file}')
    try:
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, 'w', encoding='utf-8') as fp:
            json.dump(dic, fp, indent=4, ensure_ascii=False)
    except Exception as e:
         print(f"[ERROR] Failed to write JSON file {file}: {e}")

def write_remap_index(unit2index: dict, file: str):
    """写入 remap 文件。"""
    print(f'Writing remap file: {file}')
    try:
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, 'w', encoding='utf-8') as fp:
            for unit, index in sorted(unit2index.items(), key=lambda item: int(item[1])):
                fp.write(f"{unit}\t{index}\n")
    except Exception as e:
        print(f"[ERROR] Failed to write remap file {file}: {e}")

def get_id2item_dict(item2id_file: str) -> Dict[str, str]:
    """从 `.item2id` 文件加载新 ID 到原始 ID 的映射。"""
    if not os.path.exists(item2id_file):
        raise FileNotFoundError(f"item2id 文件未找到: {item2id_file}")
    id2item = {}
    try:
        with open(item2id_file, "r", encoding='utf-8') as fp:
            for line_num, line in enumerate(fp):
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    item, item_id = parts
                    id2item[item_id] = item
                elif line.strip():
                     print(f"[WARN] item2id 文件第 {line_num+1} 行格式错误: '{line.strip()}'")
        if not id2item:
            raise RuntimeError(f"未能从 {item2id_file} 加载任何 ID 映射。")
    except Exception as e:
        print(f"[ERROR] 读取 item2id 文件失败 ({item2id_file}): {e}")
        raise
    return id2item

def load_review_interaction_texts(
    review_file: str,
    fields=("summary", "reviewText"),
    max_chars_per_review: int = 2000,
):
    if not os.path.exists(review_file):
        print(f"[WARN] review 文件不存在: {review_file}")
        return [], []

    review_texts = []
    index_records = []

    with open(review_file, "r", encoding="utf-8") as fp:
        for line_num, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] review 文件第 {line_num} 行 JSON 解析失败，已跳过。")
                continue

            parts = []
            for field in fields:
                value = record.get(field)
                if value not in (None, "", [], {}):
                    value = clean_text(str(value))
                    if value:
                        parts.append(value)
            text = " ".join(parts).strip()
            if not text:
                text = "N/A"
            if max_chars_per_review > 0:
                text = text[:max_chars_per_review]

            row_id = len(review_texts)
            review_texts.append(text)
            index_records.append({
                "row": row_id,
                "user": str(record.get("user", "")),
                "item": str(record.get("item", "")),
                "user_raw": record.get("user_raw"),
                "item_raw": record.get("item_raw"),
                "rating": record.get("rating"),
                "timestamp": record.get("timestamp"),
            })

    print(f"加载 user-item review 文本: rows={len(review_texts)}, file={review_file}")
    return review_texts, index_records


def build_text_map(args: argparse.Namespace, id2item: Dict[str, str], item_meta: Dict[str, Dict]) -> Dict[str, str]:
    """根据 `dataset_type` 从 `item_meta` 构建文本映射。"""
    if not item_meta:
         print("[WARN] item_meta 为空，无法构建 text_map。")
         return {}

    if args.dataset_type == "amazon":
        print("将使用 UniGenRec metadata sentence 标准格式构建 item 文本。")
        text_map = {}
        missing_meta_count = 0

        for new_id_str, orig_id in id2item.items():
            meta_data = item_meta.get(new_id_str)
            if not meta_data:
                missing_meta_count += 1
                text_map[orig_id] = "N/A"
                continue

            text = build_metadata_sentence(meta_data)
            if not isinstance(text, str):
                text = str(text)
            text_map[orig_id] = text if text.strip() else "N/A"

        if missing_meta_count > 0:
            print(f"[WARN] {missing_meta_count} 个 item 在 item.json 中缺少元数据。")
        return text_map

    features = []
    if args.dataset_type == 'movielens':
        features = [('title',), ('description',), ('genres',), ('year',)]
    else:
        print(f"[WARN] 未知的 dataset_type: {args.dataset_type}，将尝试使用通用字段 ['title', 'description']")
        features = [('title',), ('description',)]

    feature_names = [" / ".join(group) for group in features]
    print(f"将使用以下元数据字段构建文本: {feature_names}")
    text_map = {}
    missing_meta_count = 0

    for new_id_str, orig_id in id2item.items():
        meta_data = item_meta.get(new_id_str)
        if not meta_data:
             missing_meta_count += 1
             text_map[orig_id] = "N/A"
             continue

        parts = []
        for field_group in features:
            val = None
            for field_name in field_group:
                candidate = meta_data.get(field_name)
                if candidate not in (None, "", [], {}):
                    val = candidate
                    break

            if val is None:
                continue

            if isinstance(val, list):
                val = ", ".join(clean_text(str(x)) for x in val if str(x).strip())
            else:
                val = clean_text(str(val))

            if val:
                parts.append(val)

        text = " ".join(parts).strip()
        text_map[orig_id] = text if text else "N/A"

    if missing_meta_count > 0:
         print(f"[WARN] {missing_meta_count} 个 item 在 item.json 中缺少元数据。")

    return text_map

def find_first_image_path(original_item_id: str, images_info: Dict, image_dir: str):
    """为给定 item 查找第一个存在且非空的图像文件路径。"""
    if not images_info or not image_dir:
        return None

    names = images_info.get(original_item_id, [])
    if not isinstance(names, list):
        names = []

    for name in names:
        if not isinstance(name, str) or not name:
            continue
        fp = os.path.join(image_dir, name)
        if os.path.exists(fp) and os.path.getsize(fp) > 0:
            return fp
    return None

def load_pil_image(img_path: str):
    """安全加载 PIL 图像，失败时返回 None。"""
    if img_path is None:
        return None
    try:
        from PIL import Image, UnidentifiedImageError

        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except (UnidentifiedImageError, FileNotFoundError, OSError, Exception):
        return None

def build_output_path(args: argparse.Namespace, modality_tag: str, model_tag: str) -> str:
    """构建标准化的 embedding 输出路径。"""
    emb_dir = os.path.join(args.save_root, args.dataset, "embeddings")
    check_path(emb_dir)

    safe_model_tag = model_tag.split('/')[-1].replace('/', '-').replace('\\', '-')

    filename = f"{args.dataset}.emb-{modality_tag}-{safe_model_tag}.npy"
    return os.path.join(emb_dir, filename)

def apply_pca_and_save(original_embeddings: np.ndarray, args: argparse.Namespace, output_path: str) -> str:
    """对 embeddings 应用 PCA 并保存。"""
    pca_dim = getattr(args, 'pca_dim', 0)

    if not isinstance(original_embeddings, np.ndarray) or original_embeddings.size == 0:
        return ""

    try:
        np.save(output_path, original_embeddings)
        print(f"Saved embeddings: {output_path} {original_embeddings.shape}")

        if pca_dim <= 0 or original_embeddings.shape[1] <= pca_dim:
            return output_path

        pca = PCA(n_components=pca_dim, whiten=True, random_state=42)
        reduced_emb = pca.fit_transform(original_embeddings).astype(np.float32)

        base = Path(output_path).with_suffix("")
        pca_vec_path_str = f"{base}-pca{pca_dim}.npy"
        np.save(pca_vec_path_str, reduced_emb)
        print(f"Saved PCA embeddings: {pca_vec_path_str} {reduced_emb.shape}")
        return pca_vec_path_str

    except Exception as e:
        print(f"[ERROR] Failed to save embeddings: {e}")
        return ""
