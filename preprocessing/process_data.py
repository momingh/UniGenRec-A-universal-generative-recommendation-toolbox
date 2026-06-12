import argparse
import collections
import gzip
import html
import json
import os
import re
from tqdm import tqdm
import ast  # 用于 Amazon 2014


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def clean_text(raw_text):
    if isinstance(raw_text, list):
        new_raw_text = []
        for raw in raw_text:
            raw = html.unescape(str(raw))
            raw = re.sub(r'</?\w+[^>]*>', '', raw)
            raw = re.sub(r'["\n\r]*', '', raw)
            new_raw_text.append(raw.strip())
        cleaned_text = ' '.join(new_raw_text)
    else:
        if isinstance(raw_text, dict):
            cleaned_text = str(raw_text)[1:-1].strip()
        else:
            cleaned_text = str(raw_text or '').strip()
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


def write_json_file(dic, file):
    print('Writing json file: ', file)
    with open(file, 'w', encoding='utf-8') as fp:
        json.dump(dic, fp, indent=4)


def write_remap_index(unit2index, file):
    print('Writing remap file: ', file)
    with open(file, 'w', encoding='utf-8') as fp:
        for unit in unit2index:
            fp.write(unit + '\t' + str(unit2index[unit]) + '\n')

# --- Amazon 2014 特定的元数据加载逻辑 ---
def _flatten_amazon_categories(raw_categories):
    values = []

    def collect(node):
        if isinstance(node, list):
            for item in node:
                collect(item)
            return
        if node is None:
            return

        text = clean_text(str(node))
        if text:
            values.append(text)

    collect(raw_categories)
    return ",".join(values)


def _normalize_amazon_brand(raw_brand):
    if not isinstance(raw_brand, str):
        return clean_text(raw_brand)
    return clean_text(raw_brand.replace("by\n", " ").strip())


def _build_review_match_key(user_id, item_id, rating, timestamp):
    return (str(user_id), str(item_id), float(rating), int(timestamp))


def load_k_core_review_details(rating_inters, review_file_path):
    """
    为 K-core 过滤后保留的交互加载 review 明细。
    匹配键: reviewerID, asin, overall, unixReviewTime
    """
    if not review_file_path or not os.path.exists(review_file_path):
        print(f"[WARN] Review file not found: {review_file_path}")
        return {}

    retained_keys = {
        _build_review_match_key(user, item, rating, timestamp)
        for user, item, rating, timestamp in rating_inters
    }
    retained_items = {item for _, item, _, _ in rating_inters}
    review_details = {}

    matched_keys = set()
    matched_reviews = 0

    with gzip.open(review_file_path, "rt", encoding="utf-8") as fp:
        for line in tqdm(fp, desc="Attach K-core review details"):
            try:
                data = json.loads(line)
                item_id = data.get("asin")
                if item_id not in retained_items:
                    continue

                review_key = _build_review_match_key(
                    data.get("reviewerID"),
                    item_id,
                    data.get("overall"),
                    data.get("unixReviewTime"),
                )
                if review_key not in retained_keys or review_key in matched_keys:
                    continue

                review_record = {
                    "asin": item_id,
                    "reviewerID": data.get("reviewerID"),
                    "helpful": data.get("helpful"),
                    "overall": data.get("overall"),
                    "reviewText": data.get("reviewText"),
                    "reviewTime": data.get("reviewTime"),
                    "reviewerName": data.get("reviewerName"),
                    "summary": data.get("summary"),
                    "unixReviewTime": data.get("unixReviewTime"),
                }

                review_details[review_key] = review_record
                matched_keys.add(review_key)
                matched_reviews += 1
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue

    missing_reviews = len(retained_keys - matched_keys)
    print(f"K-core review details matched: {matched_reviews}/{len(retained_keys)}")
    if missing_reviews:
        print(f"[WARN] Missing review details for {missing_reviews} retained interactions.")

    return review_details


def load_meta_items_amazon(file_path):
    """加载 Amazon 2014 元数据文件。"""
    items = {}

    with gzip.open(file_path, "rt", encoding='utf-8') as fp:
        for line in tqdm(fp, desc="Load Amazon metas"):
            try:
                data = ast.literal_eval(line)
                item_id = data.get("asin")
                if not item_id:
                    continue

                item_meta = dict(data)

                item_meta['title_text'] = clean_text(data.get('title', ''))
                item_meta['price_text'] = clean_text(data.get('price', ''))
                item_meta['brand_text'] = _normalize_amazon_brand(data.get('brand', ''))
                item_meta['feature_text'] = clean_text(data.get('feature', ''))
                item_meta['categories_text'] = _flatten_amazon_categories(data.get('categories', []))
                item_meta['description_text'] = clean_text(data.get('description', ''))

                items[item_id] = item_meta

            except (ValueError, SyntaxError, TypeError, KeyError):
                continue

    return items


def write_review_file(args, rating_inters, review_details, user2index, item2index):
    """
    将 reviewText 及相关 review 字段单独保存，并与 user/item 交互对齐。
    """
    output_dir = os.path.join(args.output_path, args.dataset)
    review_path = os.path.join(output_dir, f'{args.dataset}.review.jsonl')

    print(f'Writing review file: {review_path}')
    with open(review_path, 'w', encoding='utf-8') as fp:
        for user, item, rating, timestamp in rating_inters:
            review_key = _build_review_match_key(user, item, rating, timestamp)
            review_info = review_details.get(review_key, {})

            record = {
                "user": str(user2index[user]),
                "item": str(item2index[item]),
                "user_raw": user,
                "item_raw": item,
                "rating": float(rating),
                "timestamp": int(timestamp),
                "reviewText": review_info.get("reviewText", ""),
                "summary": review_info.get("summary", ""),
                "helpful": review_info.get("helpful"),
                "reviewTime": review_info.get("reviewTime"),
                "reviewerName": review_info.get("reviewerName"),
            }
            json.dump(record, fp, ensure_ascii=False)
            fp.write("\n")

def preprocess_amazon(args):
    """
    处理 Amazon 2014 数据集。
    """
    print('Process Amazon rating data: ')
    print(' Dataset: ', args.dataset)

    input_root_path = os.path.join(args.input_path, 'amazon14')

    # (已移除 Amazon 脚本中未使用的 images_info 加载)

    # 动态构造 ratings 文件路径
    rating_file_path = os.path.join(input_root_path, 'Ratings', f'{args.dataset}.csv')
    if not os.path.exists(rating_file_path):
        raise FileNotFoundError(f"Ratings file not found: {rating_file_path}")

    # 调用通用的 load_ratings
    _, _, rating_inters = load_ratings(rating_file_path)

    # 动态构造 meta 文件路径
    meta_file_path = os.path.join(input_root_path, 'Metadata', f'meta_{args.dataset}.json.gz')
    if not os.path.exists(meta_file_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_file_path}")

    review_file_path = os.path.join(input_root_path, 'Review', f'{args.dataset}_5.json.gz')

    # 调用 Amazon 专属的 load_meta_items_amazon
    meta_items = load_meta_items_amazon(meta_file_path)

    print('The number of raw inters: ', len(rating_inters))
    rating_inters = make_inters_in_order(rating_inters)

    # 过滤掉没有元数据的交互
    filtered_inters = []
    for inter in tqdm(rating_inters, desc="Filtering interactions by meta items"):
        if inter[1] in meta_items:
            filtered_inters.append(inter)
    rating_inters = filtered_inters
    print(f"Interactions after meta filtering: {len(rating_inters)}")

    # K-core 过滤
    rating_inters = filter_inters(rating_inters, can_items=None,
                                  user_k_core_threshold=args.user_k,
                                  item_k_core_threshold=args.item_k)

    rating_inters = make_inters_in_order(rating_inters)
    review_details = load_k_core_review_details(rating_inters, review_file_path)
    print('\n')
    return rating_inters, meta_items, review_details

# =================================================================
# ============ 以下是两个脚本完全共享的通用函数 ============
# =================================================================

def load_ratings(file):
    """
    (通用) 加载 .csv 格式的评分数据
    格式: item, user, rating, time
    """
    users, items, seen_inters, inters = set(), set(), set(), []
    with open(file, 'r') as fp:
        for line in tqdm(fp, desc='Load ratings'):
            try:
                item, user, rating, time = line.strip().split(',')
                inter = (user, item, float(rating), int(time))
                if inter in seen_inters:
                    continue
                users.add(user)
                items.add(item)
                seen_inters.add(inter)
                inters.append(inter)
            except ValueError:
                continue
    return users, items, inters

def get_user2count(inters):
    user2count = collections.defaultdict(int)
    for unit in inters:
        user2count[unit[0]] += 1
    return user2count

def get_item2count(inters):
    item2count = collections.defaultdict(int)
    for unit in inters:
        item2count[unit[1]] += 1
    return item2count

def generate_candidates(unit2count, threshold):
    cans = set()
    for unit, count in unit2count.items():
        if count >= threshold:
            cans.add(unit)
    return cans, len(unit2count) - len(cans)

def filter_inters(inters, can_items=None,
                  user_k_core_threshold=0, item_k_core_threshold=0):
    """(通用) K-core 过滤器"""
    new_inters = []
    # 注意：can_items 逻辑在特定于数据集的预处理函数中执行了
    if can_items:
        print('\nFiltering by meta items (Deprecated in unified script): ')
        for unit in tqdm(inters):
            if unit[1] in can_items.keys():
                new_inters.append(unit)
        inters, new_inters = new_inters, []
        print('    The number of inters: ', len(inters))

    if user_k_core_threshold or item_k_core_threshold:
        print('\nFiltering by k-core:')
        idx = 0
        user2count = get_user2count(inters)
        item2count = get_item2count(inters)
        while True:
            new_user2count = collections.defaultdict(int)
            new_item2count = collections.defaultdict(int)
            users, n_filtered_users = generate_candidates(
                user2count, user_k_core_threshold)
            items, n_filtered_items = generate_candidates(
                item2count, item_k_core_threshold)
            if n_filtered_users == 0 and n_filtered_items == 0:
                break
            for unit in inters:
                if unit[0] in users and unit[1] in items:
                    new_inters.append(unit)
                    new_user2count[unit[0]] += 1
                    new_item2count[unit[1]] += 1
            idx += 1
            inters, new_inters = new_inters, []
            user2count, item2count = new_user2count, new_item2count
            print('    Epoch %d The number of inters: %d, users: %d, items: %d'
                    % (idx, len(inters), len(user2count), len(item2count)))
    return inters

def make_inters_in_order(inters):
    """(通用) 按用户和时间戳排序交互。

    Python sort is stable, so equal-timestamp interactions keep the order from
    load_ratings. This preserves the raw review file order as the tie-breaker.
    """
    user2inters, new_inters = collections.defaultdict(list), list()
    for inter in tqdm(inters):
        user, item, rating, timestamp = inter
        user2inters[user].append((user, item, rating, timestamp))
    for user in tqdm(user2inters):
        user_inters = user2inters[user]
        user_inters.sort(key=lambda d: d[3])
        interacted_item = set()
        for inter in user_inters:
            if inter[1] in interacted_item:
                continue
            interacted_item.add(inter[1])
            new_inters.append(inter)
    return new_inters

def convert_inters2dict(inters):
    """
    (通用) 将原始交互映射为 user2interactions, user2index, item2index。
    """
    all_users = {u for (u, i, r, t) in inters}
    all_items = {i for (u, i, r, t) in inters}

    users_sorted = sorted(all_users)
    items_sorted = sorted(all_items)
    user2index = {u: idx for idx, u in enumerate(users_sorted)}
    item2index = {i: idx for idx, i in enumerate(items_sorted)}

    user2interactions = collections.defaultdict(list)
    for u, it, r, ts in inters:
        uid = user2index[u]
        iid = item2index[it]
        user2interactions[uid].append({
            "item": iid,
            "timestamp": int(ts),
        })

    return user2interactions, user2index, item2index

def parse_args():
    """(通用) 统一的参数解析器"""
    parser = argparse.ArgumentParser()

    # 新增：用于分发任务的参数
    parser.add_argument('--dataset_type', type=str, required=True, choices=['amazon'],
                        help='Type of the dataset to process. Only amazon is supported.')

    # 通用参数
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., Home, Baby)')
    parser.add_argument('--user_k', type=int, default=5, help='user k-core filtering')
    parser.add_argument('--item_k', type=int, default=5, help='item k-core filtering')
    parser.add_argument('--input_path', type=str, default='../datasets',
                        help='Root path containing dataset folders (e.g., amazon14/)')
    parser.add_argument('--output_path', type=str, default='../datasets',
                        help='Root path to save processed data')

    return parser.parse_args()

# =================================================================
# =================== 主程序入口 (Main) ===================
# =================================================================

if __name__ == '__main__':
    args = parse_args()

    print('\n' + '=' * 20)
    print(f"Start processing dataset: {args.dataset} (Type: {args.dataset_type})")
    print('=' * 20 + '\n')

    # --- 1. Amazon 数据预处理 ---
    rating_inters, meta_items, review_details = preprocess_amazon(args)

    # --- 2. 重映射并保存完整用户交互序列 ---
    # Train/valid/test splitting belongs to downstream training modules.
    all_inters, user2index, item2index = convert_inters2dict(rating_inters)

    # --- 3. 执行通用的文件保存 ---
    output_dataset_path = os.path.join(args.output_path, args.dataset)
    check_path(output_dataset_path)

    # 保存 .inter.json
    write_json_file(all_inters, os.path.join(output_dataset_path, f'{args.dataset}.inter.json'))

    # 准备并保存 .item.json
    item2feature = collections.defaultdict(dict)
    for item_str, item_id_int in item2index.items():
        # 确保来自交互的 item 存在于元数据中
        if item_str in meta_items:
            item2feature[item_id_int] = meta_items[item_str]

    print("Total users:", len(user2index))
    print("Total items (with meta):", len(item2feature))
    print("Total items (in inters):", len(item2index))
    print("Total interactions:", len(rating_inters))
    write_json_file(item2feature, os.path.join(output_dataset_path, f'{args.dataset}.item.json'))
    write_review_file(args, rating_inters, review_details, user2index, item2index)

    # 保存映射文件
    write_remap_index(user2index, os.path.join(output_dataset_path, f'{args.dataset}.user2id'))
    write_remap_index(item2index, os.path.join(output_dataset_path, f'{args.dataset}.item2id'))

    print(f"\nFinished processing dataset: {args.dataset}")
