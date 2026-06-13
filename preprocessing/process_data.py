import argparse
import ast
import csv
import gzip
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

try:
    from metadata_text import build_metadata_sentence
except ImportError:
    from .metadata_text import build_metadata_sentence


def clean_text(raw_text):
    if isinstance(raw_text, list):
        values = []
        for raw in raw_text:
            raw = html.unescape(str(raw))
            raw = re.sub(r"</?\w+[^>]*>", "", raw)
            raw = re.sub(r'["\n\r]*', "", raw)
            values.append(raw.strip())
        text = " ".join(values)
    else:
        text = str(raw_text)[1:-1].strip() if isinstance(raw_text, dict) else str(raw_text or "").strip()
        text = html.unescape(text)
        text = re.sub(r"</?\w+[^>]*>", "", text)
        text = re.sub(r'["\n\r]*', "", text)

    index = -1
    while -index < len(text) and text[index] == ".":
        index -= 1
    index += 1
    text = f"{text}." if index == 0 else f"{text[:index]}."
    return "" if len(text) >= 2000 else text


def write_json_file(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing json file: {path}")
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=4)


def write_remap_index(unit2index, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing remap file: {path}")
    with path.open("w", encoding="utf-8") as fp:
        for unit, index in sorted(unit2index.items(), key=lambda item: item[1]):
            fp.write(f"{unit}\t{index}\n")


def flatten_categories(raw_categories):
    values = []

    def collect(node):
        if isinstance(node, list):
            for item in node:
                collect(item)
        elif node is not None:
            text = clean_text(str(node))
            if text:
                values.append(text)

    collect(raw_categories)
    return ",".join(values)


def normalize_brand(raw_brand):
    if isinstance(raw_brand, str):
        raw_brand = raw_brand.replace("by\n", " ").strip()
    return clean_text(raw_brand)


def review_key(user_id, item_id, rating, timestamp):
    return (str(user_id), str(item_id), float(rating), int(timestamp))


def load_ratings(path):
    inters = []
    seen = set()
    with Path(path).open("r", encoding="utf-8") as fp:
        reader = csv.reader(fp)
        for row in tqdm(reader, desc="Load ratings"):
            if len(row) != 4:
                continue
            item, user, rating, timestamp = row
            try:
                inter = (user, item, float(rating), int(timestamp))
            except ValueError:
                continue
            if inter not in seen:
                seen.add(inter)
                inters.append(inter)
    return inters


def load_meta_items(path):
    items = {}
    with gzip.open(path, "rt", encoding="utf-8") as fp:
        for line in tqdm(fp, desc="Load Amazon metas"):
            try:
                data = ast.literal_eval(line)
            except (ValueError, SyntaxError, TypeError):
                continue

            item_id = data.get("asin")
            if not item_id:
                continue

            item_meta = dict(data)
            item_meta["title_text"] = clean_text(data.get("title", ""))
            item_meta["price_text"] = clean_text(data.get("price", ""))
            item_meta["brand_text"] = normalize_brand(data.get("brand", ""))
            item_meta["feature_text"] = clean_text(data.get("feature", ""))
            item_meta["categories_text"] = flatten_categories(data.get("categories", []))
            item_meta["description_text"] = clean_text(data.get("description", ""))
            item_meta["metadata_sentence"] = build_metadata_sentence(item_meta)
            items[item_id] = item_meta
    return items


def order_and_deduplicate(inters):
    user2inters = defaultdict(list)
    for inter in tqdm(inters, desc="Group interactions"):
        user2inters[inter[0]].append(inter)

    ordered = []
    for user in tqdm(user2inters, desc="Sort interactions"):
        seen_items = set()
        for inter in sorted(user2inters[user], key=lambda row: row[3]):
            item = inter[1]
            if item in seen_items:
                continue
            seen_items.add(item)
            ordered.append(inter)
    return ordered


def filter_by_k_core(inters, user_k=0, item_k=0):
    if not user_k and not item_k:
        return inters

    print("\nFiltering by k-core:")
    epoch = 0
    while True:
        user_counts = Counter(user for user, _, _, _ in inters)
        item_counts = Counter(item for _, item, _, _ in inters)
        valid_users = {user for user, count in user_counts.items() if count >= user_k}
        valid_items = {item for item, count in item_counts.items() if count >= item_k}

        filtered = [
            inter for inter in inters
            if inter[0] in valid_users and inter[1] in valid_items
        ]
        if len(filtered) == len(inters):
            break

        epoch += 1
        inters = filtered
        print(
            f"    Epoch {epoch} inters={len(inters)}, "
            f"users={len(valid_users)}, items={len(valid_items)}"
        )
    return inters


def load_review_details(rating_inters, review_path):
    review_path = Path(review_path)
    if not review_path.exists():
        print(f"[WARN] Review file not found: {review_path}")
        return {}

    retained_keys = {review_key(user, item, rating, ts) for user, item, rating, ts in rating_inters}
    retained_items = {item for _, item, _, _ in rating_inters}
    matched_keys = set()
    details = {}

    with gzip.open(review_path, "rt", encoding="utf-8") as fp:
        for line in tqdm(fp, desc="Attach review details"):
            try:
                data = json.loads(line)
                item_id = data.get("asin")
                key = review_key(data.get("reviewerID"), item_id, data.get("overall"), data.get("unixReviewTime"))
            except (ValueError, TypeError, json.JSONDecodeError):
                continue

            if item_id not in retained_items or key not in retained_keys or key in matched_keys:
                continue

            details[key] = {
                "reviewText": data.get("reviewText", ""),
                "summary": data.get("summary", ""),
                "helpful": data.get("helpful"),
                "reviewTime": data.get("reviewTime"),
                "reviewerName": data.get("reviewerName"),
            }
            matched_keys.add(key)

    print(f"Review details matched: {len(matched_keys)}/{len(retained_keys)}")
    missing = len(retained_keys - matched_keys)
    if missing:
        print(f"[WARN] Missing review details for {missing} retained interactions.")
    return details


def remap_interactions(inters):
    users = sorted({user for user, _, _, _ in inters})
    items = sorted({item for _, item, _, _ in inters})
    user2index = {user: idx for idx, user in enumerate(users)}
    item2index = {item: idx for idx, item in enumerate(items)}

    user2interactions = defaultdict(list)
    for user, item, _, timestamp in inters:
        user2interactions[user2index[user]].append({
            "item": item2index[item],
            "timestamp": int(timestamp),
        })
    return user2interactions, user2index, item2index


def write_review_file(output_dir, dataset, rating_inters, review_details, user2index, item2index):
    review_path = Path(output_dir) / f"{dataset}.review.jsonl"
    print(f"Writing review file: {review_path}")
    with review_path.open("w", encoding="utf-8") as fp:
        for user, item, rating, timestamp in rating_inters:
            info = review_details.get(review_key(user, item, rating, timestamp), {})
            record = {
                "user": str(user2index[user]),
                "item": str(item2index[item]),
                "user_raw": user,
                "item_raw": item,
                "rating": float(rating),
                "timestamp": int(timestamp),
                "reviewText": info.get("reviewText", ""),
                "summary": info.get("summary", ""),
                "helpful": info.get("helpful"),
                "reviewTime": info.get("reviewTime"),
                "reviewerName": info.get("reviewerName"),
            }
            json.dump(record, fp, ensure_ascii=False)
            fp.write("\n")


def preprocess_amazon(args):
    input_root = Path(args.input_path) / "amazon14"
    rating_path = input_root / "Ratings" / f"{args.dataset}.csv"
    meta_path = input_root / "Metadata" / f"meta_{args.dataset}.json.gz"
    review_path = input_root / "Review" / f"{args.dataset}_5.json.gz"

    if not rating_path.exists():
        raise FileNotFoundError(f"Ratings file not found: {rating_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    print(f"Process Amazon dataset: {args.dataset}")
    rating_inters = load_ratings(rating_path)
    meta_items = load_meta_items(meta_path)

    print(f"Raw interactions: {len(rating_inters)}")
    rating_inters = order_and_deduplicate(rating_inters)
    rating_inters = [inter for inter in tqdm(rating_inters, desc="Filter by metadata") if inter[1] in meta_items]
    print(f"Interactions after metadata filtering: {len(rating_inters)}")

    rating_inters = filter_by_k_core(rating_inters, user_k=args.user_k, item_k=args.item_k)
    rating_inters = order_and_deduplicate(rating_inters)
    review_details = load_review_details(rating_inters, review_path)
    return rating_inters, meta_items, review_details


def write_outputs(args, rating_inters, meta_items, review_details):
    output_dir = Path(args.output_path) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    inter_json, user2index, item2index = remap_interactions(rating_inters)
    item_json = {
        item_id: meta_items[item]
        for item, item_id in item2index.items()
        if item in meta_items
    }

    print("Total users:", len(user2index))
    print("Total items (with meta):", len(item_json))
    print("Total items (in inters):", len(item2index))
    print("Total interactions:", len(rating_inters))

    write_json_file(inter_json, output_dir / f"{args.dataset}.inter.json")
    write_json_file(item_json, output_dir / f"{args.dataset}.item.json")
    write_review_file(output_dir, args.dataset, rating_inters, review_details, user2index, item2index)
    write_remap_index(user2index, output_dir / f"{args.dataset}.user2id")
    write_remap_index(item2index, output_dir / f"{args.dataset}.item2id")


def parse_args():
    parser = argparse.ArgumentParser(description="处理 Amazon ratings、metadata 和 review 文件。")
    parser.add_argument("--dataset_type", required=True, choices=["amazon"], help="兼容旧命令，仅支持 amazon")
    parser.add_argument("--dataset", required=True, help="数据集名称，例如 Beauty")
    parser.add_argument("--user_k", type=int, default=5, help="user k-core 阈值")
    parser.add_argument("--item_k", type=int, default=5, help="item k-core 阈值")
    parser.add_argument("--input_path", default="../datasets", help="包含 amazon14/ 的根目录")
    parser.add_argument("--output_path", default="../datasets", help="输出根目录")
    return parser.parse_args()


def main():
    args = parse_args()
    rating_inters, meta_items, review_details = preprocess_amazon(args)
    write_outputs(args, rating_inters, meta_items, review_details)
    print(f"\nFinished processing dataset: {args.dataset}")


if __name__ == "__main__":
    main()
