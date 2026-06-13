import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import requests
from tqdm import tqdm


AMAZON_BASE_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"


def download_file(url: str, output_path: Path, description: str) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"文件已存在，跳过下载: {output_path}")
        return True

    print(f"下载 {description}: {url}")
    try:
        with requests.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            with output_path.open("wb") as fp, tqdm(
                total=total,
                unit="iB",
                unit_scale=True,
                desc=f"  -> {description}",
            ) as progress:
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    fp.write(chunk)
                    progress.update(len(chunk))
    except requests.exceptions.RequestException as exc:
        print(f"下载失败: {exc}")
        return False

    print(f"  -> 保存到: {output_path}")
    return True


def amazon_review_filename(category: str, data_version: str) -> str:
    if data_version == "14":
        return f"reviews_{category}_5.json.gz"
    return f"{category}_5.json.gz"


def extract_ratings_from_reviews(review_path: Path, ratings_path: Path) -> bool:
    ratings_path.parent.mkdir(parents=True, exist_ok=True)
    if ratings_path.exists():
        print(f"Ratings 文件已存在，跳过提取: {ratings_path}")
        return True

    print(f"从 Review 提取 Ratings: {review_path.name}")
    try:
        with gzip.open(review_path, "rt", encoding="utf-8") as src, ratings_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as dst:
            writer = csv.writer(dst)
            for line in tqdm(src, desc="  -> 提取中"):
                try:
                    review = json.loads(line)
                except json.JSONDecodeError:
                    continue

                item_id = review.get("asin")
                user_id = review.get("reviewerID")
                rating = review.get("overall")
                timestamp = review.get("unixReviewTime")
                if item_id and user_id and rating is not None and timestamp is not None:
                    writer.writerow([item_id, user_id, rating, timestamp])
    except OSError as exc:
        print(f"提取失败: {exc}")
        return False

    print(f"  -> Ratings 保存到: {ratings_path}")
    return True


def process_amazon(dataset: str, data_version: str, output_dir: str) -> bool:
    base_dir = Path(output_dir) / f"amazon{data_version}"
    metadata_dir = base_dir / "Metadata"
    review_dir = base_dir / "Review"
    ratings_dir = base_dir / "Ratings"

    meta_name = f"meta_{dataset}.json.gz"
    review_server_name = amazon_review_filename(dataset, data_version)
    review_local_name = f"{dataset}_5.json.gz"

    meta_path = metadata_dir / meta_name
    review_path = review_dir / review_local_name
    ratings_path = ratings_dir / f"{dataset}.csv"

    print("\n" + "=" * 15 + f" 处理 Amazon 数据集: {dataset} (v{data_version}) " + "=" * 15)
    print(f"数据目录: {base_dir.resolve()}")

    meta_url = f"{AMAZON_BASE_URL}/{meta_name}"
    review_url = f"{AMAZON_BASE_URL}/{review_server_name}"

    if not download_file(meta_url, meta_path, "元数据"):
        return False
    if not download_file(review_url, review_path, "评论"):
        return False
    if not extract_ratings_from_reviews(review_path, ratings_path):
        return False

    print("\nAmazon 数据下载和初步提取完成。")
    print("生成的文件:")
    for path in (meta_path, review_path, ratings_path):
        print(f"  - {path.resolve()}")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="下载 Amazon review、metadata，并提取 ratings CSV。")
    parser.add_argument("--source", default="amazon", choices=["amazon"], help="兼容旧命令，仅支持 amazon")
    parser.add_argument("--dataset", required=True, help="Amazon 类目名，例如 Beauty、Baby")
    parser.add_argument("--data_version", default="14", choices=["14", "18"], help="Amazon 数据版本")
    parser.add_argument("--output_dir", default="../datasets", help="保存数据的根目录")
    return parser.parse_args()


def main():
    args = parse_args()
    if not process_amazon(args.dataset, args.data_version, args.output_dir):
        sys.exit(1)


if __name__ == "__main__":
    main()
