import argparse
from pathlib import Path

from data import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Load graph recommendation splits")
    parser.add_argument("--dataset", type=str, default="Beauty")
    parser.add_argument("--data_root", type=Path, default=PROJECT_ROOT / "datasets")
    args = parser.parse_args()

    data = load_dataset(args.data_root, args.dataset)

    print(f"Dataset: {args.dataset}")
    print(f"Users: {data['num_users']}")
    print(f"Items: {data['num_items']}")
    print(f"Users with samples: {len(data['samples'])}")
    print(f"Train interactions from valid.history: {sum(len(x.train_items) for x in data['samples'])}")
    print(f"Valid targets from valid.target: {len(data['samples'])}")
    print(f"Test targets from test.target: {len(data['samples'])}")
    print(f"Used items: {len(data['used_item_ids'])}")
    print(f"Loaded item metadata: {len(data['item_info'])}")

    if data["samples"]:
        print(f"First sample: {data['samples'][0]}")
        first_item = data["samples"][0].train_items[0]
        print(f"First used item info: {first_item} -> {data['item_info'][first_item]}")


if __name__ == "__main__":
    main()
