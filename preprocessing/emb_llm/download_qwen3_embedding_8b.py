#!/usr/bin/env python3
"""Download Qwen3-Embedding-8B from Hugging Face."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_MODEL_ID = "Qwen/Qwen3-Embedding-8B"


def parse_args() -> argparse.Namespace:
    default_local_dir = Path(__file__).resolve().parent / "Qwen3-Embedding-8B"

    parser = argparse.ArgumentParser(description="Download Qwen3-Embedding-8B.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model repo id.")
    parser.add_argument("--local-dir", type=Path, default=default_local_dir, help="Directory to save model files.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--revision", default=None, help="Optional branch, tag, or commit hash.")
    parser.add_argument("--endpoint", default="https://hf-mirror.com", help="HF endpoint mirror. Use '' for default.")
    parser.add_argument("--token", default=None, help="Hugging Face token. Defaults to HF_TOKEN env var.")
    parser.add_argument("--max-workers", type=int, default=8, help="Concurrent download workers.")
    parser.add_argument("--exclude", nargs="*", default=None, help="Optional glob patterns to exclude.")
    parser.add_argument("--include", nargs="*", default=None, help="Optional glob patterns to include.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Missing dependency: huggingface_hub")
        print("Install with: pip install huggingface_hub")
        sys.exit(1)

    args.local_dir.mkdir(parents=True, exist_ok=True)

    print(f"model_id: {args.model_id}")
    print(f"local_dir: {args.local_dir}")
    if args.endpoint:
        print(f"HF_ENDPOINT: {args.endpoint}")

    downloaded_path = snapshot_download(
        repo_id=args.model_id,
        repo_type="model",
        revision=args.revision,
        local_dir=str(args.local_dir),
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        token=args.token or os.getenv("HF_TOKEN"),
        allow_patterns=args.include,
        ignore_patterns=args.exclude,
        max_workers=args.max_workers,
        resume_download=True,
    )

    print(f"Downloaded to: {downloaded_path}")


if __name__ == "__main__":
    main()
