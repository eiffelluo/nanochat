"""
Pre-download all data needed by the full training pipeline.

Run this before training to avoid GPU idle time caused by on-demand downloads
during evaluation and SFT stages. Already-downloaded data is skipped automatically.

Usage:
    python -m scripts.prefetch_data                    # default: 82 parquet shards (runcuda.sh)
    python -m scripts.prefetch_data --num-shards 370   # speedrun.sh scale
    python -m scripts.prefetch_data --num-shards 8     # minimal / dev
    python -m scripts.prefetch_data --skip-hf          # skip HuggingFace datasets
"""

import os
import sys
import time
import argparse
import zipfile
import shutil
import tempfile

from nanochat.common import get_base_dir, download_file_with_lock


def prefetch_parquet_shards(num_shards, num_workers):
    """Download pretraining parquet shards."""
    from nanochat.dataset import download_single_file, DATA_DIR, MAX_SHARD
    from multiprocessing import Pool

    num = min(num_shards, MAX_SHARD + 1)
    existing = sum(1 for i in range(num) if os.path.exists(os.path.join(DATA_DIR, f"shard_{i:05d}.parquet")))
    if existing == num:
        print(f"[parquet] All {num} shards already present in {DATA_DIR}, skipping.")
        return
    print(f"[parquet] Downloading {num} shards ({existing} already present) → {DATA_DIR}")
    ids_to_download = list(range(num))
    with Pool(processes=num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)
    ok = sum(1 for r in results if r)
    print(f"[parquet] Done: {ok}/{num} shards ready.")


def prefetch_eval_bundle():
    """Download and unzip the CORE evaluation bundle."""
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    if os.path.exists(eval_bundle_dir):
        print(f"[eval_bundle] Already present at {eval_bundle_dir}, skipping.")
        return

    url = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
    print(f"[eval_bundle] Downloading → {eval_bundle_dir}")

    def place_eval_bundle(file_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            extracted = os.path.join(tmpdir, "eval_bundle")
            shutil.move(extracted, eval_bundle_dir)

    download_file_with_lock(url, "eval_bundle.zip", postprocess_fn=place_eval_bundle)
    print(f"[eval_bundle] Done.")


def prefetch_identity_conversations():
    """Download the identity conversations JSONL for SFT."""
    base_dir = get_base_dir()
    filepath = os.path.join(base_dir, "identity_conversations.jsonl")
    if os.path.exists(filepath):
        print(f"[identity] Already present at {filepath}, skipping.")
        return

    url = "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"
    print(f"[identity] Downloading → {filepath}")
    download_file_with_lock(url, "identity_conversations.jsonl")
    print(f"[identity] Done.")


def prefetch_word_list():
    """Download the English word list used by SpellingBee."""
    base_dir = get_base_dir()
    filename = "words_alpha.txt"
    filepath = os.path.join(base_dir, filename)
    if os.path.exists(filepath):
        print(f"[wordlist] Already present at {filepath}, skipping.")
        return

    url = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"
    print(f"[wordlist] Downloading → {filepath}")
    download_file_with_lock(url, filename)
    print(f"[wordlist] Done.")


def prefetch_hf_datasets():
    """Pre-download all HuggingFace datasets used in SFT and chat evaluation."""
    from datasets import load_dataset

    hf_specs = [
        ("HuggingFaceTB/smol-smoltalk", None, "train"),
        ("HuggingFaceTB/smol-smoltalk", None, "test"),
        ("cais/mmlu", "auxiliary_train", "train"),
        ("cais/mmlu", "all", "test"),
        ("openai/gsm8k", "main", "train"),
        ("openai/gsm8k", "main", "test"),
        ("allenai/ai2_arc", "ARC-Easy", "test"),
        ("allenai/ai2_arc", "ARC-Challenge", "test"),
        ("openai/openai_humaneval", None, "test"),
    ]

    for name, subset, split in hf_specs:
        label = f"{name}" + (f"/{subset}" if subset else "") + f" [{split}]"
        print(f"[hf] Loading {label} ...")
        t0 = time.time()
        try:
            if subset:
                load_dataset(name, subset, split=split)
            else:
                load_dataset(name, split=split)
            elapsed = time.time() - t0
            print(f"[hf] {label} ready ({elapsed:.1f}s)")
        except Exception as e:
            print(f"[hf] WARNING: failed to load {label}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Pre-download all training pipeline data")
    parser.add_argument("--num-shards", type=int, default=82,
                        help="Number of parquet shards to download (default: 82 for runcuda.sh, 370 for speedrun.sh)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Parallel workers for parquet downloads (default: 4)")
    parser.add_argument("--skip-hf", action="store_true",
                        help="Skip HuggingFace dataset downloads")
    args = parser.parse_args()

    base_dir = get_base_dir()
    print(f"NANOCHAT_BASE_DIR: {base_dir}")
    print(f"=" * 60)

    t_start = time.time()

    # 1. Pretraining parquet shards
    prefetch_parquet_shards(args.num_shards, args.num_workers)

    # 2. CORE evaluation bundle
    prefetch_eval_bundle()

    # 3. Identity conversations for SFT
    prefetch_identity_conversations()

    # 4. Word list for SpellingBee
    prefetch_word_list()

    # 5. HuggingFace datasets (SFT + chat eval)
    if not args.skip_hf:
        prefetch_hf_datasets()
    else:
        print("[hf] Skipped (--skip-hf).")

    elapsed = time.time() - t_start
    print(f"=" * 60)
    print(f"All data prefetched in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
