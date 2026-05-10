"""Create reproducible train/val/test splits for the Milestone 2 JSONL corpus.

This script shuffles JSONL rows deterministically (fixed seed) and writes three
JSONL files under `data/splits/` inside the Milestone 2 submission bundle.

It is optional for the rubric, but makes evaluation and reproducibility easier.

Run:
  .venv_py39/bin/python training_code/make_splits.py

Outputs:
  data/splits/train.jsonl
  data/splits/val.jsonl
  data/splits/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple

from training_code.corpus_utils import default_corpus_path, load_jsonl_rows


def split_counts(n: int, train_frac: float, val_frac: float, test_frac: float) -> Tuple[int, int, int]:
    if n < 0:
        raise ValueError("n must be >= 0")
    if not (0.0 <= train_frac <= 1.0 and 0.0 <= val_frac <= 1.0 and 0.0 <= test_frac <= 1.0):
        raise ValueError("fractions must be in [0, 1]")
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {total}")

    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    # Ensure exact total by assigning the remainder to test.
    n_test = max(0, n - n_train - n_val)

    # If rounding caused overflow, shrink val then train.
    while n_train + n_val + n_test > n:
        if n_val > 0:
            n_val -= 1
        elif n_train > 0:
            n_train -= 1
        else:
            break

    return n_train, n_val, n_test


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_jsonl",
        default=default_corpus_path(repo_root),
        help="Input JSONL path (defaults to data/corpus.jsonl if present, else data/hurricane_data.jsonl)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic shuffling")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument(
        "--out_dir",
        default=os.path.join(repo_root, "data", "splits"),
        help="Output directory to write train/val/test JSONL files",
    )
    args = parser.parse_args()

    rows = load_jsonl_rows(args.input_jsonl)
    if not rows:
        raise RuntimeError(f"No rows found in {args.input_jsonl}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    n_train, n_val, n_test = split_counts(len(rows), args.train_frac, args.val_frac, args.test_frac)
    train_rows = rows[:n_train]
    val_rows = rows[n_train : n_train + n_val]
    test_rows = rows[n_train + n_val : n_train + n_val + n_test]

    # Write outputs
    train_path = os.path.join(args.out_dir, "train.jsonl")
    val_path = os.path.join(args.out_dir, "val.jsonl")
    test_path = os.path.join(args.out_dir, "test.jsonl")

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    write_jsonl(test_path, test_rows)

    print("Wrote splits:")
    print("-", os.path.relpath(train_path, repo_root), "rows=", len(train_rows))
    print("-", os.path.relpath(val_path, repo_root), "rows=", len(val_rows))
    print("-", os.path.relpath(test_path, repo_root), "rows=", len(test_rows))


if __name__ == "__main__":
    main()
