"""Quick tokenizer diagnostics for Milestone 2.

Computes a few evidence metrics and writes them to `results/`:
- avg_tokens_per_doc
- tokens_per_1k_chars
- unk_token_rate
- example encodings

Run (after training the tokenizer):
  .venv_py39/bin/python training_code/tokenizer_diagnostics.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import mean
from typing import Dict, Iterable, List, Any

# Make `training_code.*` importable when running as a script.
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tokenizers import Tokenizer

from training_code.corpus_utils import default_corpus_path, iter_jsonl_texts


def iter_corpus_texts(jsonl_path: str) -> Iterable[str]:
    # Kept for backward compatibility (delegates to shared loader).
    yield from iter_jsonl_texts(jsonl_path)


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_jsonl",
        default=default_corpus_path(repo_root),
    )
    parser.add_argument(
        "--tokenizer_json",
        default=os.path.join(repo_root, "training_code", "tokenizer", "tokenizer.json"),
    )
    parser.add_argument(
        "--output_json",
        default=os.path.join(repo_root, "results", "tokenizer_diagnostics.json"),
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=200,
        help="Max number of documents/rows to evaluate for diagnostics (useful for chunked PDF corpora).",
    )
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer_json)
    unk_id = tokenizer.token_to_id("[UNK]")

    texts: List[str] = []
    for t in iter_corpus_texts(args.input_jsonl):
        texts.append(t)
        if len(texts) >= args.max_docs:
            break
    encodings = [tokenizer.encode(t) for t in texts]

    token_counts = [len(e.ids) for e in encodings]
    char_counts = [len(t) for t in texts]

    total_tokens = sum(token_counts)
    total_chars = sum(char_counts)

    unk_count = 0
    if unk_id is not None:
        for e in encodings:
            unk_count += sum(1 for i in e.ids if i == unk_id)

    examples: List[Dict[str, Any]] = []
    diagnostics: Dict[str, Any] = {
        "n_docs": len(texts),
        "max_docs": args.max_docs,
        "avg_tokens_per_doc": float(mean(token_counts)) if token_counts else 0.0,
        "tokens_per_1k_chars": float((total_tokens / max(total_chars, 1)) * 1000.0),
        "unk_token_rate": float(unk_count / max(total_tokens, 1)),
        "examples": examples,
    }

    for t in texts[:3]:
        e = tokenizer.encode(t)
        examples.append(
            {
                "text": t,
                "tokens": e.tokens,
                "ids": e.ids,
            }
        )

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    print("Wrote tokenizer diagnostics to:", args.output_json)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
