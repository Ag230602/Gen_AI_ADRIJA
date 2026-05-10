"""Train a small BPE tokenizer for the Track C milestone.

This script trains a tokenizer on `data/hurricane_data.jsonl` and writes
artifacts under `training_code/tokenizer/`.

Artifacts (required):
- tokenizer.json (full tokenizer config)
- vocab.json + merges.txt (BPE model files)

Run (recommended venv):
  .venv_py39/bin/python training_code/train_tokenizer.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterable, List

# Make `training_code.*` importable when running as a script.
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training_code.corpus_utils import default_corpus_path, iter_jsonl_texts

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS: List[str] = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

def iter_corpus_texts(jsonl_path: str) -> Iterable[str]:
    # Kept for backward compatibility (delegates to shared loader).
    yield from iter_jsonl_texts(jsonl_path)


def train_bpe_tokenizer(texts: Iterable[str], vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(texts, trainer=trainer)

    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    if bos_id is None or eos_id is None:
        raise RuntimeError("Special tokens missing after training.")

    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] $B:1 [EOS]:1",
        special_tokens=[("[BOS]", bos_id), ("[EOS]", eos_id)],
    )

    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_jsonl",
        default=default_corpus_path(os.path.dirname(os.path.dirname(__file__))),
        help="Path to jsonl file containing {'text': ...} lines.",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.dirname(__file__), "tokenizer"),
        help="Directory to write tokenizer artifacts.",
    )
    parser.add_argument("--vocab_size", type=int, default=256)
    args = parser.parse_args()

    texts = list(iter_corpus_texts(args.input_jsonl))
    if not texts:
        raise RuntimeError(f"No texts found in {args.input_jsonl}")

    tokenizer = train_bpe_tokenizer(texts, vocab_size=args.vocab_size)
    actual_vocab_size = int(tokenizer.get_vocab_size())

    os.makedirs(args.output_dir, exist_ok=True)

    # Save full tokenizer configuration.
    tokenizer_json_path = os.path.join(args.output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_json_path)

    # Save BPE model files.
    vocab_path, merges_path = tokenizer.model.save(args.output_dir)

    # Save a tiny metadata file for the report/README.
    meta = {
        "algorithm": "BPE",
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": actual_vocab_size,
        "special_tokens": SPECIAL_TOKENS,
        "normalization": ["NFKC", "lowercase"],
        "pre_tokenizer": "Whitespace",
        "input_jsonl": os.path.relpath(args.input_jsonl, os.path.dirname(os.path.dirname(__file__))),
        "artifacts": {
            "tokenizer_json": os.path.basename(tokenizer_json_path),
            "vocab_json": os.path.basename(vocab_path),
            "merges_txt": os.path.basename(merges_path),
        },
    }
    with open(os.path.join(args.output_dir, "tokenizer_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Wrote tokenizer artifacts to:", args.output_dir)
    print("-", tokenizer_json_path)
    print("-", vocab_path)
    print("-", merges_path)


if __name__ == "__main__":
    main()
