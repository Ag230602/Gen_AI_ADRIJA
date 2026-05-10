"""Shared helpers for loading the Milestone 2 text corpus."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List


def default_corpus_path(repo_root: str) -> str:
    """Prefer PDF-derived corpus if present, else fall back to the toy jsonl."""
    pdf_corpus = os.path.join(repo_root, "data", "corpus.jsonl")
    fallback = os.path.join(repo_root, "data", "hurricane_data.jsonl")
    return pdf_corpus if os.path.exists(pdf_corpus) else fallback


def iter_jsonl_texts(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text", "")
            if isinstance(text, str) and text.strip():
                yield text.strip()


def iter_jsonl_rows(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def load_jsonl_rows(path: str) -> List[Dict[str, Any]]:
    return list(iter_jsonl_rows(path))


def load_jsonl_texts(path: str) -> List[str]:
    return list(iter_jsonl_texts(path))
