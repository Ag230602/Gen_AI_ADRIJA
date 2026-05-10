"""Minimal BM25 retriever (pure Python) for Track C.

This is intentionally lightweight for Milestone 2: it provides indexing
and retrieval over a small jsonl corpus without extra dependencies.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


@dataclass
class BM25Index:
    documents: List[str]
    doc_tokens: List[List[str]]
    doc_len: List[int]
    avgdl: float
    df: Dict[str, int]
    tf: List[Counter]

    @property
    def n_docs(self) -> int:
        return len(self.documents)


def build_bm25_index(documents: Sequence[str]) -> BM25Index:
    doc_tokens = [tokenize(d) for d in documents]
    doc_len = [len(toks) for toks in doc_tokens]
    avgdl = sum(doc_len) / max(len(doc_len), 1)

    tf = [Counter(toks) for toks in doc_tokens]
    df: Dict[str, int] = {}
    for toks in doc_tokens:
        for term in set(toks):
            df[term] = df.get(term, 0) + 1

    return BM25Index(
        documents=list(documents),
        doc_tokens=doc_tokens,
        doc_len=doc_len,
        avgdl=avgdl,
        df=df,
        tf=tf,
    )


def bm25_scores(
    index: BM25Index,
    query: str,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    q_terms = tokenize(query)
    if not q_terms:
        return [0.0 for _ in range(index.n_docs)]

    scores = [0.0 for _ in range(index.n_docs)]
    N = index.n_docs

    for term in q_terms:
        df = index.df.get(term, 0)
        if df == 0:
            continue

        # Standard BM25 idf (with +1 inside log to keep it positive).
        idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))

        for i in range(N):
            freq = index.tf[i].get(term, 0)
            if freq == 0:
                continue
            dl = index.doc_len[i]
            denom = freq + k1 * (1.0 - b + b * (dl / max(index.avgdl, 1e-9)))
            scores[i] += idf * (freq * (k1 + 1.0) / denom)

    return scores


def top_k(
    index: BM25Index,
    query: str,
    k: int = 3,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[Tuple[int, float, str]]:
    scores = bm25_scores(index, query=query, k1=k1, b=b)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    out: List[Tuple[int, float, str]] = []
    for doc_id, score in ranked[:k]:
        out.append((doc_id, float(score), index.documents[doc_id]))
    return out
