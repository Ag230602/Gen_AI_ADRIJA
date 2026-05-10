"""Build a BM25 index and run a tiny retrieval demo (Track C).

Writes results to:
  results/retrieval_demo.json

Run:
  .venv_py39/bin/python training_code/retrieval/run_retrieval_demo.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import List

# Make `training_code.*` importable when running as a script.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training_code.retrieval.bm25 import build_bm25_index, top_k
from training_code.corpus_utils import default_corpus_path, load_jsonl_rows


def main() -> None:
    data_path = default_corpus_path(REPO_ROOT)
    out_path = os.path.join(REPO_ROOT, "results", "retrieval_demo.json")

    rows = load_jsonl_rows(data_path)
    docs = [(r.get("text", "") or "").strip() for r in rows]
    meta = []
    for r in rows:
        meta.append(
            {
                "source": r.get("source"),
                "page": r.get("page"),
                "chunk_id": r.get("chunk_id"),
            }
        )
    index = build_bm25_index(docs)

    queries = [
        "flooding storm surge",
        "evacuation orders",
        "emergency kits water batteries",
    ]

    demo = {"n_docs": index.n_docs, "queries": []}
    for q in queries:
        hits = top_k(index, q, k=3)
        demo["queries"].append(
            {
                "query": q,
                "top_k": [
                    {
                        "doc_id": doc_id,
                        "score": score,
                        "text": text,
                        **({"meta": meta[doc_id]} if doc_id < len(meta) else {}),
                    }
                    for (doc_id, score, text) in hits
                ],
            }
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(demo, f, indent=2)

    print("Wrote retrieval demo to:", out_path)
    for q in demo["queries"]:
        print("\nQ:", q["query"])
        for h in q["top_k"]:
            print(f"  score={h['score']:.3f} doc_id={h['doc_id']} text={h['text']}")


if __name__ == "__main__":
    main()
