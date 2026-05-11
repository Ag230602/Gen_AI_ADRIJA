"""Evaluate BM25 retrieval quality with explicit Track C metrics.

Computes:
- Recall@k
- HitRate@k
- Precision@k
- MRR

Outputs:
- results/retrieval_eval.json
- results/retrieval_eval.md
- code/results/retrieval_eval.json
- code/results/retrieval_eval.md

Run:
  .venv/bin/python code/training_code/retrieval/evaluate_retrieval.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Sequence, Set

# Make `training_code.*` importable when running as a script.
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PROJECT_ROOT = os.path.dirname(CODE_ROOT)
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from training_code.corpus_utils import default_corpus_path, load_jsonl_rows
from training_code.retrieval.bm25 import build_bm25_index, top_k


# Hand-labeled relevance judgments for representative Track C queries.
# Keys use source/page/chunk_id so they remain stable across doc_id reorderings.
EVAL_SET = [
    {
        "query": "flooding storm surge",
        "relevant": [
            {
                "source": "september-2017-hurricane-irma-event-analysis-report.pdf",
                "page": 16,
                "chunk_id": 47,
            }
        ],
    },
    {
        "query": "evacuation orders",
        "relevant": [
            {
                "source": "september-2017-hurricane-irma-event-analysis-report.pdf",
                "page": 12,
                "chunk_id": 36,
            }
        ],
    },
    {
        "query": "emergency kits water batteries",
        "relevant": [
            {
                "source": "september-2017-hurricane-irma-event-analysis-report.pdf",
                "page": 20,
                "chunk_id": 58,
            }
        ],
    },
]


def meta_key(row: Dict) -> str:
    return f"{row.get('source')}|{row.get('page')}|{row.get('chunk_id')}"


def resolve_relevant_doc_ids(rows: Sequence[Dict], relevant_meta: Sequence[Dict]) -> Set[int]:
    row_key_to_doc_id = {meta_key(r): i for i, r in enumerate(rows)}
    out: Set[int] = set()
    missing: List[str] = []

    for rel in relevant_meta:
        k = f"{rel.get('source')}|{rel.get('page')}|{rel.get('chunk_id')}"
        if k in row_key_to_doc_id:
            out.add(row_key_to_doc_id[k])
        else:
            missing.append(k)

    if missing:
        raise RuntimeError(f"Missing relevance labels in corpus for keys: {missing}")

    return out


def precision_at_k(retrieved_ids: Sequence[int], relevant_ids: Set[int], k: int) -> float:
    topk = retrieved_ids[:k]
    if k <= 0:
        return 0.0
    hits = sum(1 for d in topk if d in relevant_ids)
    return float(hits / k)


def recall_at_k(retrieved_ids: Sequence[int], relevant_ids: Set[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    topk = retrieved_ids[:k]
    hits = sum(1 for d in topk if d in relevant_ids)
    return float(hits / len(relevant_ids))


def hit_rate_at_k(retrieved_ids: Sequence[int], relevant_ids: Set[int], k: int) -> float:
    topk = retrieved_ids[:k]
    return 1.0 if any(d in relevant_ids for d in topk) else 0.0


def reciprocal_rank(retrieved_ids: Sequence[int], relevant_ids: Set[int]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def write_json_and_md(result: Dict) -> None:
    json_targets = [
        os.path.join(CODE_ROOT, "results", "retrieval_eval.json"),
        os.path.join(PROJECT_ROOT, "results", "retrieval_eval.json"),
    ]
    md_targets = [
        os.path.join(CODE_ROOT, "results", "retrieval_eval.md"),
        os.path.join(PROJECT_ROOT, "results", "retrieval_eval.md"),
    ]

    for p in json_targets:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    metrics = result["metrics"]
    with open(md_targets[0], "w", encoding="utf-8") as f:
        f.write("# Retrieval Evaluation (Track C)\n\n")
        f.write("| Setting | Recall@1 | Recall@3 | Recall@5 | Hit@1 | Hit@3 | Hit@5 | Precision@1 | Precision@3 | Precision@5 | MRR |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        f.write(
            "| BM25 (k1=1.5, b=0.75) "
            f"| {metrics['Recall@1']:.4f}"
            f" | {metrics['Recall@3']:.4f}"
            f" | {metrics['Recall@5']:.4f}"
            f" | {metrics['HitRate@1']:.4f}"
            f" | {metrics['HitRate@3']:.4f}"
            f" | {metrics['HitRate@5']:.4f}"
            f" | {metrics['Precision@1']:.4f}"
            f" | {metrics['Precision@3']:.4f}"
            f" | {metrics['Precision@5']:.4f}"
            f" | {metrics['MRR']:.4f} |\n"
        )
        f.write("\n")
        f.write("Queries are evaluated with hand-labeled relevant chunks in this small project benchmark.\n")

    # copy markdown to code/results too
    os.makedirs(os.path.dirname(md_targets[1]), exist_ok=True)
    with open(md_targets[0], "r", encoding="utf-8") as src, open(md_targets[1], "w", encoding="utf-8") as dst:
        dst.write(src.read())


def main() -> None:
    data_path = default_corpus_path(CODE_ROOT)
    rows = load_jsonl_rows(data_path)
    docs = [(r.get("text", "") or "").strip() for r in rows]

    if not docs:
        raise RuntimeError(f"No documents found at: {data_path}")

    index = build_bm25_index(docs)

    k_values = [1, 3, 5]
    per_query: List[Dict] = []

    recall_sums = {k: 0.0 for k in k_values}
    hit_sums = {k: 0.0 for k in k_values}
    precision_sums = {k: 0.0 for k in k_values}
    mrr_sum = 0.0

    for q in EVAL_SET:
        query = q["query"]
        relevant_ids = resolve_relevant_doc_ids(rows, q["relevant"])

        ranked_hits = top_k(index, query, k=index.n_docs)
        retrieved_ids = [doc_id for (doc_id, _, _) in ranked_hits]

        metrics_q = {
            "query": query,
            "relevant_doc_ids": sorted(list(relevant_ids)),
            "first_relevant_rank": next((i + 1 for i, d in enumerate(retrieved_ids) if d in relevant_ids), None),
        }

        for k in k_values:
            metrics_q[f"Recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
            metrics_q[f"HitRate@{k}"] = hit_rate_at_k(retrieved_ids, relevant_ids, k)
            metrics_q[f"Precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
            recall_sums[k] += metrics_q[f"Recall@{k}"]
            hit_sums[k] += metrics_q[f"HitRate@{k}"]
            precision_sums[k] += metrics_q[f"Precision@{k}"]

        rr = reciprocal_rank(retrieved_ids, relevant_ids)
        metrics_q["RR"] = rr
        mrr_sum += rr

        preview = top_k(index, query, k=5)
        metrics_q["top5"] = [
            {
                "doc_id": doc_id,
                "score": score,
                "source": rows[doc_id].get("source"),
                "page": rows[doc_id].get("page"),
                "chunk_id": rows[doc_id].get("chunk_id"),
            }
            for (doc_id, score, _text) in preview
        ]

        per_query.append(metrics_q)

    n = len(EVAL_SET)
    overall = {
        "Recall@1": recall_sums[1] / n,
        "Recall@3": recall_sums[3] / n,
        "Recall@5": recall_sums[5] / n,
        "HitRate@1": hit_sums[1] / n,
        "HitRate@3": hit_sums[3] / n,
        "HitRate@5": hit_sums[5] / n,
        "Precision@1": precision_sums[1] / n,
        "Precision@3": precision_sums[3] / n,
        "Precision@5": precision_sums[5] / n,
        "MRR": mrr_sum / n,
    }

    result = {
        "method": "BM25",
        "index": {
            "type": "in-memory sparse index",
            "k1": 1.5,
            "b": 0.75,
            "n_docs": index.n_docs,
        },
        "evaluation": {
            "n_queries": len(EVAL_SET),
            "k_values": k_values,
            "labeling": "hand-labeled relevant chunks (source/page/chunk_id)",
        },
        "metrics": overall,
        "per_query": per_query,
    }

    write_json_and_md(result)

    print("Wrote: results/retrieval_eval.json")
    print("Wrote: results/retrieval_eval.md")
    print("Wrote: code/results/retrieval_eval.json")
    print("Wrote: code/results/retrieval_eval.md")


if __name__ == "__main__":
    main()
