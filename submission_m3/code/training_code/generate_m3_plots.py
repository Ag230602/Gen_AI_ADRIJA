"""Generate Milestone 3 summary plots.

Outputs:
- code/results/base_vs_adapted_bar.png
- code/results/retrieval_top1_scores.png
- code/results/training_loss_trend.png

Run:
  .venv/bin/python code/training_code/generate_m3_plots.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


CODE_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = CODE_ROOT / "results"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tsv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not path.exists():
        return rows
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < 2:
        return rows
    header = lines[0].split("\t")
    for ln in lines[1:]:
        parts = ln.split("\t")
        if len(parts) != len(header):
            continue
        rows.append(dict(zip(header, parts)))
    return rows


def plot_base_vs_adapted(results_dir: Path) -> None:
    data = load_json(results_dir / "base_vs_adapted.json")
    base = data["base_model"]["results"]
    adapted = data["adapted_model"]["results"]

    labels = ["Base", "Adapted (LoRA)"]
    losses = [base["test_loss"], adapted["test_loss"]]
    perplexities = [base["test_perplexity"], adapted["test_perplexity"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(labels, losses, color=["#1f77b4", "#ff7f0e"])
    axes[0].set_title("Test Loss")
    axes[0].set_ylabel("Loss")

    axes[1].bar(labels, perplexities, color=["#1f77b4", "#ff7f0e"])
    axes[1].set_title("Test Perplexity")
    axes[1].set_ylabel("Perplexity")

    fig.suptitle("Milestone 3: Base vs Adapted")
    fig.tight_layout()
    out = results_dir / "base_vs_adapted_bar.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote: {out}")


def plot_retrieval_top1(results_dir: Path) -> None:
    demo = load_json(results_dir / "retrieval_demo.json")

    queries = []
    scores = []
    for q in demo.get("queries", []):
        query = q.get("query", "")
        hits = q.get("top_k", [])
        top_score = hits[0].get("score", 0.0) if hits else 0.0
        queries.append(query)
        scores.append(top_score)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(queries, scores, color="#2ca02c")
    ax.set_xlabel("Top-1 BM25 score")
    ax.set_title("Milestone 3 Retrieval Quality (Top-1 per query)")
    fig.tight_layout()

    out = results_dir / "retrieval_top1_scores.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote: {out}")


def plot_training_loss_trend(results_dir: Path) -> None:
    rows = load_tsv(results_dir / "loss_log.txt")
    if not rows:
        print("Skip: loss_log.txt not found or empty")
        return

    steps = [int(r["step"]) for r in rows if "step" in r]
    train = [float(r["train_loss"]) for r in rows if "train_loss" in r]
    val = [float(r["val_loss"]) for r in rows if "val_loss" in r]
    test = [float(r["test_loss"]) for r in rows if "test_loss" in r]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    if steps and train and len(steps) == len(train):
        ax.plot(steps, train, label="train_loss", linewidth=1.8)
    if steps and val and len(steps) == len(val):
        ax.plot(steps, val, label="val_loss", linewidth=1.8)
    if steps and test and len(steps) == len(test):
        ax.plot(steps, test, label="test_loss", linewidth=1.8)

    ax.set_title("Milestone 3 Training Trend")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out = results_dir / "training_loss_trend.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote: {out}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_base_vs_adapted(RESULTS_DIR)
    plot_retrieval_top1(RESULTS_DIR)
    plot_training_loss_trend(RESULTS_DIR)


if __name__ == "__main__":
    main()
