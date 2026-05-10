"""Track C (Retrieval-Augmented Generation) - Milestone 2 smoke test.

This smoke test is a minimal, end-to-end *trainable* pipeline that:
- loads a small text corpus (`data/hurricane_data.jsonl`)
- tokenizes it with a trained BPE tokenizer (`training_code/tokenizer/tokenizer.json`)
- builds a causal language modeling dataset that returns:
    input_ids: [B, T]
    labels:    [B, T]  (next-token labels)
- trains a tiny decoder-only Transformer for a small number of steps
- logs train/val loss and saves a simple loss plot under `results/`

Run (recommended):
  .venv_py39/bin/python training_code/smoke_test.py
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Iterable, List, Tuple

# Make `training_code.*` importable when running as a script.
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tokenizers import Tokenizer
from torch import nn
from torch.utils.data import DataLoader, Dataset

from training_code.corpus_utils import default_corpus_path, load_jsonl_texts


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_train_val(items: List[str], val_fraction: float = 0.25, seed: int = 42) -> Tuple[List[str], List[str]]:
    if not items:
        return [], []
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(items) * val_fraction)))
    val_idx = set(idx[:n_val])
    train = [items[i] for i in range(len(items)) if i not in val_idx]
    val = [items[i] for i in range(len(items)) if i in val_idx]
    return train, val


def encode_texts_limited(tokenizer: Tokenizer, texts: List[str], max_tokens: int) -> List[int]:
    ids: List[int] = []
    for t in texts:
        enc = tokenizer.encode(t)
        remaining = max_tokens - len(ids)
        if remaining <= 0:
            break
        ids.extend(enc.ids[:remaining])
    return ids


class CausalLMDataset(Dataset):
    """Produces fixed-length next-token examples from a token stream."""

    def __init__(self, token_ids: List[int], seq_len: int):
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        self.seq_len = seq_len
        self.token_ids = token_ids

        # Each sample needs seq_len+1 tokens (for next-token labels).
        usable = len(token_ids) - (seq_len + 1)
        self.n_samples = 0 if usable < 0 else (usable // seq_len + 1)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        chunk = self.token_ids[start : start + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return {"input_ids": x, "labels": y}


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    # Shape [T, T] where upper triangle is -inf (masked).
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int = 128,
        n_head: int = 4,
        n_layer: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(enc_layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T]
        B, T = input_ids.shape
        if T != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got T={T}")

        pos = torch.arange(0, T, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        x = self.blocks(x, mask=causal_mask(T, input_ids.device))
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, V]
        return logits


@dataclass
class SmokeConfig:
    seq_len: int = 64
    batch_size: int = 8
    lr: float = 3e-4
    max_steps: int = 80
    eval_every: int = 10
    # For tiny corpora we repeat to get enough steps.
    # For large corpora (e.g., PDFs), repetition is capped by max_train_tokens.
    repeat_corpus: int = 200
    max_train_tokens: int = 200_000
    seed: int = 42


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def eval_loss(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 5) -> float:
    model.eval()
    losses: List[float] = []
    criterion = nn.CrossEntropyLoss()
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        losses.append(float(loss.item()))
    return float(mean_or_nan(losses))


def mean_or_nan(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def main() -> None:
    cfg = SmokeConfig()
    set_seed(cfg.seed)

    repo_root = os.path.dirname(os.path.dirname(__file__))
    data_path = default_corpus_path(repo_root)
    tok_path = os.path.join(repo_root, "training_code", "tokenizer", "tokenizer.json")
    results_dir = os.path.join(repo_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    loss_log_path = os.path.join(results_dir, "loss_log.txt")
    loss_plot_path = os.path.join(results_dir, "loss_plot.png")

    if not os.path.exists(tok_path):
        raise FileNotFoundError(
            f"Missing tokenizer artifact at {tok_path}. Run: .venv_py39/bin/python training_code/train_tokenizer.py"
        )

    texts = load_jsonl_texts(data_path)
    if not texts:
        raise RuntimeError(f"No texts found in {data_path}")

    train_texts, val_texts = split_train_val(texts, val_fraction=0.25, seed=cfg.seed)

    tokenizer = Tokenizer.from_file(tok_path)
    vocab_size = int(tokenizer.get_vocab_size())

    # Encode corpus. If it's tiny, repeat it; if it's large (PDFs), cap tokens.
    base_train_ids = encode_texts_limited(tokenizer, train_texts, max_tokens=cfg.max_train_tokens)
    base_val_ids = encode_texts_limited(tokenizer, val_texts, max_tokens=max(10_000, cfg.max_train_tokens // 5))

    if not base_train_ids or not base_val_ids:
        raise RuntimeError("Empty token stream after encoding. Check that PDF text extraction worked.")

    train_repeat = min(cfg.repeat_corpus, max(1, cfg.max_train_tokens // max(len(base_train_ids), 1)))
    val_repeat = max(1, train_repeat // 4)

    train_ids = (base_train_ids * train_repeat)[: cfg.max_train_tokens]
    val_ids = base_val_ids * val_repeat

    train_ds = CausalLMDataset(train_ids, seq_len=cfg.seq_len)
    val_ds = CausalLMDataset(val_ids, seq_len=cfg.seq_len)

    if len(train_ds) == 0:
        raise RuntimeError("Train dataset produced 0 samples. Try reducing seq_len or increasing repeat_corpus.")
    if len(val_ds) == 0:
        raise RuntimeError("Val dataset produced 0 samples. Try reducing seq_len or increasing repeat_corpus.")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_batch_size = min(cfg.batch_size, max(1, len(val_ds)))
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniGPT(vocab_size=vocab_size, seq_len=cfg.seq_len, d_model=128, n_head=4, n_layer=2, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"device={device}")
    print(
        "model=MiniGPT",
        f"vocab_size={vocab_size}",
        f"n_layer={2}",
        f"n_head={4}",
        f"d_model={128}",
        f"params={count_parameters(model)}",
    )
    print(f"train_samples={len(train_ds)} val_samples={len(val_ds)}")

    steps: List[int] = []
    train_losses: List[float] = []
    val_losses: List[float] = []

    global_step = 0
    last_val_loss = float("nan")
    with open(loss_log_path, "w", encoding="utf-8") as f:
        f.write("step\ttrain_loss\tval_loss\n")

        while global_step < cfg.max_steps:
            for batch in train_loader:
                model.train()
                x = batch["input_ids"].to(device)  # [B, T]
                y = batch["labels"].to(device)  # [B, T]

                logits = model(x)  # [B, T, V]
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {global_step}: {loss.item()}")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                train_loss = float(loss.item())
                if global_step % cfg.eval_every == 0:
                    last_val_loss = eval_loss(model, val_loader, device=device, max_batches=5)
                val_loss = last_val_loss

                steps.append(global_step)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(f"step={global_step} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
                f.write(f"{global_step}\t{train_loss:.6f}\t{val_loss:.6f}\n")

                global_step += 1
                if global_step >= cfg.max_steps:
                    break

            if global_step >= cfg.max_steps:
                break

    # Plot losses.
    plt.figure(figsize=(7, 4))
    plt.plot(steps, train_losses, label="train")
    # Only plot non-NaN val points.
    val_x = [s for s, v in zip(steps, val_losses) if not math.isnan(v)]
    val_y = [v for v in val_losses if not math.isnan(v)]
    if val_x:
        plt.plot(val_x, val_y, label="val")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Smoke test loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    print("Wrote loss log to:", loss_log_path)
    print("Wrote loss plot to:", loss_plot_path)


if __name__ == "__main__":
    main()
