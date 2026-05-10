"""Milestone 3: fair quantitative base-vs-PEFT comparison for the MiniGPT pipeline.

This script trains and evaluates two systems on the same task and held-out set:
1) Base system: full-parameter training
2) Adapted system: LoRA-style PEFT on selected linear modules

Task: next-token language modeling
Metric: test loss + perplexity

Run:
  .venv_py39/bin/python training_code/base_vs_peft.py

Outputs:
  results/base_vs_adapted.json
  results/base_vs_adapted.md
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from tokenizers import Tokenizer
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Make `training_code.*` importable when running as a script.
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training_code.corpus_utils import default_corpus_path, load_jsonl_texts


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_train_val_test(
    items: List[str], val_fraction: float = 0.1, test_fraction: float = 0.1, seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    if not items:
        return [], [], []

    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)

    n = len(items)
    n_val = max(1, int(round(n * val_fraction)))
    n_test = max(1, int(round(n * test_fraction)))
    n_train = max(1, n - n_val - n_test)

    train_idx = set(idx[:n_train])
    val_idx = set(idx[n_train : n_train + n_val])
    test_idx = set(idx[n_train + n_val : n_train + n_val + n_test])

    train = [items[i] for i in range(n) if i in train_idx]
    val = [items[i] for i in range(n) if i in val_idx]
    test = [items[i] for i in range(n) if i in test_idx]
    return train, val, test


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
    def __init__(self, token_ids: List[int], seq_len: int):
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        self.seq_len = seq_len
        self.token_ids = token_ids
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
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)


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
        bsz, seqlen = input_ids.shape
        if seqlen != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seqlen}")

        pos = torch.arange(0, seqlen, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        x = self.blocks(x, mask=causal_mask(seqlen, input_ids.device))
        x = self.ln_f(x)
        return self.lm_head(x)


class LoRALinear(nn.Module):
    """LoRA wrapper for an nn.Linear module.

    Forward: xW^T + scale * (xA^T)B^T (+ bias if present in base linear)
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / max(rank, 1)
        self.drop = nn.Dropout(dropout)

        in_features = base.in_features
        out_features = base.out_features

        # Freeze base weights.
        for p in self.base.parameters():
            p.requires_grad = False

        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.lora_b(self.drop(self.lora_a(x)))


@dataclass
class ExperimentConfig:
    seq_len: int = 64
    batch_size: int = 8
    lr: float = 3e-4
    max_steps: int = 120
    eval_every: int = 20
    max_train_tokens: int = 220_000
    repeat_corpus: int = 250
    seed: int = 42


@dataclass
class LoRAConfig:
    target_modules: Tuple[str, ...] = ("lm_head",)
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


@torch.no_grad()
def eval_loss(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 8) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses: List[float] = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        losses.append(float(loss.item()))
    return float(sum(losses) / max(len(losses), 1))


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: ExperimentConfig,
) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)

    global_step = 0
    last_val_loss = float("nan")

    while global_step < cfg.max_steps:
        for batch in train_loader:
            model.train()
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)

            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {global_step}: {loss.item()}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if global_step % cfg.eval_every == 0:
                last_val_loss = eval_loss(model, val_loader, device=device)

            global_step += 1
            if global_step >= cfg.max_steps:
                break

    return {"train_loss_last": float(loss.item()), "val_loss_last": float(last_val_loss)}


def apply_lora(model: nn.Module, lora_cfg: LoRAConfig) -> List[str]:
    replaced: List[str] = []

    # Freeze everything first.
    for p in model.parameters():
        p.requires_grad = False

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        if not any(name.endswith(t) for t in lora_cfg.target_modules):
            continue

        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        attr_name = name.split(".")[-1]

        parent = model
        if parent_name:
            for tok in parent_name.split("."):
                parent = getattr(parent, tok)

        setattr(parent, attr_name, LoRALinear(module, rank=lora_cfg.rank, alpha=lora_cfg.alpha, dropout=lora_cfg.dropout))
        replaced.append(name)

    if not replaced:
        raise RuntimeError("No linear modules matched LoRA target_modules.")

    return replaced


def build_loaders(cfg: ExperimentConfig, tokenizer: Tokenizer, texts: List[str]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_texts, val_texts, test_texts = split_train_val_test(texts, val_fraction=0.1, test_fraction=0.1, seed=cfg.seed)

    base_train_ids = encode_texts_limited(tokenizer, train_texts, max_tokens=cfg.max_train_tokens)
    base_val_ids = encode_texts_limited(tokenizer, val_texts, max_tokens=max(10_000, cfg.max_train_tokens // 6))
    base_test_ids = encode_texts_limited(tokenizer, test_texts, max_tokens=max(10_000, cfg.max_train_tokens // 6))

    if not base_train_ids or not base_val_ids or not base_test_ids:
        raise RuntimeError("Tokenized train/val/test streams were empty. Check corpus and tokenizer artifacts.")

    train_repeat = min(cfg.repeat_corpus, max(1, cfg.max_train_tokens // max(len(base_train_ids), 1)))
    val_repeat = max(1, train_repeat // 5)
    test_repeat = max(1, train_repeat // 5)

    train_ids = (base_train_ids * train_repeat)[: cfg.max_train_tokens]
    val_ids = base_val_ids * val_repeat
    test_ids = base_test_ids * test_repeat

    train_ds = CausalLMDataset(train_ids, seq_len=cfg.seq_len)
    val_ds = CausalLMDataset(val_ids, seq_len=cfg.seq_len)
    test_ds = CausalLMDataset(test_ids, seq_len=cfg.seq_len)

    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError("A split produced 0 samples. Reduce seq_len or increase data volume.")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=min(cfg.batch_size, len(val_ds)), shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=min(cfg.batch_size, len(test_ds)), shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader


def main() -> None:
    cfg = ExperimentConfig()
    lora_cfg = LoRAConfig()
    set_seed(cfg.seed)

    tok_path = os.path.join(REPO_ROOT, "training_code", "tokenizer", "tokenizer.json")
    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"Missing tokenizer at {tok_path}. Run training_code/train_tokenizer.py first.")

    corpus_path = default_corpus_path(REPO_ROOT)
    texts = load_jsonl_texts(corpus_path)
    if not texts:
        raise RuntimeError(f"No texts found at {corpus_path}")

    tokenizer = Tokenizer.from_file(tok_path)
    vocab_size = int(tokenizer.get_vocab_size())
    train_loader, val_loader, test_loader = build_loaders(cfg, tokenizer, texts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Shared initialization for fair comparison.
    base_init = MiniGPT(vocab_size=vocab_size, seq_len=cfg.seq_len, d_model=128, n_head=4, n_layer=2, dropout=0.1)
    init_state = copy.deepcopy(base_init.state_dict())

    # Base system.
    base_model = MiniGPT(vocab_size=vocab_size, seq_len=cfg.seq_len, d_model=128, n_head=4, n_layer=2, dropout=0.1).to(device)
    base_model.load_state_dict(init_state)
    base_params = count_parameters(base_model)
    base_train = train_model(base_model, train_loader, val_loader, device, cfg)
    base_test_loss = eval_loss(base_model, test_loader, device=device, max_batches=16)

    # Adapted (LoRA) system.
    lora_model = MiniGPT(vocab_size=vocab_size, seq_len=cfg.seq_len, d_model=128, n_head=4, n_layer=2, dropout=0.1).to(device)
    lora_model.load_state_dict(init_state)
    replaced = apply_lora(lora_model, lora_cfg)
    lora_params = count_parameters(lora_model)
    lora_train = train_model(lora_model, train_loader, val_loader, device, cfg)
    lora_test_loss = eval_loss(lora_model, test_loader, device=device, max_batches=16)

    result = {
        "task": "next_token_language_modeling",
        "evaluation_set": "held-out test split (same split for base and adapted)",
        "metrics": ["test_loss", "test_perplexity"],
        "base_model": {
            "name": "MiniGPT-2L-128d (custom)",
            "size": "~0.47M parameters",
            "tokenizer": "training_code/tokenizer/tokenizer.json",
            "trainable_parameters": base_params["trainable"],
            "total_parameters": base_params["total"],
            "training": asdict(cfg),
            "results": {
                **base_train,
                "test_loss": float(base_test_loss),
                "test_perplexity": float(math.exp(min(base_test_loss, 20.0))),
            },
        },
        "adapted_model": {
            "method": "LoRA (PEFT)",
            "applied_to": replaced,
            "lora_config": asdict(lora_cfg),
            "trainable_parameters": lora_params["trainable"],
            "total_parameters": lora_params["total"],
            "training": asdict(cfg),
            "results": {
                **lora_train,
                "test_loss": float(lora_test_loss),
                "test_perplexity": float(math.exp(min(lora_test_loss, 20.0))),
            },
        },
        "fairness": {
            "same_task": True,
            "same_heldout_set": True,
            "same_metrics": True,
            "same_eval_procedure": True,
            "shared_initialization": True,
            "seed": cfg.seed,
        },
    }

    results_dir = os.path.join(REPO_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "base_vs_adapted.json")
    md_path = os.path.join(results_dir, "base_vs_adapted.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    base_ppl = result["base_model"]["results"]["test_perplexity"]
    lora_ppl = result["adapted_model"]["results"]["test_perplexity"]
    base_loss = result["base_model"]["results"]["test_loss"]
    lora_loss = result["adapted_model"]["results"]["test_loss"]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Base vs Adapted (LoRA)\n\n")
        f.write("| System | Trainable Params | Total Params | Test Loss | Test Perplexity |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        f.write(
            f"| Base | {base_params['trainable']} | {base_params['total']} | {base_loss:.4f} | {base_ppl:.4f} |\n"
        )
        f.write(
            f"| Adapted (LoRA) | {lora_params['trainable']} | {lora_params['total']} | {lora_loss:.4f} | {lora_ppl:.4f} |\n"
        )

    print("Wrote:", json_path)
    print("Wrote:", md_path)
    print("Base test loss / ppl:", f"{base_loss:.4f}", f"{base_ppl:.4f}")
    print("LoRA test loss / ppl:", f"{lora_loss:.4f}", f"{lora_ppl:.4f}")


if __name__ == "__main__":
    main()
