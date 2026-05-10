           v                        # Milestone 2 — StormCare RAG (Track C)

This repo includes:
- a minimal retrieval system (BM25 indexing + top‑k retrieval)
- a trained tokenizer with saved artifacts + diagnostics
- a minimal end‑to‑end training smoke test that runs without errors and produces finite loss

The goal is evidence that indexing/retrieval + tokenization + batching + a trainable model work end‑to‑end.

## Milestone 3 updates (final integration)

I added a fair base-vs-adapted (LoRA) comparison script and a final integration guide.

New files:
- `training_code/base_vs_peft.py`
- `README_M3.md`

Run the Milestone 3 comparison:

```bash
.venv_py39/bin/python training_code/base_vs_peft.py
```

This writes:
- `results/base_vs_adapted.json`
- `results/base_vs_adapted.md`

Comparison setup (fairness):
- same task (next-token LM)
- same held-out split
- same metrics (test loss, perplexity)
- same evaluation procedure
- same seed and initialization

LoRA details reported by the script:
- target modules
- rank, alpha, dropout
- trainable vs total parameter counts
- base vs adapted quantitative table

## Setup (reproducible)

Recommended (macOS): create a clean venv using the system Python 3.9:

```bash
/usr/bin/python3 -m venv .venv_py39
.venv_py39/bin/python -m pip install -U pip
.venv_py39/bin/python -m pip install -r requirements.txt
```

## (1) Track C retrieval demo (index + retrieve)

Dataset indexed:
- Preferred: `data/corpus.jsonl` (generated from your PDFs)
- Fallback: `data/hurricane_data.jsonl` (tiny toy corpus if PDFs are not provided)

If your real corpus is **two PDFs**, put them in:
- `data/pdfs/`

Then build the JSONL corpus:

```bash
.venv_py39/bin/python training_code/pdf_to_jsonl.py --mode paragraph --min_chars 200
```

This creates `data/corpus.jsonl` with one row per paragraph chunk and includes metadata fields:
- `source` (PDF filename)
- `page` (1-based page number)
- `chunk_id`

Run:

```bash
.venv_py39/bin/python training_code/retrieval/run_retrieval_demo.py
```

Output:
- `results/retrieval_demo.json` (queries + top‑k hits + scores)

In paragraph-chunk mode, each retrieved hit also includes `meta.source`, `meta.page`, and `meta.chunk_id`.

## (2) Tokenizer and vocabulary (required)

### 2.1 Tokenizer training (custom)

Corpus used for tokenizer training
- Source: preferred `data/corpus.jsonl` (built from `data/pdfs/*.pdf`), else fallback `data/hurricane_data.jsonl`
- Size used: all documents present in the chosen JSONL (latest PDF paragraph corpus: 91 chunks)
- Selection: in paragraph mode, all extracted paragraph chunks with `len(text) >= 200` characters (`--min_chars 200`), then all non-empty `text` entries
- Train/validation split policy: not applicable for tokenizer training (trained on the full corpus)

Preprocessing decisions
- Normalization: Unicode NFKC + lowercasing
- Whitespace handling: HuggingFace `Whitespace` pre-tokenizer
- Deduplication: none
- Filtering: empty/blank lines dropped; in PDF paragraph mode, short chunks are filtered out by `--min_chars` to reduce page headers/footers

Why these steps
- NFKC + lowercasing reduces superficial token variants.
- Whitespace pre-tokenization keeps training stable/transparent.

Tokenization method
- Algorithm: BPE
- Implementation: HuggingFace `tokenizers`
- Special tokens:
	- `[PAD]` padding token
	- `[UNK]` unknown token
	- `[BOS]` begin-of-sequence
	- `[EOS]` end-of-sequence
- Unknown handling: `[UNK]` id

Vocabulary size justification
- Requested vocab size: 256
- Actual vocab size: 256 (see `training_code/tokenizer/tokenizer_meta.json`).
- Justification:
	- Memory/compute tradeoff: a small vocab keeps embeddings + softmax cheap for a CPU smoke test.
	- Sequence length impact: smaller vocab tends to create slightly longer token sequences, but keeps training stable given the very short context length (T=64).
	- Domain coverage: BPE merges still learn common subwords in this hurricane/meteorology corpus; diagnostics show 0.0 UNK rate on the sampled corpus.

Train tokenizer (writes artifacts to `training_code/tokenizer/`):

```bash
.venv_py39/bin/python training_code/train_tokenizer.py
```

Artifacts produced
- `training_code/tokenizer/tokenizer.json`
- `training_code/tokenizer/vocab.json`
- `training_code/tokenizer/merges.txt`
- `training_code/tokenizer/tokenizer_meta.json`

Tokenizer diagnostics (evidence)

Run:

```bash
.venv_py39/bin/python training_code/tokenizer_diagnostics.py
```

Output:
- `results/tokenizer_diagnostics.json` (avg tokens/doc, tokens/1k chars, UNK rate, example encodings)

Quick checks from the latest run (PDF paragraph corpus):
- Avg tokens/doc: 605.48
- Tokens per 1k chars: 445.19
- Unknown token rate: 0.0

Example encodings (short snippets):
- Text: "flooding storm surge" → Tokens: `[BOS] f lo o d ing storm s ur ge [EOS]`
- Text: "data assimilation improves genesis forecast" → Tokens: `[BOS] d at a assimil ation im pro v es genesis forecast [EOS]`

## (3) Model skeleton + data pipeline smoke test (required)

### 3.1 Data and loader

Dataset/corpus:
- preferred `data/corpus.jsonl` (PDF-derived), else fallback `data/hurricane_data.jsonl`

Train split used:
- document-level split with fixed seed (`seed=42`): 75% train / 25% val

Loader details:
- Input format: JSONL
- Context length ($T$): 64
- Batch size ($B$): 8
- Chunking: build a long token stream and slice into fixed blocks
- Output tensors:
	- `input_ids`: `[B, T]`
	- `labels`: `[B, T]` (next-token labels)

### 3.2 Model skeleton

Small decoder-only Transformer (`MiniGPT`) in `training_code/smoke_test.py`:
- Layers: 2
- Heads: 4
- Hidden size: 128
- Dropout: 0.1
- Masking: causal (future tokens masked)
- Parameter count: 470,528 (with vocab size 256)

### 3.3 Minimal training loop (smoke test)

- Objective: next-token cross entropy
- Optimizer: AdamW
- Learning rate: 3e-4
- Steps: 80 (enough to exercise the pipeline)
- Hardware: CPU by default (uses CUDA if available)

Run:

```bash
.venv_py39/bin/python training_code/smoke_test.py
```

### 3.4 Evidence the smoke test worked

The run:
- completes without errors
- logs finite losses
- shows stable loss (not exploding)

Outputs:
- `results/loss_log.txt` (tab-separated: step, train_loss, val_loss)
- `results/loss_plot.png`

Sanity evidence from `results/loss_log.txt`:
- Step 0: train_loss=5.7481, val_loss=5.6928
- Step 79: train_loss=4.9200, val_loss=4.9846

### 3.5 What worked and what did not work yet

Worked:
- End-to-end next-token training completes and produces finite, non-exploding loss.
- Tokenizer artifacts + diagnostics are generated from the PDF-derived JSONL corpus.
- Retrieval demo returns relevant paragraph chunks with source/page/chunk metadata.

Known limitations:
- Retrieval is BM25-only; this milestone does not implement a full generator that conditions on retrieved context.
- PDF extraction is text-based (no OCR); scanned PDFs would require an OCR fallback.
- Vocab size is intentionally small (256) for a fast smoke test, not for best language modeling quality.

## (4) Reproducibility practices (required)

Checklist:
- Fixed random seeds: Python / NumPy / PyTorch (`seed=42` in `training_code/smoke_test.py`)
- Exact run instructions: commands in this README
- Clear output locations: tokenizer artifacts under `training_code/tokenizer/`, logs/plots under `results/`

## (5) ZIP contents and structure

Expected contents:
- `report.pdf`
- `results/` (loss log, loss plot, retrieval demo, tokenizer diagnostics)
- `training_code/` (tokenizer training, retrieval demo, smoke test)
- `README.md` (this file)
