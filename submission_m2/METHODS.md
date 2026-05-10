# Milestone 2 (Track C) — Methods / Data Processing

This document is a report-ready description of the Milestone 2 Track C pipeline implemented in this repo. It describes: (1) how the PDF data is converted to JSONL, (2) how the tokenizer is trained and validated, (3) how retrieval is implemented and demonstrated, and (4) how the end-to-end next-token “smoke test” trains a small causal language model.

## Dataset (PDF → JSONL)

**Raw data**: two PDF documents placed under `data/pdfs/`.

**Ingestion script**: `training_code/pdf_to_jsonl.py` uses `pypdf` to extract text from each page.

**Chunking mode (RAG-style)**: the recommended mode is paragraph chunking, which turns each PDF page into multiple chunks split on blank lines.

- Output file: `data/corpus.jsonl`
- Output schema (paragraph mode):
  - `source`: PDF filename
  - `page`: 1-based page number
  - `chunk_id`: integer chunk id within the PDF
  - `text`: extracted paragraph text
- Filtering: very short chunks are removed via `--min_chars` (default: `200`) to reduce headers/page numbers.

**Latest corpus size (PDF paragraph chunks)**: `n_docs = 91` chunks (see `results/tokenizer_diagnostics.json` and `results/retrieval_demo.json`).

**Corpus selection policy**:

- Source: all PDFs under `data/pdfs/*.pdf`
- Size used: all extracted paragraph chunks meeting the minimum length threshold
- Selection: in paragraph mode, paragraphs are split on blank lines and filtered by `--min_chars` (default 200)
- Train/validation split policy: not used for tokenizer training (tokenizer trained on the full corpus); used only for the model smoke test (75/25 split with seed=42)

Reproducible command used:

```bash
.venv_py39/bin/python training_code/pdf_to_jsonl.py --mode paragraph
```

## Tokenizer (Trainable BPE)

**Goal**: build a trainable tokenizer and save required artifacts for submission.

**Training script**: `training_code/train_tokenizer.py` trains a Byte-Pair Encoding (BPE) tokenizer using HuggingFace `tokenizers`.

Tokenizer configuration (from `training_code/tokenizer/tokenizer_meta.json`):

- Algorithm: BPE
- Vocabulary size: requested `256`, actual `256`
- Normalization: Unicode NFKC + lowercase
- Whitespace handling / pre-tokenization: HuggingFace `Whitespace` pre-tokenizer
- Special tokens: `[PAD] [UNK] [BOS] [EOS]`
- Training data: `data/corpus.jsonl`

Additional preprocessing decisions:

- Deduplication: none
- Filtering: empty/blank texts are skipped; in PDF paragraph mode, short chunks are removed using `--min_chars` to reduce page headers/footers

Why these steps:

- NFKC + lowercase reduces superficial string variants (e.g., punctuation/Unicode variants and casing).
- Whitespace pre-tokenization keeps the training process simple and interpretable.
- Filtering short chunks reduces noisy non-content tokens (page numbers, section headers).

Artifacts written under `training_code/tokenizer/`:

- `tokenizer.json`
- `vocab.json`
- `merges.txt`
- `tokenizer_meta.json` (compact config for the report)

Reproducible command used:

```bash
.venv_py39/bin/python training_code/train_tokenizer.py
```

### Vocabulary size justification (256)

Chosen vocab size: 256.

- Memory/embedding tradeoff: vocab 256 keeps the embedding table small and makes the softmax cheap for a CPU-based smoke test.
- Sequence length impact: smaller vocab can increase token counts, but the model uses a short context length (T=64) and a capped token budget, so training remains stable.
- Domain term coverage: BPE still learns frequent domain subwords; diagnostics show 0.0 `[UNK]` rate on the sampled corpus.

## Tokenizer diagnostics (Evidence)

Diagnostics script: `training_code/tokenizer_diagnostics.py`

Diagnostics output: `results/tokenizer_diagnostics.json`

Key metrics from the latest PDF-chunk corpus:

- `n_docs`: `91`
- `avg_tokens_per_doc`: `605.48`
- `tokens_per_1k_chars`: `445.19`
- `unk_token_rate`: `0.0`

Short example encodings (from the trained tokenizer):

- Text: "flooding storm surge" → Tokens: `[BOS] f lo o d ing storm s ur ge [EOS]`
- Text: "mandatory evacuations for low-lying areas" → Tokens: `[BOS] m and at or y ev ac u ations for low - ly ing are as [EOS]`
- Text: "data assimilation improves genesis forecast" → Tokens: `[BOS] d at a assimil ation im pro v es genesis forecast [EOS]`

These metrics provide evidence that:

- the tokenizer is actually being applied to the ingested PDF text,
- the corpus is non-trivial (91 chunks), and
- the trained BPE vocabulary covers the corpus well (0% `[UNK]` rate on the diagnostic sample).

## Retrieval (BM25 demo)

**Goal**: implement a minimal retrieval component that resembles a RAG retriever (retrieve relevant chunks for a query).

Implementation:

- Index + scoring: `training_code/retrieval/bm25.py` (pure Python BM25)
- Demo runner: `training_code/retrieval/run_retrieval_demo.py`
- Output: `results/retrieval_demo.json`

The retriever operates over the same JSONL chunks as the tokenizer/model. Each returned hit includes:

- the retrieved `text` chunk
- the BM25 `score`
- metadata `meta` containing `source`, `page`, and `chunk_id`

Example query set (from the demo):

- “flooding storm surge”
- “evacuation orders”
- “emergency kits water batteries”

Reproducible command used:

```bash
.venv_py39/bin/python training_code/retrieval/run_retrieval_demo.py
```

## (3) Model skeleton and data pipeline smoke test (required)

Which type of model are we using?

- Trainable ML model in the smoke test: a tiny decoder-only Transformer language model (`MiniGPT`) trained with next-token prediction.

This section is the required end-to-end evidence that the implementation works.

Script: `training_code/smoke_test.py`

### 3.1 Data and loader

Dataset or corpus:

- Preferred: `data/corpus.jsonl` (PDF-derived corpus).
- Fallback: `data/hurricane_data.jsonl` (toy corpus).
- Selection rule: the loader reads JSONL rows and uses the `text` field when non-empty.

Train split used for the smoke test:

- Preferred: use explicit reproducible split files under `data/splits/` (seed=42).
- Fallback: document-level split of JSONL rows using a fixed seed (`seed=42`).
- Split ratio (fallback): 80% train / 10% validation / 10% test.

Loader details:

- Input format: `jsonl` (each line is a JSON dict containing `{"text": "..."}`).
- Context length (T): 64.
- Batch size (B): 8.
- Packing or chunking:
  - Encode the selected documents into a single long token-id stream.
  - Cap training tokens to avoid memory issues on large PDFs (`max_train_tokens = 200,000`).
  - If the corpus is tiny, repeat the token stream so the loop can run enough steps (`repeat_corpus = 200`, capped).
  - Slice the stream into fixed examples of length `T+1`; create next-token pairs.
- Output tensors:
  - `input_ids`: shape `[B, T]` (tokens 0..T-1)
  - `labels`: shape `[B, T]` (tokens 1..T, i.e., next-token labels)

### 3.2 Model skeleton

Describe the smallest reasonable model implemented:

- Architecture: small decoder-only Transformer LM (`MiniGPT`) implemented with:
  - learned token embedding + learned position embedding
  - 2 stacked Transformer blocks (PyTorch `nn.TransformerEncoderLayer`)
  - final LayerNorm
  - linear language-model head to vocab logits
- Layers (`n_layer`): 2
- Heads (`n_head`): 4
- Hidden size (`d_model`): 128
- Dropout: 0.1
- Masking: causal attention mask (future tokens are masked via an upper-triangular `-inf` mask)
- Parameter count (recommended): 470,528 parameters (with vocab size 256)

### 3.3 Minimal training loop (smoke test)

- Objective: next-token cross entropy loss.
- Optimizer: AdamW.
- Learning rate: 3e-4.
- Steps: 80 (evaluation every 10 steps, using a small fixed number of validation batches).
- Hardware: CPU by default; uses CUDA automatically if available.

### 3.4 Evidence that the smoke test worked

At minimum:

- The run completes without errors (the script raises an exception if loss becomes non-finite).
- The loss is finite.
- The loss is not exploding.

Saved evidence under `results/`:

- `results/loss_log.txt` (tab-separated columns: `step`, `train_loss`, `val_loss`, `test_loss`)
- `results/loss_plot.png`

Sanity evidence from `results/loss_log.txt` (latest run):

- Step 0: train_loss=5.7216, val_loss=5.7109, test_loss=5.6910
- Step 79: train_loss=4.9590, val_loss=4.9196, test_loss=5.0266

### 3.5 What worked and what did not work yet

Worked:

- JSONL loading tokenization  batching into `[B, T]` tensors  causal Transformer forward/backward  finite loss.
- Train/val loss logging to disk and plotting to `results/loss_plot.png`.

Known issues or limitations:

- This is intentionally a smoke test (tiny model, short context length, few steps) intended to validate the pipeline, not maximize LM quality.
- Retrieval (BM25) is demonstrated separately; the milestone does not yet include a full generator conditioned on retrieved passages.
- PDF extraction is text-based (no OCR). Scanned PDFs would require an OCR fallback.

Reproducible command used:

```bash
.venv_py39/bin/python training_code/smoke_test.py
```

## Reproducibility / outputs checklist

Outputs used as Milestone 2 evidence (all produced from the PDF-derived corpus):

- Tokenizer artifacts: `training_code/tokenizer/*`
- Tokenizer diagnostics: `results/tokenizer_diagnostics.json`
- Retrieval demo output: `results/retrieval_demo.json`
- Training log + plot: `results/loss_log.txt`, `results/loss_plot.png`

If `data/corpus.jsonl` is missing, scripts fall back to the toy dataset `data/hurricane_data.jsonl`, but the intended submission path is the PDF-derived corpus.
