# Milestone 2 submission contents

This folder is intended to be zipped and submitted.

## Required deliverables

### Tokenizer and vocabulary
- `training_code/train_tokenizer.py`
- `training_code/tokenizer_diagnostics.py`
- `training_code/tokenizer/`
  - `tokenizer.json`
  - `vocab.json`
  - `merges.txt`
  - `tokenizer_meta.json`

### Model skeleton + data pipeline smoke test
- `training_code/smoke_test.py`
- `results/loss_log.txt`
- `results/loss_plot.png`

## Retrieval (Track C)
- `training_code/retrieval/bm25.py`
- `training_code/retrieval/run_retrieval_demo.py`
- `results/retrieval_demo.json`

## Reproducibility + report
- `README.md`
- `requirements.txt`
- `METHODS.md` (report-ready methods text)
- `report.pdf`

## Data
- `data/hurricane_data.jsonl` (fallback toy corpus)
- `data/pdfs/` (place your 2 PDFs here)
- `training_code/pdf_to_jsonl.py` (builds `data/corpus.jsonl` from PDFs)

Optional (recommended for clarity)
- `training_code/make_splits.py` (creates reproducible shuffled train/val/test splits)
- `data/splits/`
  - `train.jsonl`
  - `val.jsonl`
  - `test.jsonl`
