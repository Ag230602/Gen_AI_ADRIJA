# StormCare RAG — Milestone 3 Final Submission

This package provides a complete, reproducible end-to-end workflow:
- PDF preprocessing to JSONL corpus
- custom tokenizer training + diagnostics
- retrieval demo (BM25)
- fair quantitative base-vs-adapted (LoRA) comparison
- dashboard orchestration and submission ZIP generation

---

## What file runs the dashboard

- `code/training_code/m3_dashboard.py`

Run command:

```bash
.venv/bin/streamlit run code/training_code/m3_dashboard.py
```

Local dashboard URL after launch:
- `http://localhost:8501`

---

## Environment setup

```bash
/usr/bin/python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt
```

## Reproducible run commands (from this folder)

1. Build corpus from PDFs (preferred path)

```bash
.venv/bin/python code/training_code/pdf_to_jsonl.py --mode paragraph --min_chars 200
```

2. Train tokenizer

```bash
.venv/bin/python code/training_code/train_tokenizer.py
```

3. Tokenizer diagnostics

```bash
.venv/bin/python code/training_code/tokenizer_diagnostics.py
```

4. Retrieval demo

```bash
.venv/bin/python code/training_code/retrieval/run_retrieval_demo.py
```

5. Base vs adapted (LoRA) comparison

```bash
.venv/bin/python code/training_code/base_vs_peft.py
.venv/bin/python code/training_code/generate_m3_plots.py
```

## Optional dashboard mode (single-file orchestration)

```bash
.venv/bin/streamlit run code/training_code/m3_dashboard.py
```

The dashboard runs all pipeline steps, checks required artifacts, and shows key metrics.
It also generates the final required ZIP and provides a direct download button.

---

## Access map (where everything is)

### A) Core code

- Pipeline scripts: `code/training_code/`
- Retrieval module: `code/training_code/retrieval/`
- Tokenizer artifacts: `code/training_code/tokenizer/`

### B) Data

- Main corpus: `code/data/corpus.jsonl`
- Fallback corpus: `code/data/hurricane_data.jsonl`
- Raw PDFs: `code/data/pdfs/`

### C) Results (primary submission location)

- Primary results folder: `code/results/`

Expected files:
- `code/results/tokenizer_diagnostics.json`
- `code/results/retrieval_demo.json`
- `code/results/loss_log.txt`
- `code/results/loss_plot.png`
- `code/results/base_vs_adapted.json`
- `code/results/base_vs_adapted.md`
- `code/results/base_vs_adapted_bar.png`
- `code/results/retrieval_top1_scores.png`
- `code/results/training_loss_trend.png`

### D) Documentation

- Final submission README: `README.md`
- Milestone 3 integration guide: `README_M3.md`
- M2+M3 paper guide: `M2_M3_RESULTS_AND_PAPER_GUIDE.md`
- Milestone 3 test report: `M3_TEST_REPORT.md`
- Submission checklist: `SUBMISSION_CHECKLIST.md`

---

## Script-by-script usage

1. Build corpus from PDFs

```bash
.venv/bin/python code/training_code/pdf_to_jsonl.py --mode paragraph --min_chars 200
```

2. Train tokenizer

```bash
.venv/bin/python code/training_code/train_tokenizer.py
```

3. Tokenizer diagnostics

```bash
.venv/bin/python code/training_code/tokenizer_diagnostics.py
```

4. Retrieval demo

```bash
.venv/bin/python code/training_code/retrieval/run_retrieval_demo.py
```

5. Smoke test training

```bash
.venv/bin/python code/training_code/smoke_test.py
```

6. Base vs adapted (LoRA)

```bash
.venv/bin/python code/training_code/base_vs_peft.py
```

7. Plot generation

```bash
.venv/bin/python code/training_code/generate_m3_plots.py
```

8. Dashboard mode

```bash
.venv/bin/streamlit run code/training_code/m3_dashboard.py
```

---

## Evaluation and fairness summary

Base-vs-adapted comparison uses:
- same task: next-token language modeling
- same held-out split
- same metrics: test loss and perplexity
- same evaluation procedure
- shared initialization and fixed seed (`seed=42`)

Reference metrics are in:
- `code/results/base_vs_adapted.json`
- `code/results/base_vs_adapted.md`

---

## Test status

Executable validation report:
- `M3_TEST_REPORT.md`

Combined M2+M3 summary for writing:
- `M2_M3_RESULTS_AND_PAPER_GUIDE.md`

## Hardware assumptions

- Runs on CPU; CUDA is used automatically if available.
- Designed as a compact course-project pipeline, not a large-scale training setup.

---

## Final ZIP requirements

Required files for final submission zip:
- `paper.pdf`
- `demo_video.mp4` or `demo_link.txt`
- `code/`
- `results/`
- `README.md`
- `requirements.txt`

After adding paper/demo files, build the ZIP with:

```bash
./make_zip.sh <LastName> <FirstName>
```
