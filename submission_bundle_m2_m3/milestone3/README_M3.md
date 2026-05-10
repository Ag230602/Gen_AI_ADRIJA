# Milestone 3 — Final Integration Guide (StormCare RAG)

This document maps your current repository to Milestone 3 requirements and gives reproducible commands for final delivery.

## 1) End-to-End System (required)

Run the full workflow from raw PDFs to outputs:

1. Build corpus from PDFs
```bash
.venv_py39/bin/python training_code/pdf_to_jsonl.py --mode paragraph --min_chars 200
```

2. Train tokenizer
```bash
.venv_py39/bin/python training_code/train_tokenizer.py
```

3. Tokenizer diagnostics
```bash
.venv_py39/bin/python training_code/tokenizer_diagnostics.py
```

4. Retrieval demo
```bash
.venv_py39/bin/python training_code/retrieval/run_retrieval_demo.py
```

5. Train/evaluate base vs adapted (LoRA)
```bash
.venv_py39/bin/python training_code/base_vs_peft.py
```

Core outputs:
- `results/tokenizer_diagnostics.json`
- `results/retrieval_demo.json`
- `results/base_vs_adapted.json`
- `results/base_vs_adapted.md`

## 2) PEFT / LoRA comparison (required)

Implemented in `training_code/base_vs_peft.py`.

The script enforces fairness:
- same task: next-token LM
- same train/val/test split
- same held-out test set
- same metrics (`test_loss`, perplexity)
- same seed and initialization

Reported details:
- base model name/size
- tokenizer path
- LoRA target modules, rank, alpha, dropout
- optimizer, learning rate, batch size, steps
- trainable vs total parameters
- comparison table in `results/base_vs_adapted.md`

## 3) Retrieval component (if required for your approved project)

Current retriever: BM25 in `training_code/retrieval/bm25.py`.

Demo evidence:
- `results/retrieval_demo.json`

For final grading, add quantitative retrieval metrics (`Recall@k`, `MRR`) if retrieval is part of your approved scope.

## 4) README requirements (required)

Your final `README.md` should include:
- project overview
- setup steps and environment assumptions
- exact run commands (above)
- where each artifact is written
- how to reproduce demo
- hardware limits (CPU/GPU, runtime)

## 5) Paper requirements (required)

Use your conference structure exactly:
- Title, Abstract, Intro, Related Work, Data, Method, Experimental Setup, Results, Analysis, Limitations/Responsible Use, Conclusion, References

Recommended tables/figures from this repo:
- Base vs Adapted table from `results/base_vs_adapted.md`
- Loss curve from `results/loss_plot.png`
- Retrieval examples from `results/retrieval_demo.json`

## 6) Demo (required, 5 minutes)

Suggested timeline:
1. Problem + task (45s)
2. System architecture (45s)
3. End-to-end run (90s)
4. Base vs adapted quantitative results (75s)
5. One failure case + takeaway (45s)

## 7) Final ZIP structure

Name:
`CS5590_Grad_M3_<LastName>_<FirstName>.zip`

Must include:
- `paper.pdf`
- `demo_video.mp4` or `demo_link.txt`
- `code/` (training/eval/adaptation/retrieval)
- `results/` (plots, metrics, logs, outputs)
- `README.md`
