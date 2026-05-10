# Milestone 3 — Final Integration Guide (StormCare RAG)

## New integration artifacts

- `training_code/base_vs_peft.py`
- `results/base_vs_adapted.json`
- `results/base_vs_adapted.md`

## Run command

```bash
.venv_py39/bin/python training_code/base_vs_peft.py
```

## What this script does

- trains a base system and a LoRA-adapted system
- uses the same held-out split and same evaluation metrics
- reports test loss and perplexity
- reports trainable vs total parameters
- logs LoRA settings (target modules, rank, alpha, dropout)

## Fair comparison controls

- same task definition
- same tokenizer
- same train/val/test split
- same seed and initialization
- same training/evaluation procedure
