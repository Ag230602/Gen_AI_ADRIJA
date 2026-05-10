# Milestone 3 Test Report

Date: 2026-05-10

## Scope
Validation run for the Milestone 3 packaged pipeline.

## Commands executed

1. `code/training_code/retrieval/run_retrieval_demo.py`
2. `code/training_code/smoke_test.py`
3. `code/training_code/base_vs_peft.py`
4. `code/training_code/tokenizer_diagnostics.py`

## Results

| Test | Status | Key Output |
|---|---|---|
| Retrieval demo | PASS | `submission_m3/code/results/retrieval_demo.json` written |
| Smoke training | PASS | `submission_m3/code/results/loss_log.txt` and `loss_plot.png` written |
| Base vs LoRA comparison | PASS | `submission_m3/code/results/base_vs_adapted.json` and `.md` written |
| Tokenizer diagnostics | PASS | diagnostics execution completed and output available in `code/results` |

## Base vs LoRA snapshot

| System | Test Loss | Test Perplexity |
|---|---:|---:|
| Base | 4.8625 | 129.3472 |
| Adapted (LoRA) | 5.3904 | 219.2973 |

## Conclusion
Milestone 3 core evaluation pipeline ran successfully end-to-end in the submission layout.
