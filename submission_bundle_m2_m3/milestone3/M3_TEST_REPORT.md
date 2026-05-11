# Milestone 3 Test Report

Date: 2026-05-10

## Scope
Validation run for the Milestone 3 packaged pipeline.

## Commands executed

1. `code/training_code/retrieval/run_retrieval_demo.py`
2. `code/training_code/retrieval/evaluate_retrieval.py`
3. `code/training_code/smoke_test.py`
4. `code/training_code/base_vs_peft.py`
5. `code/training_code/tokenizer_diagnostics.py`

## Results

| Test | Status | Key Output |
|---|---|---|
| Retrieval demo | PASS | `code/results/retrieval_demo.json` written |
| Retrieval metrics | PASS | `code/results/retrieval_eval.json` and `code/results/retrieval_eval.md` written |
| Smoke training | PASS | `code/results/loss_log.txt` and `code/results/loss_plot.png` written |
| Base vs LoRA comparison | PASS | `code/results/base_vs_adapted.json` and `code/results/base_vs_adapted.md` written |
| Tokenizer diagnostics | PASS | diagnostics execution completed and output available in `code/results` |

## Retrieval snapshot

| Setting | Recall@5 | HitRate@5 | MRR |
|---|---:|---:|---:|
| BM25 | 1.0000 | 1.0000 | 0.8333 |

## Base vs LoRA snapshot

| System | Test Loss | Test Perplexity |
|---|---:|---:|
| Base | 4.8625 | 129.3472 |
| Adapted (LoRA) | 5.3904 | 219.2973 |

## Conclusion
Milestone 3 core evaluation pipeline ran successfully end-to-end in the submission layout.

## Agent / Orchestration applicability note (Track C)

This project is approved under Track C (Retrieval / RAG focus).

- A full autonomous agent loop with tool-trace logs was **not** implemented as a core project requirement.
- Therefore, agent-loop traces are treated as **not applicable** for this submission.
- Orchestration evidence is still present through the dashboard pipeline runner and execution logs in:
	- `code/training_code/m3_dashboard.py`

This satisfies the Track C expectation of a coherent orchestrated workflow without claiming an autonomous agent-loop system.
