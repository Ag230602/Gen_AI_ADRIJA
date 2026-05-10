# Combined Milestone 2 + Milestone 3 Submission Content

This document combines what matters from both milestones for final delivery.

## Folder layout

- `milestone2/` -> Milestone 2 package (methods, tokenizer/retrieval/smoke-test evidence)
- `milestone3/` -> Milestone 3 final package (required final ZIP source)

---

## What from Milestone 2 is retained (supporting evidence)

Use `milestone2/` as historical/technical evidence in paper and appendix:
- tokenizer diagnostics (`milestone2/results/tokenizer_diagnostics.json`)
- retrieval demo output (`milestone2/results/retrieval_demo.json`)
- smoke-test logs/plot (`milestone2/results/loss_log.txt`, `milestone2/results/loss_plot.png`)
- methods writeup (`milestone2/METHODS.md`)

These support your story of system build-up.

---

## What from Milestone 3 is required for final submission

Your **final course submission ZIP** should be built from `milestone3/` and include:
- `paper.pdf`
- `demo_video.mp4` or `demo_link.txt`
- `code/`
- `results/`
- `README.md`
- `requirements.txt`

Milestone 3 key evidence includes:
- base-vs-adapted metrics (`milestone3/results/base_vs_adapted.json`, `.md`)
- retrieval/tokenizer outputs (`milestone3/results/retrieval_demo.json`, `tokenizer_diagnostics.json`)
- plots/logs (`milestone3/results/loss_plot.png`, `loss_log.txt`, plus generated summary plots)
- validation summary (`milestone3/M3_TEST_REPORT.md`)
- integrated writing guide (`milestone3/M2_M3_RESULTS_AND_PAPER_GUIDE.md`)

---

## Final packaging steps

1. Put `paper.pdf` into `milestone3/`
2. Put either `demo_video.mp4` or `demo_link.txt` into `milestone3/`
3. Build zip from `milestone3/`:

```bash
./make_zip.sh <LastName> <FirstName>
```

Output name:

`CS5590_Grad_M3_<LastName>_<FirstName>.zip`

---

## Pre-submit checklist

- [ ] `paper.pdf` added
- [ ] demo file added (`demo_video.mp4` or `demo_link.txt`)
- [ ] `code/` present
- [ ] `results/` present
- [ ] `README.md` present
- [ ] ZIP name format correct
- [ ] ZIP opens and contains expected files
