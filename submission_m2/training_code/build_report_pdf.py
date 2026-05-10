"""Build a report-ready PDF for Milestone 2.

This script generates `report.pdf` inside the `submission_m2/` bundle using the
already-produced artifacts under `results/` and `training_code/tokenizer/`.

Run (from repository root):
  /Users/agd9c/.local/bin/python3.14 submission_m2/training_code/build_report_pdf.py

Or (from submission_m2):
  /Users/agd9c/.local/bin/python3.14 training_code/build_report_pdf.py

Outputs:
  submission_m2/report.pdf (default)

Notes
- The report summarizes *your* pipeline and outputs without embedding large
  excerpts of the source PDFs.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _try_read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    return _read_json(path)


def _read_loss_log(path: str, max_rows: int = 12, max_cols: int = 4) -> Tuple[List[str], List[List[str]]]:
    """Return (header, rows) where rows are string cells for a small table."""
    if not os.path.exists(path):
        return ["step", "train_loss", "val_loss", "test_loss"][:max_cols], []

    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        return ["step", "train_loss", "val_loss", "test_loss"][:max_cols], []

    header = lines[0].split("\t")
    data = [ln.split("\t") for ln in lines[1:]]

    # Take a few early and late rows for evidence.
    if len(data) <= max_rows:
        keep = data
    else:
        k1 = max_rows // 2
        k2 = max_rows - k1
        keep = data[:k1] + data[-k2:]

    # Normalize to a fixed number of columns.
    keep_norm: List[List[str]] = []
    for row in keep:
        row = (row + [""] * max_cols)[:max_cols]
        keep_norm.append(row)

    header = (header + [""] * max_cols)[:max_cols]
    return header, keep_norm


def _summarize_loss_log(path: str) -> Dict[str, Any]:
    """Return a small summary (start/end rows) for narrative text."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 2:
        return {}

    header = lines[0].split("\t")
    rows = [ln.split("\t") for ln in lines[1:]]
    rows = [r for r in rows if len(r) >= 3]
    if not rows:
        return {}

    def row_to_dict(r: List[str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in zip(header[:4], r[:4]):
            out[k] = v
        return out

    return {
        "n_rows": len(rows),
        "start": row_to_dict(rows[0]),
        "end": row_to_dict(rows[-1]),
    }


def _read_requirements(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    reqs: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            reqs.append(line)
    return reqs


def _list_pdfs(pdf_dir: str) -> List[str]:
    if not os.path.isdir(pdf_dir):
        return []
    pdfs = [p for p in os.listdir(pdf_dir) if p.lower().endswith(".pdf")]
    pdfs.sort()
    return pdfs


def build_pdf(out_pdf: str, submission_root: str) -> None:
    # Lazy import so the script can be imported without reportlab.
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    results_dir = os.path.join(submission_root, "results")
    training_code_dir = os.path.join(submission_root, "training_code")
    tokenizer_dir = os.path.join(training_code_dir, "tokenizer")
    data_dir = os.path.join(submission_root, "data")
    requirements_path = os.path.join(submission_root, "requirements.txt")

    tokenizer_meta_path = os.path.join(tokenizer_dir, "tokenizer_meta.json")
    tok_diag_path = os.path.join(results_dir, "tokenizer_diagnostics.json")
    retrieval_demo_path = os.path.join(results_dir, "retrieval_demo.json")
    loss_log_path = os.path.join(results_dir, "loss_log.txt")
    loss_plot_path = os.path.join(results_dir, "loss_plot.png")

    tok_meta = _try_read_json(tokenizer_meta_path) or {}
    tok_diag = _try_read_json(tok_diag_path) or {}
    retrieval = _try_read_json(retrieval_demo_path) or {}

    loss_header, loss_rows = _read_loss_log(loss_log_path)
    loss_summary = _summarize_loss_log(loss_log_path)
    requirements = _read_requirements(requirements_path)

    pdf_dir = os.path.join(data_dir, "pdfs")
    pdf_files = _list_pdfs(pdf_dir)

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="H1",
            parent=styles["Heading1"],
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2",
            parent=styles["Heading2"],
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            leading=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Mono",
            parent=styles["BodyText"],
            fontName="Courier",
            fontSize=9,
            leading=11,
        )
    )

    doc = SimpleDocTemplate(
        out_pdf,
        pagesize=letter,
        rightMargin=0.9 * inch,
        leftMargin=0.9 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.85 * inch,
        title="Milestone 2 Report",
        author="(student)",
    )

    story: List[Any] = []

    # Title
    story.append(Paragraph("Milestone 2 — Implementation Foundation Report", styles["H1"]))
    story.append(
        Paragraph(
            f"Project: <b>StormCare RAG</b> (Track C: Retrieval system / RAG foundation) • Date: {date.today().isoformat()}",
            styles["Body"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    # Executive summary
    story.append(Paragraph("1. Executive Summary", styles["H2"]))
    story.append(
        Paragraph(
            "This milestone delivers the end-to-end implementation foundation for a retrieval-augmented generation (RAG) style project. "
            "The submission includes: (1) a trained tokenizer with saved artifacts and diagnostics, (2) a minimal retriever demo (BM25) over PDF-derived paragraph chunks, "
            "and (3) a fully trainable smoke test that loads the corpus, batches token sequences, runs a small causal Transformer, and logs finite train/validation loss.",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "The goal is not to maximize model quality yet; it is to prove the pipeline is real, correct, and reproducible: PDF ingestion → JSONL corpus → tokenizer → batching → model forward/backward → loss logging.",
            styles["Body"],
        )
    )

    # Data
    story.append(Paragraph("2. Data and Preprocessing", styles["H2"]))
    story.append(
        Paragraph(
            "<b>Primary source</b>: two PDF documents placed under <font face='Courier'>data/pdfs/</font>. "
            "A conversion script extracts text using pypdf and writes a JSONL corpus (<font face='Courier'>data/corpus.jsonl</font>) with one row per paragraph chunk.",
            styles["Body"],
        )
    )

    if pdf_files:
        pdf_list = ", ".join([f"<font face='Courier'>{p}</font>" for p in pdf_files])
        story.append(
            Paragraph(
                f"<b>PDFs included in this submission</b>: {pdf_list}.",
                styles["Body"],
            )
        )
        story.append(
            Paragraph(
                "These PDFs were selected because they are relevant to hurricanes/storm impacts and contain enough domain-specific terminology to exercise tokenization, paragraph chunking, and retrieval.",
                styles["Body"],
            )
        )
    story.append(
        Paragraph(
            "<b>Chunking</b>: paragraph-based splitting (blank-line separated), with a minimum character threshold (default 200) to filter headers/footers and very short fragments. "
            "Each chunk includes metadata fields: <font face='Courier'>source</font>, <font face='Courier'>page</font>, and <font face='Courier'>chunk_id</font>.",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "<b>Corpus format</b>: JSONL with one dict per line, containing at least <font face='Courier'>text</font> and (in the PDF-derived corpus) metadata fields. This format keeps the pipeline simple and auditable.",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "<b>Fallback data</b>: if <font face='Courier'>data/corpus.jsonl</font> is not present, scripts fall back to a small toy corpus <font face='Courier'>data/hurricane_data.jsonl</font> "
            "to keep the pipeline runnable from a clean environment.",
            styles["Body"],
        )
    )

    # Optional splits
    splits_dir = os.path.join(data_dir, "splits")
    if os.path.exists(os.path.join(splits_dir, "train.jsonl")):
        story.append(
            Paragraph(
                "<b>Reproducible splits</b>: a deterministic shuffle + train/val/test split (seed=42) is provided under <font face='Courier'>data/splits/</font>. "
                "This is optional for the milestone, but improves clarity and repeatability.",
                styles["Body"],
            )
        )

    story.append(Paragraph("2.1 Artifacts Produced", styles["H2"]))
    artifacts_table = Table(
        [
            ["Artifact", "Purpose"],
            ["data/corpus.jsonl", "PDF-derived paragraph corpus (preferred input for all components)"],
            ["training_code/tokenizer/tokenizer.json", "Trained tokenizer configuration"],
            ["training_code/tokenizer/vocab.json + merges.txt", "BPE model files"],
            ["results/tokenizer_diagnostics.json", "Tokenizer evidence metrics (tokens/doc, UNK rate, examples)"],
            ["results/retrieval_demo.json", "BM25 retrieval demo output"],
            ["results/loss_log.txt", "Training/validation losses from smoke test"],
            ["results/loss_plot.png", "Loss plot saved from smoke test"],
        ],
        hAlign="LEFT",
        colWidths=[2.3 * inch, 3.8 * inch],
    )
    artifacts_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(artifacts_table)

    # Tokenizer
    story.append(Paragraph("3. Tokenizer and Vocabulary", styles["H2"]))
    algo = tok_meta.get("algorithm", "BPE")
    vocab_size = tok_meta.get("actual_vocab_size", tok_meta.get("requested_vocab_size", "(unknown)"))
    normalization = ", ".join(tok_meta.get("normalization", [])) or "NFKC, lowercase"
    pre_tok = tok_meta.get("pre_tokenizer", "Whitespace")
    special = tok_meta.get("special_tokens", ["[PAD]", "[UNK]", "[BOS]", "[EOS]"])

    story.append(
        Paragraph(
            f"Tokenizer is trained from scratch using HuggingFace <b>tokenizers</b> with <b>{algo}</b>. "
            f"Normalization: <b>{normalization}</b>. Pre-tokenization: <b>{pre_tok}</b>. "
            f"Special tokens: <font face='Courier'>{' '.join(special)}</font>. "
            f"Vocabulary size: <b>{vocab_size}</b>.",
            styles["Body"],
        )
    )

    story.append(
        Paragraph(
            "<b>Vocabulary size justification (256)</b>: the smoke test is designed to run quickly on CPU, so a small vocabulary keeps the embedding table and softmax efficient. "
            "A smaller vocab can increase sequence length, but the smoke test uses short context length (T=64), so training remains stable. "
            "For later milestones, the vocab size can be increased (e.g., 8k–32k) once the full training setup is in place.",
            styles["Body"],
        )
    )

    # Tokenizer diagnostics
    if tok_diag:
        story.append(Paragraph("3.1 Tokenizer Diagnostics (Evidence)", styles["H2"]))
        n_docs = tok_diag.get("n_docs", "(unknown)")
        avg_toks = tok_diag.get("avg_tokens_per_doc", "(unknown)")
        tpk = tok_diag.get("tokens_per_1k_chars", "(unknown)")
        unk = tok_diag.get("unk_token_rate", "(unknown)")
        story.append(
            Paragraph(
                f"Diagnostics computed on up to {tok_diag.get('max_docs', '(unknown)')} documents. "
                f"Observed: <b>n_docs={n_docs}</b>, <b>avg_tokens/doc={avg_toks:.2f}</b>, <b>tokens/1k chars={tpk:.2f}</b>, <b>UNK rate={unk:.4f}</b>.",
                styles["Body"],
            )
        )

        ex = tok_diag.get("examples", [])
        if isinstance(ex, list) and ex:
            story.append(
                Paragraph(
                    "Example encodings (first 1–2 samples from diagnostics):", styles["Body"]
                )
            )
            for sample in ex[:2]:
                text = (sample.get("text", "") or "").strip()
                toks = sample.get("tokens", [])
                text_preview = (text[:180] + "…") if len(text) > 180 else text
                toks_preview = " ".join([str(t) for t in toks[:40]]) + (" …" if len(toks) > 40 else "")
                story.append(
                    Paragraph(
                        f"Text preview: {text_preview}",
                        styles["Body"],
                    )
                )
                story.append(
                    Paragraph(
                        f"Tokens preview: <font face='Courier'>{toks_preview}</font>",
                        styles["Body"],
                    )
                )

    # Retrieval
    story.append(Paragraph("4. Retrieval Component (Track C)", styles["H2"]))
    story.append(
        Paragraph(
            "A minimal retriever is implemented using BM25 scoring over the same paragraph-chunk corpus used for tokenization and training. "
            "This provides a RAG-like foundation: a query maps to top-k relevant chunks, each with (source, page, chunk_id) metadata.",
            styles["Body"],
        )
    )

    if retrieval and retrieval.get("queries"):
        q0 = retrieval["queries"][0]
        top0 = q0.get("top_k", [])
        if top0:
            hit = top0[0]
            meta = hit.get("meta", {})
            story.append(
                Paragraph(
                    "Example retrieval evidence (from the saved demo output):", styles["Body"]
                )
            )
            story.append(
                Paragraph(
                    f"Query: <font face='Courier'>{q0.get('query','')}</font><br/>"
                    f"Top hit: source=<font face='Courier'>{meta.get('source','')}</font>, page={meta.get('page','')}, chunk_id={meta.get('chunk_id','')}, score={hit.get('score','')}",
                    styles["Body"],
                )
            )

    # Model + training
    story.append(PageBreak())
    story.append(Paragraph("5. Model Skeleton and Data Pipeline Smoke Test", styles["H1"]))

    story.append(Paragraph("5.1 Data Loader", styles["H2"]))
    story.append(
        Paragraph(
            "The smoke test reads the JSONL corpus, tokenizes each document using the trained tokenizer, then concatenates tokens into a long stream and slices fixed-length blocks. "
            "Each training example uses a (T+1)-token window to build next-token labels.",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "<b>Input</b>: JSONL rows with a <font face='Courier'>text</font> field. "
            "<b>Context length</b>: T=64. <b>Batch size</b>: B=8. "
            "<b>Tensors</b>: <font face='Courier'>input_ids</font> shape [B, T], <font face='Courier'>labels</font> shape [B, T] (shifted by one token for next-token prediction).",
            styles["Body"],
        )
    )

    story.append(Paragraph("5.2 Model Skeleton", styles["H2"]))
    story.append(
        Paragraph(
            "The model is a small decoder-only Transformer language model (named <b>MiniGPT</b> in code). "
            "It uses token embeddings + position embeddings, 2 Transformer blocks, LayerNorm, and a linear LM head. "
            "Causal masking prevents attention to future tokens.",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "Configuration: n_layer=2, n_head=4, d_model=128, dropout=0.1, vocab_size=256. "
            "Reported parameter count is approximately 470,528 parameters.",
            styles["Body"],
        )
    )

    story.append(Paragraph("5.3 Minimal Training Loop", styles["H2"]))
    story.append(
        Paragraph(
            "Objective: next-token cross-entropy loss. Optimizer: AdamW. Learning rate: 3e-4. Steps: 80. "
            "Hardware: CPU by default (uses CUDA automatically if available). Gradients are clipped (max_norm=1.0) for stability.",
            styles["Body"],
        )
    )

    story.append(Paragraph("5.4 Evidence of Correct Execution", styles["H2"]))
    story.append(
        Paragraph(
            "The smoke test writes a tab-separated loss log and a loss plot. Loss is checked for non-finite values and the run aborts if loss becomes NaN/Inf.",
            styles["Body"],
        )
    )

    # Loss table
    table_data: List[List[str]] = [loss_header] + loss_rows
    # Support either 3 columns (step/train/val) or 4 columns (step/train/val/test).
    if len(loss_header) >= 4:
        col_widths = [0.9 * inch, 1.25 * inch, 1.25 * inch, 1.25 * inch]
    else:
        col_widths = [1.0 * inch, 1.5 * inch, 1.5 * inch]
    loss_table = Table(table_data, hAlign="LEFT", colWidths=col_widths)
    loss_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("Loss excerpts from results/loss_log.txt:", styles["Body"]))
    story.append(loss_table)

    if loss_summary:
        s = loss_summary.get("start", {})
        e = loss_summary.get("end", {})
        story.append(Spacer(1, 0.12 * inch))
        story.append(
            Paragraph(
                f"Summary: start step={s.get('step','?')} train={s.get('train_loss','?')} val={s.get('val_loss','?')} → "
                f"end step={e.get('step','?')} train={e.get('train_loss','?')} val={e.get('val_loss','?')}.",
                styles["Body"],
            )
        )

    if os.path.exists(loss_plot_path):
        story.append(Spacer(1, 0.15 * inch))
        story.append(Paragraph("Loss plot (saved artifact):", styles["Body"]))
        img = Image(loss_plot_path)
        max_w, max_h = 6.0 * inch, 3.2 * inch
        iw, ih = float(getattr(img, "imageWidth", max_w)), float(getattr(img, "imageHeight", max_h))
        scale = min(max_w / max(iw, 1.0), max_h / max(ih, 1.0))
        img.drawWidth = iw * scale
        img.drawHeight = ih * scale
        story.append(img)

    # Limitations & future
    story.append(PageBreak())
    story.append(Paragraph("6. Limitations", styles["H2"]))
    story.append(
        Paragraph(
            "- This milestone uses BM25 retrieval only; it does not yet include a generator that conditions on retrieved passages. "
            "<br/>- PDF extraction is text-based (no OCR); scanned PDFs would need OCR fallback. "
            "<br/>- The tokenizer vocabulary size (256) and model size are intentionally small to keep the smoke test fast and stable, not to maximize quality.",
            styles["Body"],
        )
    )

    story.append(Paragraph("7. Future Work / Milestone 3+ Plan", styles["H2"]))
    story.append(
        Paragraph(
            "Next steps to turn this foundation into a full RAG system:",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "- Upgrade retrieval (e.g., dense embeddings) and build an index that supports faster querying and better semantic recall. "
            "<br/>- Add a generator model that consumes retrieved chunks (prompting baseline first, then fine-tuning or lightweight adaptation if applicable). "
            "<br/>- Increase context length and vocabulary size once training infrastructure is stable. "
            "<br/>- Add more systematic evaluation: retrieval metrics (Recall@k), generation metrics (faithfulness/attribution), and ablations (with/without retrieval).",
            styles["Body"],
        )
    )

    story.append(Paragraph("8. Reproducibility Notes", styles["H2"]))
    story.append(
        Paragraph(
            "The submission includes runnable scripts and fixed seeds. Key commands:",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "<font face='Courier'>python training_code/pdf_to_jsonl.py --mode paragraph --min_chars 200</font><br/>"
            "<font face='Courier'>python training_code/train_tokenizer.py</font><br/>"
            "<font face='Courier'>python training_code/tokenizer_diagnostics.py</font><br/>"
            "<font face='Courier'>python training_code/smoke_test.py</font><br/>"
            "<font face='Courier'>python training_code/retrieval/run_retrieval_demo.py</font><br/>"
            "<font face='Courier'>python training_code/make_splits.py</font>",
            styles["Mono"],
        )
    )

    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("8.1 Why requirements.txt is included", styles["H2"]))
    story.append(
        Paragraph(
            "<font face='Courier'>requirements.txt</font> pins the Python dependencies needed to reproduce the tokenizer training, retrieval demo, and end-to-end smoke test from a fresh environment. "
            "It allows the grader to run <font face='Courier'>pip install -r requirements.txt</font> and get the same library stack.",
            styles["Body"],
        )
    )

    if requirements:
        req_rows = [["Dependency"], *[[r] for r in requirements]]
        req_table = Table(req_rows, hAlign="LEFT", colWidths=[6.0 * inch])
        req_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(Spacer(1, 0.08 * inch))
        story.append(Paragraph("Dependencies listed in requirements.txt:", styles["Body"]))
        story.append(req_table)

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("9. AI Assistance Disclosure", styles["H2"]))
    story.append(
        Paragraph(
            "I used GitHub Copilot powered by <b>GPT-5.2</b> to help debug Python scripts, improve project structure, and edit the written report for clarity and grammar. "
            "The tool was also used to help check Milestone 2 requirements and organize the submission contents (including verifying that the included PDFs and artifacts fit the milestone). "
            "All final design choices, data selection, and results interpretation are my own.",
            styles["Body"],
        )
    )

    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    doc.build(story)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submission_root",
        default=os.path.dirname(os.path.dirname(__file__)),
        help="Path to submission_m2 directory (default: parent of training_code)",
    )
    parser.add_argument(
        "--out_pdf",
        default=None,
        help="Output PDF path (default: <submission_root>/report.pdf)",
    )
    args = parser.parse_args()

    submission_root = os.path.abspath(args.submission_root)
    out_pdf = args.out_pdf or os.path.join(submission_root, "report.pdf")

    build_pdf(out_pdf=out_pdf, submission_root=submission_root)
    print("Wrote:", out_pdf)


if __name__ == "__main__":
    main()
