"""Convert a folder of PDFs into a JSONL text corpus.

This supports the Milestone 2 requirement to use your own dataset. If your
corpus is "two PDFs", place them under:
  data/pdfs/

This script extracts text from each PDF (all pages), concatenates it into a
single document string, and writes:
  data/corpus.jsonl

Each line is:
  {"source": "<pdf filename>", "text": "..."}

Run:
  .venv_py39/bin/python training_code/pdf_to_jsonl.py

Notes:
- PDF text extraction quality depends on the PDF (scanned PDFs may need OCR).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import re
from glob import glob
from typing import List

from pypdf import PdfReader

# Make repo root importable if needed.
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def extract_pdf_text(pdf_path: str, max_pages: int | None = None) -> str:
    reader = PdfReader(pdf_path)
    texts: List[str] = []
    for i, page in enumerate(reader.pages):
        if max_pages is not None and i >= max_pages:
            break
        page_text = page.extract_text() or ""
        page_text = page_text.strip()
        if page_text:
            texts.append(page_text)
    return "\n\n".join(texts).strip()


def extract_pdf_pages(pdf_path: str, max_pages: int | None = None) -> List[str]:
    reader = PdfReader(pdf_path)
    pages: List[str] = []
    for i, page in enumerate(reader.pages):
        if max_pages is not None and i >= max_pages:
            break
        page_text = (page.extract_text() or "").strip()
        pages.append(page_text)
    return pages


_PARA_SPLIT_RE = re.compile(r"\n\s*\n+")


def split_paragraphs(text: str, min_chars: int) -> List[str]:
    if not text.strip():
        return []
    parts = [p.strip() for p in _PARA_SPLIT_RE.split(text) if p.strip()]
    # Filter very short paragraphs (headers, page numbers, etc.).
    return [p for p in parts if len(p) >= min_chars]


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf_dir",
        default=os.path.join(repo_root, "data", "pdfs"),
        help="Directory containing PDF files to ingest.",
    )
    parser.add_argument(
        "--output_jsonl",
        default=os.path.join(repo_root, "data", "corpus.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--mode",
        choices=["document", "paragraph"],
        default="document",
        help="document: 1 JSONL row per PDF. paragraph: chunk into paragraphs with page/chunk metadata.",
    )
    parser.add_argument(
        "--min_chars",
        type=int,
        default=200,
        help="Minimum characters for a paragraph chunk (paragraph mode only).",
    )
    parser.add_argument(
        "--max_pages",
        type=int,
        default=None,
        help="Optional cap on pages read per PDF (for quick debugging).",
    )
    args = parser.parse_args()

    pdf_paths = sorted(glob(os.path.join(args.pdf_dir, "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in {args.pdf_dir}. Put your 2 PDFs there (ending with .pdf)."
        )

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)

    n_written = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as out:
        for pdf_path in pdf_paths:
            if args.mode == "document":
                text = extract_pdf_text(pdf_path, max_pages=args.max_pages)
                if not text:
                    print(f"WARNING: no extractable text in {os.path.basename(pdf_path)}")
                    continue
                row = {"source": os.path.basename(pdf_path), "text": text}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1
            else:
                pages = extract_pdf_pages(pdf_path, max_pages=args.max_pages)
                chunk_id = 0
                for page_idx, page_text in enumerate(pages):
                    for para in split_paragraphs(page_text, min_chars=args.min_chars):
                        row = {
                            "source": os.path.basename(pdf_path),
                            "page": page_idx + 1,
                            "chunk_id": chunk_id,
                            "text": para,
                        }
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        n_written += 1
                        chunk_id += 1

    print(f"Wrote {n_written} rows to: {args.output_jsonl} (mode={args.mode})")


if __name__ == "__main__":
    main()
