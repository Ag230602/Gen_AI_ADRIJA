"""Milestone 3 dashboard for end-to-end orchestration and checks.

Run:
  .venv/bin/streamlit run code/training_code/m3_dashboard.py

This dashboard can:
- run each pipeline step individually
- run the full end-to-end pipeline
- check required artifacts
- preview key evaluation metrics
"""

from __future__ import annotations

import json
import io
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


st: Any = None


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "code" / "results"
DATA_DIR = REPO_ROOT / "code" / "data"


@dataclass
class Step:
    key: str
    label: str
    script_rel: str
    args: List[str]


STEPS: List[Step] = [
    Step("pdf", "1) Build corpus from PDFs", "code/training_code/pdf_to_jsonl.py", ["--mode", "paragraph", "--min_chars", "200"]),
    Step("tok", "2) Train tokenizer", "code/training_code/train_tokenizer.py", []),
    Step("diag", "3) Tokenizer diagnostics", "code/training_code/tokenizer_diagnostics.py", []),
    Step("ret", "4) Retrieval demo", "code/training_code/retrieval/run_retrieval_demo.py", []),
    Step("peft", "5) Base vs Adapted (LoRA)", "code/training_code/base_vs_peft.py", []),
]

REQUIRED_ARTIFACTS = [
    "code/results/tokenizer_diagnostics.json",
    "code/results/retrieval_demo.json",
    "code/results/base_vs_adapted.json",
    "code/results/base_vs_adapted.md",
    "code/results/loss_log.txt",
    "code/results/loss_plot.png",
]


def run_python_script(script_rel: str, args: List[str]) -> Dict[str, str | int]:
    script_path = REPO_ROOT / script_rel
    cmd = [sys.executable, str(script_path), *args]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "command": " ".join(cmd),
    }


def artifact_exists(rel_path: str) -> bool:
    return (REPO_ROOT / rel_path).exists()


def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_last_loss(path: Path) -> Optional[dict]:
    if not path.exists():
        return None

    rows: List[str] = [r.strip() for r in path.read_text(encoding="utf-8").splitlines() if r.strip()]
    if len(rows) < 2:
        return None

    header = rows[0].split("\t")
    values = rows[-1].split("\t")
    if len(values) != len(header):
        return None
    return dict(zip(header, values))


def collect_submission_sources() -> Tuple[Dict[str, Path], List[str]]:
    demo_video = REPO_ROOT / "demo_video.mp4"
    demo_link = REPO_ROOT / "demo_link.txt"

    sources: Dict[str, Path] = {
        "README.md": REPO_ROOT / "README.md",
        "requirements.txt": REPO_ROOT / "requirements.txt",
        "paper.pdf": REPO_ROOT / "paper.pdf",
        "code": REPO_ROOT / "code",
        "results": REPO_ROOT / "results",
    }

    if demo_video.exists():
        sources["demo_video.mp4"] = demo_video
    elif demo_link.exists():
        sources["demo_link.txt"] = demo_link

    missing = [arc for arc, src in sources.items() if not src.exists()]
    if not demo_video.exists() and not demo_link.exists():
        missing.append("demo_video.mp4 or demo_link.txt")

    return sources, missing


def build_submission_zip(last_name: str, first_name: str) -> Tuple[str, Optional[bytes], List[str]]:
    zip_name = f"CS5590_Grad_M3_{last_name}_{first_name}.zip"
    sources, missing = collect_submission_sources()
    if missing:
        return zip_name, None, missing

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arc, src in sources.items():
            if src.is_file():
                zf.write(src, arc)
                continue
            for p in src.rglob("*"):
                if p.is_file():
                    rel = p.relative_to(src)
                    zf.write(p, f"{arc}/{rel.as_posix()}")

    return zip_name, buf.getvalue(), []


def render_metrics() -> None:
    st.subheader("Key Results")

    base_vs = read_json(RESULTS_DIR / "base_vs_adapted.json")
    if base_vs:
        base = base_vs.get("base_model", {}).get("results", {})
        adapted = base_vs.get("adapted_model", {}).get("results", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Base test loss", f"{base.get('test_loss', 'NA'):.4f}" if isinstance(base.get("test_loss"), (int, float)) else "NA")
        c2.metric("Adapted test loss", f"{adapted.get('test_loss', 'NA'):.4f}" if isinstance(adapted.get("test_loss"), (int, float)) else "NA")
        c3.metric("Base perplexity", f"{base.get('test_perplexity', 'NA'):.2f}" if isinstance(base.get("test_perplexity"), (int, float)) else "NA")
        c4.metric("Adapted perplexity", f"{adapted.get('test_perplexity', 'NA'):.2f}" if isinstance(adapted.get("test_perplexity"), (int, float)) else "NA")

        st.markdown("**Trainable parameter counts**")
        st.table(
            [
                {
                    "system": "base",
                    "trainable": base_vs.get("base_model", {}).get("trainable_parameters", "NA"),
                    "total": base_vs.get("base_model", {}).get("total_parameters", "NA"),
                },
                {
                    "system": "adapted",
                    "trainable": base_vs.get("adapted_model", {}).get("trainable_parameters", "NA"),
                    "total": base_vs.get("adapted_model", {}).get("total_parameters", "NA"),
                },
            ]
        )
    else:
        st.info("No base_vs_adapted.json yet. Run step 5.")

    tok = read_json(RESULTS_DIR / "tokenizer_diagnostics.json")
    if tok:
        st.markdown("**Tokenizer diagnostics**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg tokens/doc", f"{tok.get('avg_tokens_per_doc', 'NA')}")
        c2.metric("Tokens/1k chars", f"{tok.get('tokens_per_1k_chars', 'NA')}")
        c3.metric("UNK rate", f"{tok.get('unk_token_rate', 'NA')}")

    loss_last = read_last_loss(RESULTS_DIR / "loss_log.txt")
    if loss_last:
        st.markdown("**Last logged loss row**")
        st.json(loss_last)


def render_retrieval_preview() -> None:
    st.subheader("Retrieval Preview")
    demo = read_json(RESULTS_DIR / "retrieval_demo.json")
    if not demo:
        st.info("No retrieval_demo.json yet. Run step 4.")
        return

    queries = demo.get("queries", [])
    st.write(f"Indexed docs: {demo.get('n_docs', 'NA')}")
    for q in queries[:3]:
        with st.expander(f"Query: {q.get('query', '')}"):
            for hit in q.get("top_k", []):
                st.write(
                    {
                        "doc_id": hit.get("doc_id"),
                        "score": hit.get("score"),
                        "meta": hit.get("meta", {}),
                    }
                )
                st.caption((hit.get("text", "") or "")[:500])


def main() -> None:
    global st
    if st is None:
        import importlib

        try:
            st = importlib.import_module("streamlit")
        except Exception as e:
            raise RuntimeError(
                "Streamlit is required for dashboard mode. Install with: pip install streamlit"
            ) from e

    st.set_page_config(page_title="StormCare M3 Dashboard", layout="wide")
    st.title("StormCare RAG — Milestone 3 Dashboard")
    st.caption("End-to-end runner + artifact checker + metrics view")

    st.sidebar.header("Paths")
    st.sidebar.write(f"Repo root: {REPO_ROOT}")
    st.sidebar.write(f"Data dir: {DATA_DIR}")
    st.sidebar.write(f"Results dir: {RESULTS_DIR}")

    st.subheader("Submission Readiness Check")
    check_rows = []
    for rel in REQUIRED_ARTIFACTS:
        check_rows.append({"artifact": rel, "status": "✅" if artifact_exists(rel) else "❌"})
    st.table(check_rows)

    st.subheader("Run Pipeline Steps")

    if "run_logs" not in st.session_state:
        st.session_state.run_logs = []

    c_all, _ = st.columns([1, 4])
    with c_all:
        if st.button("Run Full End-to-End Pipeline", type="primary"):
            for step in STEPS:
                out = run_python_script(step.script_rel, step.args)
                st.session_state.run_logs.append({"step": step.label, **out})
                if out["returncode"] != 0:
                    st.error(f"Stopped on: {step.label}")
                    break
            st.rerun()

    cols = st.columns(len(STEPS))
    for i, step in enumerate(STEPS):
        with cols[i]:
            if st.button(step.label, key=f"btn_{step.key}"):
                out = run_python_script(step.script_rel, step.args)
                st.session_state.run_logs.append({"step": step.label, **out})
                st.rerun()

    st.subheader("Build Final Submission ZIP")
    c1, c2 = st.columns(2)
    with c1:
        last_name = st.text_input("Last name", value="LastName")
    with c2:
        first_name = st.text_input("First name", value="FirstName")

    if st.button("Prepare ZIP for Download"):
        zip_name, zip_bytes, missing = build_submission_zip(last_name.strip(), first_name.strip())
        if missing:
            st.error("Missing required item(s):")
            for item in missing:
                st.write(f"- {item}")
        elif zip_bytes is not None:
            st.success(f"Ready: {zip_name}")
            st.download_button(
                label="Download Final ZIP",
                data=zip_bytes,
                file_name=zip_name,
                mime="application/zip",
            )

    st.subheader("Execution Logs")
    if not st.session_state.run_logs:
        st.info("No runs yet.")
    else:
        for idx, log in enumerate(reversed(st.session_state.run_logs[-10:])):
            with st.expander(f"{log['step']} (exit={log['returncode']})"):
                st.code(log.get("command", ""), language="bash")
                if log.get("stdout"):
                    st.text_area("stdout", log["stdout"], height=180, key=f"stdout_{idx}")
                if log.get("stderr"):
                    st.text_area("stderr", log["stderr"], height=120, key=f"stderr_{idx}")

    render_metrics()
    render_retrieval_preview()


if __name__ == "__main__":
    main()
