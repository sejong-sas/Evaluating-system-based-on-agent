from typing import Dict, Any
import requests
import json
import os
from pathlib import Path
import re
import fitz  # PyMuPDF


# -----------------------------
# Helpers for technical report harvesting
# -----------------------------
def _extract_urls(text: str) -> list[str]:
    """Return unique http(s) URLs from text (order-preserving)."""
    if not text:
        return []
    urls = re.findall(r'https?://[^\s)>\"]+', text)
    return list(dict.fromkeys(urls))


def _is_probable_report_url(u: str) -> bool:
    """
    Heuristics for deciding if a URL likely points to a 'technical report' page.
    - Accept direct PDFs
    - Accept docs/blog/research/whitepaper-like pages
    """
    ul = u.lower()
    if ul.endswith(".pdf"):
        return True
    return any(k in ul for k in [
        "technical-report", "tech-report", "whitepaper", "white-paper", "paper",
        "/docs", "docs.", "/blog", "blog.", "/research", "research."
    ])


def _fetch_pdf_text(url: str) -> str:
    """Download PDF and return plain text using PyMuPDF."""
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    with fitz.open(stream=r.content, filetype="pdf") as doc:
        return "\n".join(p.get_text() for p in doc)


def _fetch_html_text(url: str) -> str:
    """Download HTML and strip tags/scripts/styles to plain text."""
    r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    html = r.text
    # strip scripts/styles first
    html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    # strip all tags
    text = re.sub(r"(?is)<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text[:800_000]


def _save_reports_for_model(model_id: str, output_dir: str | Path, items: list[dict]) -> None:
    """
    Append/merge into reports_fulltext_{model}.json.
    Schema compatible with arxiv dispatcher expectations:
      { "model_id": str, "full_texts": [ { "arxiv_id": str, "full_text": str }, ... ] }
    Note: 'arxiv_id' holds the source URL even if it's not from arXiv.
    """
    if not items:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = model_id.replace("/", "_").lower()
    out = output_dir / f"reports_fulltext_{base}.json"

    existing = []
    if out.exists():
        try:
            existing = (json.load(open(out, encoding="utf-8")).get("full_texts") or [])
        except Exception:
            existing = []

    payload = {"model_id": model_id, "full_texts": existing + items}
    json.dump(payload, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"📄 Reports saved/merged: {out}")


def huggingface_fetcher(model_id: str, save_to_file: bool = True, output_dir: str | Path = ".") -> Dict[str, Any]:
    base_api = f"https://huggingface.co/api/models/{model_id}?full=true"
    resp = requests.get(base_api)
    resp.raise_for_status()
    data = resp.json()

    # 1) collect siblings list
    siblings = [f.get("rfilename", "") for f in data.get("siblings", [])]

    # helper: build raw URL and fetch text
    def fetch_raw(filename: str) -> str:
        branches = ["main", "refs/convert", "refs/pr/1"]
        for branch in branches:
            url = f"https://huggingface.co/{model_id}/resolve/{branch}/{filename}"
            r = requests.get(url)
            if r.status_code == 200:
                return r.text
        return ""

    # 2) fetch contents of key files
    readme = fetch_raw("README.md") if "README.md" in siblings else ""
    config = fetch_raw("config.json") if "config.json" in siblings else ""
    generation_config = fetch_raw("generation_config.json") if "generation_config.json" in siblings else ""

    # search LICENSE file
    license_candidates = [fn for fn in siblings if fn.upper().startswith("LICENSE")]
    license_file = ""
    if license_candidates:
        license_file = fetch_raw(license_candidates[0])

    # 3) fetch all .py files
    py_files = {}
    for fn in siblings:
        if fn.endswith(".py"):
            py_files[fn] = fetch_raw(fn)

    # 4) assemble result
    result = {
        "model_id": model_id,
        "files": siblings,
        "readme": readme,
        "config": config,
        "generation_config": generation_config,
        "license_file": license_file,
        "py_files": py_files
    }

    # 4.5) (NEW) Harvest technical reports from README links + repo PDFs (save as reports_fulltext_{model}.json)
    #      We keep this independent from the main JSON to avoid changing downstream readers.
    try:
        report_texts: list[dict] = []
        seen_urls: set[str] = set()

        # (A) README links → probable report pages (PDF or HTML)
        for u in _extract_urls(readme):
            if not _is_probable_report_url(u):
                continue
            if u in seen_urls:
                continue
            try:
                txt = _fetch_pdf_text(u) if u.lower().endswith(".pdf") else _fetch_html_text(u)
                if txt.strip():
                    report_texts.append({"arxiv_id": u, "full_text": txt})
                    seen_urls.add(u)
            except Exception as e:
                print("⚠️ report fetch failed:", u, e)

        # (B) Repository PDFs hosted on the HF model repo itself
        #     Try a few resolve branches to be resilient to LFS / converted refs
        for fn in siblings:
            if fn.lower().endswith(".pdf"):
                for br in ["main", "refs/convert", "refs/pr/1"]:
                    url = f"https://huggingface.co/{model_id}/resolve/{br}/{fn}"
                    if url in seen_urls:
                        break
                    try:
                        txt = _fetch_pdf_text(url)
                        if txt.strip():
                            report_texts.append({"arxiv_id": url, "full_text": txt})
                            seen_urls.add(url)
                        break  # success on this branch
                    except Exception:
                        continue

        if save_to_file and report_texts:
            _save_reports_for_model(model_id, output_dir, report_texts)
        elif report_texts:
            print("ℹ️ Technical reports were found but save_to_file=False, so they were not written to disk.")

    except Exception as e:
        print("⚠️ report extraction (HF) failed:", e)

    # 5) save JSON (main HF metadata/output)
    if save_to_file:
        filename_safe = model_id.replace("/", "_")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)         # ★ create folder
        output_path = output_dir / f"huggingface_{filename_safe}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"✅ JSON file saved: {output_path}")

    return result


# test code when run standalone
if __name__ == "__main__":
    test_model_id = "skt/A.X-4.0"
    huggingface_fetcher(test_model_id)


# usage
if __name__ == "__main__":
    result = huggingface_fetcher("deepseek-ai/DeepSeek-R1")
    # import json
    # print(json.dumps(result, indent=2, ensure_ascii=False))

    for key, values in result.items():
        print("*"*30)
        print(key)
        print(values)

    """
    Fetches the file list and the contents of key files for a given Hugging Face model.

    Returns a dict with:
    - model_id
    - files: list of sibling filenames
    - readme: content of README.md (or empty if none)
    - config: content of config.json (or empty if none)
    - generation_config: content of generation_config.json (or empty if none)
    - license_file: content of LICENSE* files (or empty if none)
    - py_files: dict mapping each *.py filename to its content

    NEW:
    - Additionally scans README links and repo PDFs for technical reports and, if found,
      saves them as 'reports_fulltext_{model}.json' alongside the main JSON.
    """
