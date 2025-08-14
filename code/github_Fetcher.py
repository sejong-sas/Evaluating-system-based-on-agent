from typing import Dict, Any
import requests
import json
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


def _save_reports_for_repo(repo_full_name: str, output_dir: str | Path, items: list[dict]) -> Path | None:
    """
    Save reports as reports_fulltext_github_{owner_repo}.json
    Schema:
      { "repo": str, "full_texts": [ { "arxiv_id": str, "full_text": str }, ... ] }
    Note: 'arxiv_id' holds the source URL even if it's not from arXiv.
    """
    if not items:
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = repo_full_name.replace("/", "_").lower()
    out = output_dir / f"reports_fulltext_github_{base}.json"

    existing = []
    if out.exists():
        try:
            existing = (json.load(open(out, encoding="utf-8")).get("full_texts") or [])
        except Exception:
            existing = []

    payload = {"repo": repo_full_name, "full_texts": existing + items}
    json.dump(payload, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"📄 Reports saved/merged (GH): {out}")
    return out


def _merge_reports_to_hf_if_possible(readme_text: str, items: list[dict], output_dir: str | Path = ".") -> Path | None:
    """
    Try to detect a Hugging Face model ID from README and merge the same items
    into reports_fulltext_{hf_org_model}.json so downstream dispatchers can use it directly.
    """
    if not (readme_text and items):
        return None

    # Find the first HF model link like huggingface.co/org/model
    m = re.search(r"https?://huggingface\.co/([\w\-\._]+/[\w\-\._]+)", readme_text, re.IGNORECASE)
    if not m:
        return None

    hf_id = m.group(1).lower()
    if hf_id.startswith("collections/"):
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = hf_id.replace("/", "_").lower()
    out = output_dir / f"reports_fulltext_{base}.json"

    existing = []
    if out.exists():
        try:
            existing = (json.load(open(out, encoding="utf-8")).get("full_texts") or [])
        except Exception:
            existing = []

    payload = {"model_id": hf_id, "full_texts": existing + items}
    json.dump(payload, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"📄 Reports merged to HF file: {out}")
    return out


def github_fetcher(repo_full_name: str,
                   branch: str = "main",
                   token: str = None,
                   save_to_file: bool = True,
                   output_dir: str | Path = ".") -> Dict[str, Any]:
    """
    Fetches from a GitHub repo:
      - files: list of all file paths
      - license_files: all files starting with 'LICENSE' and their contents
      - readme: contents of README.md (empty string if missing)
      - py_files: all .py files and their contents
      Includes save option (when save_to_file=True, saves as a JSON file)

    NEW:
      - Also scans README links and repo PDFs for *technical reports* (PDF/HTML),
        saves them to 'reports_fulltext_github_{owner_repo}.json'.
      - If a Hugging Face model link is detected in README, the same reports are
        also merged into 'reports_fulltext_{hf_org_model}.json'.
    """
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    # 1) Fetch repository tree
    tree_url = f"https://api.github.com/repos/{repo_full_name}/git/trees/{branch}?recursive=1"
    resp = requests.get(tree_url, headers=headers)
    resp.raise_for_status()
    tree = resp.json().get("tree", [])

    # 2) Extract all file paths
    paths = [item["path"] for item in tree if item["type"] == "blob"]

    # Content fetch helper
    def fetch_raw(path: str) -> str:
        raw_url = f"https://raw.githubusercontent.com/{repo_full_name}/{branch}/{path}"
        r = requests.get(raw_url, headers=headers)
        return r.text if r.status_code == 200 else ""

    # 3) LICENSE* files
    license_paths = [p for p in paths if p.upper().startswith("LICENSE")]
    license_files = {p: fetch_raw(p) for p in license_paths}

    # 4) README.md
    readme = fetch_raw("README.md") if "README.md" in paths else ""

    # 5) .py files
    py_files = {p: fetch_raw(p) for p in paths if p.endswith(".py")}

    result = {
        "repo": repo_full_name,
        "branch": branch,
        "files": paths,
        "license_files": license_files,
        "readme": readme,
        "py_files": py_files
    }

    # 5.5) (NEW) Harvest technical reports from README links + repo PDFs
    try:
        report_texts: list[dict] = []
        seen: set[str] = set()

        # (A) README links
        for u in _extract_urls(readme):
            if not _is_probable_report_url(u):
                continue
            if u in seen:
                continue
            try:
                txt = _fetch_pdf_text(u) if u.lower().endswith(".pdf") else _fetch_html_text(u)
                if txt.strip():
                    report_texts.append({"arxiv_id": u, "full_text": txt})
                    seen.add(u)
            except Exception as e:
                print("⚠️ report fetch failed:", u, e)

        # (B) Repo PDFs via raw.githubusercontent.com
        for p in paths:
            if p.lower().endswith(".pdf"):
                raw_pdf = f"https://raw.githubusercontent.com/{repo_full_name}/{branch}/{p}"
                if raw_pdf in seen:
                    continue
                try:
                    txt = _fetch_pdf_text(raw_pdf)
                    if txt.strip():
                        report_texts.append({"arxiv_id": raw_pdf, "full_text": txt})
                        seen.add(raw_pdf)
                except Exception as e:
                    print("⚠️ report fetch failed:", raw_pdf, e)

        # Save GH-based report file, and HF-based one if possible
        if report_texts and save_to_file:
            _save_reports_for_repo(repo_full_name, output_dir, report_texts)
            _merge_reports_to_hf_if_possible(readme, report_texts, output_dir)
        elif report_texts:
            print("ℹ️ Technical reports were found but save_to_file=False, so they were not written to disk.")

    except Exception as e:
        print("⚠️ report extraction (GitHub) failed:", e)

    # 6) Save file
    if save_to_file:
        filename_safe = repo_full_name.replace("/", "_")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"github_{filename_safe}.json"   # ★
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"✅ GitHub JSON file saved: {output_path}")
    return result


# Test run
if __name__ == "__main__":
    info = github_fetcher("google-gemini/gemma-cookbook", branch="main")


# Usage example
if __name__ == "__main__":
    info = github_fetcher("google-gemini/gemma-cookbook", branch="main")
    import json
    # print(json.dumps(info, indent=2, ensure_ascii=False))

    # print(json.dumps(result, indent=2, ensure_ascii=False))
    for key, values in info.items():
        print("*"*30)
        print(key)
        print(values)

    """
    Fetches from a GitHub repo:
      - files: list of all file paths
      - license_files: all files starting with 'LICENSE' and their contents
      - readme: contents of README.md (empty string if missing)
      - py_files: all .py files and their contents

    NEW:
      - Scans README links + repo PDFs for technical reports (PDF/HTML).
      - Saves as 'reports_fulltext_github_{owner_repo}.json'.
      - If a HF model link is detected in README, also merges into
        'reports_fulltext_{hf_org_model}.json' so downstream dispatchers can use it directly.
    """
