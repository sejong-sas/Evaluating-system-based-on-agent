# github_Fetcher.py
from typing import Dict, Any, List
import requests
import json
import os
from pathlib import Path
import re
import fitz  # PyMuPDF
from urllib.parse import urlparse

GITHUB_TOKEN = (os.getenv("GITHUB_TOKEN") or "").strip()
_UA = {"User-Agent": "Mozilla/5.0"}
_GH_HEADERS = {"Accept": "application/vnd.github.v3+json", **_UA}
if GITHUB_TOKEN:
    _GH_HEADERS["Authorization"] = f"token {GITHUB_TOKEN}"


# ---------------- URL utils ----------------
def _clean_url(u: str) -> str:
    if not isinstance(u, str):
        return ""
    s = u.strip()
    # keep the last http(s) chunk if the string accidentally includes two URLs mashed together
    if s.count("http://") + s.count("https://") >= 2:
        idx = s.rfind("http")
        s = s[idx:]
    return s.strip().rstrip('.,;:)]}>"\'')


def _extract_urls(text: str) -> List[str]:
    """
    Robust URL extractor for Markdown/HTML text.
    - Avoids trailing punctuation/brackets that cause 404s (e.g., '...12948}' → '...12948')
    - Keeps the last full 'http...' chunk if multiple got concatenated
    - De-duplicates while preserving order
    """
    if not text:
        return []
    # Grab liberal URL spans but stop before obvious terminators
    raw = re.findall(r'https?://[^\s<>"\'\)\]\}]+', text)
    out: List[str] = []
    seen: set[str] = set()
    for u in raw:
        cu = _clean_url(u)
        if cu and cu not in seen:
            seen.add(cu)
            out.append(cu)
    return out


def _is_probable_report_url(u: str) -> bool:
    """
    Generic detector for 'report-like' or 'paper-like' links — model/vendor agnostic.
    Prioritizes scholarly & technical content without hardcoding org-specific strings.
    """
    if not isinstance(u, str) or not u:
        return False
    ul = u.lower()

    # 1) Obvious: direct PDFs
    if ul.endswith(".pdf"):
        return True

    # 2) Parse host/path
    try:
        parsed = urlparse(u)
        host = (parsed.netloc or "").lower()
        path = (parsed.path or "").lower()
    except Exception:
        return False

    # Shorteners (let them pass so we can expand and parse the target)
    shorteners = {
        "goo.gle", "g.co", "bit.ly", "t.co", "tinyurl.com",
        "ow.ly", "lnkd.in", "rb.gy", "rebrand.ly"
    }
    if host in shorteners:
        return True

    # 3) Scholarly hosts (generic, not vendor-specific)
    scholarly_hosts = (
        "arxiv.org",
        "openreview.net",
        "aclanthology.org",
        "ieeexplore.ieee.org",
        "dl.acm.org",
        "papers.nips.cc",
        "proceedings.mlr.press",
        "hal.science",
        "biorxiv.org",
        "medrxiv.org",
        "arxiv-vanity.com",
    )
    if any(h in host for h in scholarly_hosts):
        return True

    # 4) Generic path tokens that typically indicate technical documents
    path_tokens = (
        "/paper", "/papers",
        "/publication", "/publications",
        "whitepaper", "white-paper",
        "technical-report", "techreport", "tech-report",
        "/docs", "/documentation",
        "/research",
        "/resources/",
        "/post/", "/posts/",
        "/blog/",  # blogs often announce and link to reports
        "/news/",  # news/announcements pages that typically link to reports
        "announce", "announc", "release",
        "report",
    )
    if any(tok in path for tok in path_tokens):
        return True

    # 5) End-of-path patterns for doc-like slugs
    if re.search(r"/(paper|report|whitepaper|technical[-_]report)s?/?$", path):
        return True

    return False


def _fetch_pdf_text(url: str) -> str:
    r = requests.get(url, headers=_UA, timeout=25)
    r.raise_for_status()
    with fitz.open(stream=r.content, filetype="pdf") as doc:
        return "\n".join(p.get_text("text") for p in doc)


def _fetch_html_text(url: str) -> str:
    r = requests.get(url, headers=_UA, timeout=20, allow_redirects=True)
    r.raise_for_status()
    ct = (r.headers.get("Content-Type") or "").lower()
    final_url = (r.url or "").lower()
    if ("pdf" in ct) or final_url.endswith(".pdf"):
        with fitz.open(stream=r.content, filetype="pdf") as doc:
            return "\n".join(p.get_text("text") for p in doc)
    html = r.text
    html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?is)<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text[:800_000]


def _save_reports_for_repo(repo_full_name: str, output_dir: str | Path, items: List[dict]) -> Path | None:
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


def _merge_reports_to_hf_if_possible(readme_text: str, items: List[dict], output_dir: str | Path = ".") -> Path | None:
    """
    Only merge to HF file if README mentions exactly ONE HF model id.
    This avoids accidental merges (e.g., 8x7B run merging into 8x22B file).
    """
    if not (readme_text and items):
        return None
    ids = re.findall(r"https?://huggingface\.co/([\w\-\._]+/[\w\-\._]+)", readme_text, re.IGNORECASE)
    ids = [i.lower() for i in ids if not i.lower().startswith("collections/")]
    ids = list(dict.fromkeys(ids))
    if len(ids) != 1:
        return None  # ambiguous → skip; higher-level pipeline will do anchored merge

    hf_id = ids[0]
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
                   token: str | None = None,
                   save_to_file: bool = True,
                   output_dir: str | Path = ".") -> Dict[str, Any]:
    headers = dict(_GH_HEADERS)
    if token:
        headers["Authorization"] = f"token {token}"

    # 1) repo tree
    tree_url = f"https://api.github.com/repos/{repo_full_name}/git/trees/{branch}?recursive=1"
    resp = requests.get(tree_url, headers=headers, timeout=25)
    resp.raise_for_status()
    tree = resp.json().get("tree", []) or []

    # 2) file paths
    paths = [item["path"] for item in tree if item.get("type") == "blob"]

    def fetch_raw(path: str) -> str:
        raw_url = f"https://raw.githubusercontent.com/{repo_full_name}/{branch}/{path}"
        r = requests.get(raw_url, headers=_UA, timeout=20)
        return r.text if r.status_code == 200 else ""

    # 3) LICENSE*
    license_paths = [p for p in paths if p.upper().startswith("LICENSE")]
    license_files = {p: fetch_raw(p) for p in license_paths}

    # 4) README (allow README variants)
    readme = ""
    for cand in ["README.md", "README.MD", "README", "Readme.md", "readme.md"]:
        if cand in paths:
            readme = fetch_raw(cand)
            break

    # 5) .py files (limit optional if needed)
    py_files = {p: fetch_raw(p) for p in paths if p.endswith(".py")}

    result = {
        "repo": repo_full_name,
        "branch": branch,
        "files": paths,
        "license_files": license_files,
        "readme": readme,
        "py_files": py_files
    }

    # 5.5) Technical reports (README links + repo PDFs)
    try:
        report_texts: List[dict] = []
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

        # (B) Repo PDFs
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

        if report_texts and save_to_file:
            _save_reports_for_repo(repo_full_name, output_dir, report_texts)
            _merge_reports_to_hf_if_possible(readme, report_texts, output_dir)
        elif report_texts:
            print("ℹ️ Technical reports were found but save_to_file=False, so they were not written to disk.")

    except Exception as e:
        print("⚠️ report extraction (GitHub) failed:", e)

    # 6) Save main JSON
    if save_to_file:
        filename_safe = repo_full_name.replace("/", "_")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"github_{filename_safe}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"✅ GitHub JSON file saved: {output_path}")
    return result


if __name__ == "__main__":
    info = github_fetcher("google-gemini/gemma-cookbook", branch="main")
    for k, v in info.items():
        print("*" * 30, k)
        print(v)
