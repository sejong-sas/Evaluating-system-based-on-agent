# huggingface_Fetcher.py
from typing import Dict, Any, List, Tuple
import requests
import json
import os
from pathlib import Path
import re
import fitz  # PyMuPDF
from urllib.parse import urlparse, urljoin

# =========================
# Auth / HTTP
# =========================
HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip()
_UA = {"User-Agent": "Mozilla/5.0"}
_HF_HEADERS = dict(_UA)
if HF_TOKEN:
    _HF_HEADERS["Authorization"] = f"Bearer {HF_TOKEN}"

def _get_hf(url: str, timeout: int = 20, allow_redirects: bool = True) -> requests.Response:
    """
    GET for Hugging Face endpoints:
    - Try with token first (if present).
    - If 401/403, retry anonymously so public models still work.
    Returns the Response without raising for non-200 (so caller can check status_code).
    """
    if "Authorization" in _HF_HEADERS:
        r = requests.get(url, headers=_HF_HEADERS, timeout=timeout, allow_redirects=allow_redirects)
        if r.status_code in (401, 403):
            r = requests.get(url, headers=_UA, timeout=timeout, allow_redirects=allow_redirects)
    else:
        r = requests.get(url, headers=_UA, timeout=timeout, allow_redirects=allow_redirects)
    return r

# =========================
# Report-ish link heuristics (model-agnostic)
# =========================
REPORTISH_STATIC: Tuple[str, ...] = (
    "technical-report", "tech-report", "techreport",
    "whitepaper", "white-paper", "white_paper",
    "paper", "/docs", "docs.", "/blog", "blog.",
    "/research", "research."
)

_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")  # [text](url)
_URL_RE = re.compile(r"https?://[^\s)>\]}]+")

def _extract_md_links(md: str) -> List[Tuple[str, str]]:
    """
    Extract (url, anchor_text) from Markdown.
    Also includes plain URLs found in the text with empty anchor.
    """
    if not md:
        return []
    links: List[Tuple[str, str]] = []
    for m in _MD_LINK_RE.finditer(md):
        text, href = m.group(1).strip(), m.group(2).strip()
        links.append((href, text))
    # plus plain URLs without markdown
    for u in _URL_RE.findall(md):
        links.append((u, ""))  # no anchor text
    # preserve order but de-dup
    seen = set()
    uniq = []
    for href, text in links:
        key = (href, text)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((href, text))
    return uniq

def _is_reportish_url_or_anchor(url: str, anchor_text: str = "") -> bool:
    """
    Model-agnostic check:
    - Accept if 'report' OR 'technical' appears anywhere in URL or anchor text (substring match),
      so things like 'gemma3report' are caught WITHOUT hardcoding model names.
    - OR if any static generic keyword appears (docs/blog/research/paper/whitepaper/...).
    """
    if not url:
        return False
    s = f"{url} {anchor_text}".lower()
    if ("report" in s) or ("technical" in s):
        return True
    return any(k in s for k in REPORTISH_STATIC)

def _norm_url_key(u: str) -> str:
    """Normalize URL to a scheme-agnostic key for de-dup (host+path, lowercase, no query/fragment)."""
    try:
        pr = urlparse(u)
        return (pr.netloc.lower().strip("/") + pr.path.lower().rstrip("/")) or u
    except Exception:
        return u

# =========================
# Fetchers
# =========================
def _fetch_pdf_text(url: str) -> str:
    """Download PDF and return plain text using PyMuPDF."""
    r = requests.get(url, timeout=20, headers=_UA, allow_redirects=True)
    r.raise_for_status()
    with fitz.open(stream=r.content, filetype="pdf") as doc:
        return "\n".join(p.get_text() for p in doc)

def _fetch_html_text(url: str) -> str:
    """
    Download HTML and strip tags/scripts/styles to plain text.
    If the URL (e.g., a short link) actually returns a PDF, detect it by
    Content-Type or final URL and extract text as PDF instead.
    Also tries to follow a direct PDF link inside the HTML (e.g., 'Download PDF').
    """
    r = requests.get(url, timeout=15, headers=_UA, allow_redirects=True)
    r.raise_for_status()

    # Detect PDF even if the URL doesn't end with .pdf (e.g., short link → GCS PDF)
    ct = (r.headers.get("Content-Type") or "").lower()
    final_url = (r.url or "").lower()
    if ("pdf" in ct) or final_url.endswith(".pdf"):
        with fitz.open(stream=r.content, filetype="pdf") as doc:
            return "\n".join(p.get_text() for p in doc)

    html = r.text

    # Try to locate embedded PDF links and fetch one
    pdf_links = re.findall(r'href=["\']([^"\']+\.pdf)["\']', html, flags=re.I)
    if pdf_links:
        first = pdf_links[0]
        try:
            abs_pdf = urljoin(final_url, first)
            rr = requests.get(abs_pdf, timeout=20, headers=_UA, allow_redirects=True)
            rr.raise_for_status()
            if ("pdf" in (rr.headers.get("Content-Type") or "").lower()) or abs_pdf.lower().endswith(".pdf"):
                with fitz.open(stream=rr.content, filetype="pdf") as doc:
                    return "\n".join(p.get_text() for p in doc)
        except Exception:
            pass

    # Otherwise treat as HTML
    html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?is)<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text[:800_000]

# =========================
# Save reports_fulltext
# =========================
def _save_reports_for_model(model_id: str, output_dir: str | Path, items: List[Dict[str, str]]) -> None:
    """
    Append/merge into reports_fulltext_{model}.json.
    Schema compatible with arxiv dispatcher expectations:
      { "model_id": str, "full_texts": [ { "arxiv_id": str, "full_text": str }, ... ] }
    Note: 'arxiv_id' holds the source URL even if it's not from ArXiv.
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

# =========================
# Main fetcher
# =========================
def huggingface_fetcher(model_id: str, save_to_file: bool = True, output_dir: str | Path = ".") -> Dict[str, Any]:
    """
    Fetches metadata and key files for a Hugging Face model repository,
    and additionally harvests 'technical/report-ish' documents from README links
    and repository-hosted PDFs, using ONLY model-agnostic heuristics.
    """
    base_api = f"https://huggingface.co/api/models/{model_id}?full=true"
    resp = _get_hf(base_api)
    resp.raise_for_status()
    data = resp.json()

    # 1) list siblings
    siblings: List[str] = [f.get("rfilename", "") for f in data.get("siblings", []) if isinstance(f, dict)]

    # Helper: fetch raw file text from a few branches
    def fetch_raw(filename: str) -> str:
        branches = ["main", "refs/convert", "refs/pr/1"]
        for branch in branches:
            url = f"https://huggingface.co/{model_id}/resolve/{branch}/{filename}"
            r = _get_hf(url)
            if r.status_code == 200:
                return r.text
        return ""

    # 2) fetch contents of key files
    readme = fetch_raw("README.md") if "README.md" in siblings else ""
    config = fetch_raw("config.json") if "config.json" in siblings else ""
    generation_config = fetch_raw("generation_config.json") if "generation_config.json" in siblings else ""

    # LICENSE file(s)
    license_candidates = [fn for fn in siblings if isinstance(fn, str) and fn.upper().startswith("LICENSE")]
    license_file = fetch_raw(license_candidates[0]) if license_candidates else ""

    # 3) fetch all .py files (not strictly needed for reports but useful downstream)
    py_files: Dict[str, str] = {}
    for fn in siblings:
        if isinstance(fn, str) and fn.endswith(".py"):
            py_files[fn] = fetch_raw(fn)

    # 4) assemble main result
    result: Dict[str, Any] = {
        "model_id": model_id,
        "files": siblings,
        "readme": readme,
        "config": config,
        "generation_config": generation_config,
        "license_file": license_file,
        "py_files": py_files
    }

    # 4.5) Harvest technical/report-ish docs from README links + repo PDFs
    try:
        report_texts: List[Dict[str, str]] = []
        seen_keys: set = set()   # normalized keys for de-dup
        seen_urls: set = set()   # original URLs for sanity

        # (A) README links (markdown links + plain URLs)
        repo_home = f"https://huggingface.co/{model_id}"
        for href, text in _extract_md_links(readme):
            if not href:
                continue
            # Make absolute if relative
            try:
                if href.startswith("/"):
                    href = urljoin(repo_home + "/", href.lstrip("/"))
            except Exception:
                pass

            if not _is_reportish_url_or_anchor(href, text):
                continue

            key = _norm_url_key(href)
            if key in seen_keys:
                continue

            try:
                # If endswith .pdf use PDF fetcher; else try HTML (with embedded-PDF sniffing)
                txt = _fetch_pdf_text(href) if href.lower().endswith(".pdf") else _fetch_html_text(href)
                if txt and txt.strip():
                    report_texts.append({"arxiv_id": href, "full_text": txt})
                    seen_keys.add(key)
                    seen_urls.add(href)
            except Exception as e:
                print("⚠️ report fetch failed:", href, e)

        # (B) PDFs hosted directly in the repo
        for fn in siblings:
            if not isinstance(fn, str) or not fn.lower().endswith(".pdf"):
                continue
            for br in ["main", "refs/convert", "refs/pr/1"]:
                url = f"https://huggingface.co/{model_id}/resolve/{br}/{fn}"
                key = _norm_url_key(url)
                if key in seen_keys:
                    break
                try:
                    txt = _fetch_pdf_text(url)
                    if txt and txt.strip():
                        report_texts.append({"arxiv_id": url, "full_text": txt})
                        seen_keys.add(key)
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

    # 5) save main JSON
    if save_to_file:
        filename_safe = model_id.replace("/", "_")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"huggingface_{filename_safe}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"✅ JSON file saved: {output_path}")

    return result

# =========================
# Standalone test
# =========================
if __name__ == "__main__":
    # Example run
    test_model_id = "google/gemma-3-4b-it"  # change to any repo; logic is model-agnostic
    result = huggingface_fetcher(test_model_id)

    # Quick print
    for key, values in result.items():
        print("*" * 30)
        print(key)
        if isinstance(values, dict):
            print(list(values.keys()))
        else:
            print(values if isinstance(values, (str, list)) else type(values))
