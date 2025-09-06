# huggingface_Fetcher.py
from typing import Dict, Any, List
import requests
import json
import os
from pathlib import Path
import re
import fitz  # PyMuPDF
from urllib.parse import urlparse

# ==== HF auth + safe GET (token → fallback) ====
HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip()
_UA = {"User-Agent": "Mozilla/5.0"}
_HF_HEADERS = dict(_UA)
if HF_TOKEN:
    _HF_HEADERS["Authorization"] = f"Bearer {HF_TOKEN}"


def _get_hf(url: str, timeout: int = 20, allow_redirects: bool = True) -> requests.Response:
    if "Authorization" in _HF_HEADERS:
        r = requests.get(url, headers=_HF_HEADERS, timeout=timeout, allow_redirects=allow_redirects)
        if r.status_code in (401, 403):
            r = requests.get(url, headers=_UA, timeout=timeout, allow_redirects=allow_redirects)
    else:
        r = requests.get(url, headers=_UA, timeout=timeout, allow_redirects=allow_redirects)
    return r


# ----------------------------- URL helpers -----------------------------
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


def _save_reports_for_model(model_id: str, output_dir: str | Path, items: List[dict]) -> None:
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


# ----------------------------- Main HF fetcher -----------------------------
def huggingface_fetcher(model_id: str, save_to_file: bool = True, output_dir: str | Path = ".") -> Dict[str, Any]:
    base_api = f"https://huggingface.co/api/models/{model_id}?full=true"
    resp = _get_hf(base_api)
    resp.raise_for_status()
    data = resp.json()

    siblings = [f.get("rfilename", "") for f in data.get("siblings", [])]

    # Resolve helper: try multiple branches in order
    def fetch_raw(filename: str) -> str:
        for br in ["main", "refs/convert", "refs/pr/1"]:
            url = f"https://huggingface.co/{model_id}/resolve/{br}/{filename}"
            r = _get_hf(url)
            if r.status_code == 200:
                return r.text
        return ""

    # README variants
    readme = ""
    for cand in ["README.md", "README.MD", "README", "Readme.md", "readme.md"]:
        if cand in siblings:
            readme = fetch_raw(cand)
            break

    # Key JSONs
    config = fetch_raw("config.json") if "config.json" in siblings else ""
    generation_config = fetch_raw("generation_config.json") if "generation_config.json" in siblings else ""

    # LICENSE*
    license_file = ""
    lic_cands = [fn for fn in siblings if fn.upper().startsWith("LICENSE")] if False else [fn for fn in siblings if fn.upper().startswith("LICENSE")]
    if lic_cands:
        license_file = fetch_raw(lic_cands[0])

    # .py files (raw)
    py_files = {}
    for fn in siblings:
        if fn.endswith(".py"):
            py_files[fn] = fetch_raw(fn)

    result = {
        "model_id": model_id,
        "files": siblings,
        "readme": readme,
        "config": config,
        "generation_config": generation_config,
        "license_file": license_file,
        "py_files": py_files
    }

    # Technical reports (README links + PDFs in HF repo)
    try:
        report_texts: List[dict] = []
        seen_urls: set[str] = set()

        # (A) README links
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

        # (B) PDFs included in the model repo itself
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
                        break
                    except Exception:
                        continue

        if save_to_file and report_texts:
            _save_reports_for_model(model_id, output_dir, report_texts)
        elif report_texts:
            print("ℹ️ Technical reports were found but save_to_file=False, so they were not written to disk.")

    except Exception as e:
        print("⚠️ report extraction (HF) failed:", e)

    # Save main JSON
    if save_to_file:
        filename_safe = model_id.replace("/", "_")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"huggingface_{filename_safe}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"✅ JSON file saved: {output_path}")

    return result


if __name__ == "__main__":
    test_model_id = "google/gemma-3-4b-it"
    result = huggingface_fetcher(test_model_id)
    for k, v in result.items():
        print("*" * 30, k)
        print(v)
