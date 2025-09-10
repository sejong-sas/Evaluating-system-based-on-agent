# huggingface_Fetcher.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import requests, json, os, re
from pathlib import Path
import fitz  # PyMuPDF
from urllib.parse import urlparse, urljoin

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
    Robust URL extractor for Markdown/HTML (README).
    - Avoid trailing punctuation/brackets that cause 404s
    - Keep the last full 'http...' chunk if multiple got concatenated
    - De-duplicate preserving order
    """
    if not text:
        return []
    raw = re.findall(r'https?://[^\s<>"\'\)\]\}]+', text)
    out: List[str] = []
    seen: set[str] = set()
    for u in raw:
        cu = _clean_url(u)
        if cu and cu not in seen:
            seen.add(cu)
            out.append(cu)
    return out

# ===== Model-family tokens (for version-aware following & relatedness) ====
_STOPWORDS = {
    "ai","llm","language","model","models","chat","instruct","sft","rl","eval","hf",
    "release","preview","alpha","beta","rc","v","it"
}
def _family_tokens_from_model_id(model_id: str) -> set[str]:
    name = (model_id or "").split("/", 1)[-1].lower()
    raw = re.split(r"[^a-z0-9.]+", name)
    base: set[str] = set()
    for tt in (t.strip() for t in raw):
        if not tt: continue
        if tt in _STOPWORDS: continue
        if len(tt) >= 2: base.add(tt)
        m = re.match(r"([a-z]+)(\d+(?:\.\d+)*)$", tt)  # e.g., llama3 / llama3.1
        if m:
            base.add(m.group(1))
            base.add(m.group(1)+m.group(2).replace(".",""))
            base.add(m.group(2))              # "3.1"
            base.add(m.group(2).replace(".",""))  # "31"
    joined = re.sub(r"[^a-z0-9]", "", name)
    nodigit = re.sub(r"\d+", "", joined)
    if len(joined) >= 3: base.add(joined)
    if len(nodigit) >= 3: base.add(nodigit)
    return base

def _looks_related_to_model(text: str, url: str, model_id: str, min_hits: int = 1) -> bool:
    toks = _family_tokens_from_model_id(model_id)
    tl = (text or "").lower()
    ul = (url or "").lower()
    hits = 0
    for t in toks:
        if not t: continue
        if t in ul: hits += 2  # URL hit is strong
        if re.search(rf"\b{re.escape(t)}\b", tl): hits += 1
    return hits >= max(1, min_hits)

# ----------------------------- Report URL detector -----------------------------
def _is_probable_report_url(u: str) -> bool:
    """
    Generic detector for 'report-like' or 'paper-like' links — model/vendor agnostic.
    NOTE: We do NOT parse HF top badges; we only look at README links (caller controls that).
    """
    if not isinstance(u, str) or not u:
        return False
    ul = u.lower()
    if ul.endswith(".pdf"):
        return True
    try:
        parsed = urlparse(u)
        host = (parsed.netloc or "").lower()
        path = (parsed.path or "").lower()
    except Exception:
        return False

    shorteners = {
        "goo.gle","g.co","bit.ly","t.co","tinyurl.com","ow.ly","lnkd.in","rb.gy","rebrand.ly"
    }
    if host in shorteners:
        return True

    scholarly_hosts = (
        "arxiv.org","openreview.net","aclanthology.org","ieeexplore.ieee.org","dl.acm.org",
        "papers.nips.cc","proceedings.mlr.press","hal.science","biorxiv.org","medrxiv.org",
        "arxiv-vanity.com"
    )
    if any(h in host for h in scholarly_hosts):
        return True

    path_tokens = (
        "/paper","/papers","/publication","/publications",
        "whitepaper","white-paper","technical-report","techreport","tech-report",
        "/docs","/documentation","/research","/resources/",
        "/post/","/posts/","/blog/","/news/",
        "announce","announc","release","report"
    )
    if any(tok in path for tok in path_tokens):
        return True

    if re.search(r"/(paper|report|whitepaper|technical[-_]report)s?/?$", path):
        return True

    return False

# ----------------------------- HTML helpers -----------------------------
_HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.I)
def _extract_links_from_html(html: str, base_url: str) -> List[str]:
    urls = []
    for href in _HREF_RE.findall(html or ""):
        try:
            full = urljoin(base_url, href.strip())
            if full.startswith("http"):
                urls.append(_clean_url(full))
        except Exception:
            continue
    # de-dup preserve order
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u); out.append(u)
    return out

def _maybe_follow_version_page(html: str, base_url: str, model_id: str) -> Tuple[str, str] | None:
    """
    If the page is a family/overview page, try to find a version-specific page
    matching our model tokens and follow one hop.
    Returns (url, text) if followed, else None.
    """
    toks = _family_tokens_from_model_id(model_id)
    links = _extract_links_from_html(html, base_url)
    # Prefer links with any token in URL or anchor-ish words
    candidates = [u for u in links if any(t in u.lower() for t in toks)]
    # heuristics: also allow 'model card' / 'model page'
    if not candidates:
        extra = [u for u in links if re.search(r"model\s+(card|page)|documentation|docs", u, re.I)]
        candidates = extra[:5]
    for u in candidates[:10]:
        try:
            r = requests.get(u, headers=_UA, timeout=20, allow_redirects=True)
            r.raise_for_status()
            ct = (r.headers.get("Content-Type") or "").lower()
            fin = (r.url or "").lower()
            if ("pdf" in ct) or fin.endswith(".pdf"):
                with fitz.open(stream=r.content, filetype="pdf") as doc:
                    txt = "\n".join(p.get_text("text") for p in doc)
            else:
                txt = r.text
                txt = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", txt)
                txt = re.sub(r"(?is)<[^>]+>", " ", txt)
                txt = re.sub(r"\s+", " ", txt)
            if _looks_related_to_model(txt, u, model_id):
                return (u, txt[:800_000])
        except Exception:
            continue
    return None

# ----------------------------- Fetchers for content -----------------------------
def _fetch_pdf_text(url: str) -> str:
    r = requests.get(url, headers=_UA, timeout=25)
    r.raise_for_status()
    with fitz.open(stream=r.content, filetype="pdf") as doc:
        return "\n".join(p.get_text("text") for p in doc)

def _fetch_html_text(url: str, model_id: str | None = None) -> str:
    r = requests.get(url, headers=_UA, timeout=20, allow_redirects=True)
    r.raise_for_status()
    ct = (r.headers.get("Content-Type") or "").lower()
    final_url = (r.url or "").lower()
    if ("pdf" in ct) or final_url.endswith(".pdf"):
        with fitz.open(stream=r.content, filetype="pdf") as doc:
            return "\n".join(p.get_text("text") for p in doc)
    html = r.text
    # optional one-hop follow to version-specific page
    hop = None
    try:
        if model_id:
            hop = _maybe_follow_version_page(html, final_url, model_id)
    except Exception:
        hop = None
    if hop:
        _, hop_text = hop
        html = hop_text
    else:
        html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
        html = re.sub(r"(?is)<[^>]+>", " ", html)
        html = re.sub(r"\s+", " ", html)
    return html[:800_000]

# ----------------------------- Report file IO -----------------------------
def _save_reports_for_model(model_id: str, output_dir: str | Path, items: List[dict]) -> Path | None:
    """Save/merge HF-origin reports into reports_fulltext_huggingface_{base}.json"""
    if not items:
        return None
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    base = model_id.replace("/", "_").lower()
    out = output_dir / f"reports_fulltext_huggingface_{base}.json"

    existing = []
    if out.exists():
        try:
            existing = (json.load(open(out, encoding="utf-8")).get("full_texts") or [])
        except Exception:
            existing = []

    # de-dup by (arxiv_id, first 64 chars hash) to be safe
    seen = set()
    merged: List[dict] = []
    def _key(x: dict) -> Tuple[str, str]:
        aid = (x.get("arxiv_id") or "").strip()
        ft = (x.get("full_text") or "")
        h = str(abs(hash(ft[:512])))
        return (aid, h)

    for lst in [existing, items]:
        for it in (lst or []):
            if not isinstance(it, dict): continue
            if not (it.get("arxiv_id") and it.get("full_text")): continue
            k = _key(it)
            if k in seen: continue
            seen.add(k); merged.append({"arxiv_id": it["arxiv_id"], "full_text": it["full_text"]})

    payload = {"model_id": model_id, "full_texts": merged}
    json.dump(payload, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"📄 Reports saved/merged (HF): {out}")
    return out

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

    # README variants (only README is used for link crawling; badges are ignored)
    readme = ""
    for cand in ["README.md","README.MD","README","Readme.md","readme.md"]:
        if cand in siblings:
            readme = fetch_raw(cand)
            break

    # Key JSONs
    config = fetch_raw("config.json") if "config.json" in siblings else ""
    generation_config = fetch_raw("generation_config.json") if "generation_config.json" in siblings else ""

    # LICENSE*
    license_file = ""
    lic_cands = [fn for fn in siblings if fn.upper().startswith("LICENSE")]
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

        # (A) README links (not badges)
        for u in _extract_urls(readme):
            if not _is_probable_report_url(u):
                continue
            if u in seen_urls:
                continue
            try:
                txt = _fetch_pdf_text(u) if u.lower().endswith(".pdf") else _fetch_html_text(u, model_id=model_id)
                if txt.strip() and _looks_related_to_model(txt, u, model_id, min_hits=1):
                    report_texts.append({"arxiv_id": u, "full_text": txt})
                    seen_urls.add(u)
            except Exception as e:
                print("⚠️ report fetch failed:", u, e)

        # (B) PDFs included in the model repo itself
        for fn in siblings:
            if fn.lower().endswith(".pdf"):
                for br in ["main", "refs/convert", "refs/pr/1"]:
                    url = f"https://huggingface.co/{model_id}/resolve/{br}/{fn}"
                    if url in seen_urls: break
                    try:
                        txt = _fetch_pdf_text(url)
                        if txt.strip() and _looks_related_to_model(txt, url, model_id, min_hits=1):
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
        print("*"*30, k)
        print(v)
