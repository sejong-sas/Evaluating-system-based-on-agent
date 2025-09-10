# github_Fetcher.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import requests, json, os, re
from pathlib import Path
import fitz  # PyMuPDF
from urllib.parse import urlparse, urljoin

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
    if s.count("http://") + s.count("https://") >= 2:
        s = s[s.rfind("http"):]
    return s.strip().rstrip('.,;:)]}>"\'')

def _extract_urls(text: str) -> List[str]:
    if not text:
        return []
    raw = re.findall(r'https?://[^\s<>"\'\)\]\}]+', text)
    out: List[str] = []
    seen: set[str] = set()
    for u in raw:
        cu = _clean_url(u)
        if cu and cu not in seen:
            seen.add(cu); out.append(cu)
    return out

# ===== Model-family tokens (for relatedness) ====
_STOPWORDS = {"ai","llm","language","model","models","chat","instruct","sft","rl","eval","hf",
              "release","preview","alpha","beta","rc","v","it"}
def _family_tokens_from_model_id(model_id: str) -> set[str]:
    name = (model_id or "").split("/", 1)[-1].lower()
    raw = re.split(r"[^a-z0-9.]+", name)
    base: set[str] = set()
    for tt in (t.strip() for t in raw):
        if not tt: continue
        if tt in _STOPWORDS: continue
        if len(tt) >= 2: base.add(tt)
        m = re.match(r"([a-z]+)(\d+(?:\.\d+)*)$", tt)
        if m:
            base.add(m.group(1))
            base.add(m.group(1)+m.group(2).replace(".",""))
            base.add(m.group(2)); base.add(m.group(2).replace(".",""))
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
        if t in ul: hits += 2
        if re.search(rf"\b{re.escape(t)}\b", tl): hits += 1
    return hits >= max(1, min_hits)

# ---------------- Report URL detector ----------------
def _is_probable_report_url(u: str) -> bool:
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

    shorteners = {"goo.gle","g.co","bit.ly","t.co","tinyurl.com","ow.ly","lnkd.in","rb.gy","rebrand.ly"}
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

# ---------------- HTML helpers (one-hop version follow) ----------------
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
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u); out.append(u)
    return out

def _maybe_follow_version_page(html: str, base_url: str, model_id: str) -> Tuple[str, str] | None:
    toks = _family_tokens_from_model_id(model_id)
    links = _extract_links_from_html(html, base_url)
    candidates = [u for u in links if any(t in u.lower() for t in toks)]
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

# ---------------- Fetchers for content ----------------
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
    hop = None
    try:
        if model_id:
            hop = _maybe_follow_version_page(html, final_url, model_id)
    except Exception:
        hop = None
    if hop:
        _, hop_text = hop
        text = hop_text
    else:
        text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
        text = re.sub(r"(?is)<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
    return text[:800_000]

# ---------------- Report file IO ----------------
def _save_reports_for_repo(repo_full_name: str, output_dir: str | Path, items: List[dict]) -> Path | None:
    """Save/merge GH-origin into reports_fulltext_github_{owner_repo}.json"""
    if not items:
        return None
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    base = repo_full_name.replace("/", "_").lower()
    out = output_dir / f"reports_fulltext_github_{base}.json"

    existing = []
    if out.exists():
        try:
            existing = (json.load(open(out, encoding="utf-8")).get("full_texts") or [])
        except Exception:
            existing = []

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

    payload = {"repo": repo_full_name, "full_texts": merged}
    json.dump(payload, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"📄 Reports saved/merged (GH): {out}")
    return out

def _merge_hf_and_gh_reports(model_id: str, repo_full_name: str, output_dir: str | Path) -> Path | None:
    """
    Merge:
      - reports_fulltext_huggingface_{model}.json
      - reports_fulltext_github_{owner_repo}.json
    → reports_fulltext_{model}.json  (dedup by (url, content-hash))
    """
    output_dir = Path(output_dir)
    base_model = model_id.replace("/", "_").lower()
    base_repo  = repo_full_name.replace("/", "_").lower()

    hf_fp = output_dir / f"reports_fulltext_huggingface_{base_model}.json"
    gh_fp = output_dir / f"reports_fulltext_github_{base_repo}.json"
    if not hf_fp.exists() and not gh_fp.exists():
        print("ℹ️ No HF/GH reports to merge.")
        return None

    def _load(path: Path) -> List[dict]:
        if path.exists():
            try:
                return (json.load(open(path, encoding="utf-8")).get("full_texts") or [])
            except Exception:
                return []
        return []

    hf_items = _load(hf_fp)
    gh_items = _load(gh_fp)

    seen = set()
    merged: List[dict] = []

    def _key(x: dict) -> Tuple[str, str]:
        aid = (x.get("arxiv_id") or "").strip()
        ft = (x.get("full_text") or "")
        h = str(abs(hash(ft[:512])))
        return (aid, h)

    for lst in [hf_items, gh_items]:
        for it in (lst or []):
            if not isinstance(it, dict): continue
            if not (it.get("arxiv_id") and it.get("full_text")): continue
            k = _key(it)
            if k in seen: continue
            seen.add(k)
            merged.append({"arxiv_id": it["arxiv_id"], "full_text": it["full_text"]})

    out = output_dir / f"reports_fulltext_{base_model}.json"
    payload = {"model_id": model_id, "full_texts": merged}
    json.dump(payload, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"✅ HF+GH reports merged → {out}  (hf:{len(hf_items)} + gh:{len(gh_items)} → {len(merged)})")
    return out

# ---------------- Main GitHub fetcher ----------------
def github_fetcher(repo_full_name: str,
                   branch: str = "main",
                   token: str | None = None,
                   save_to_file: bool = True,
                   output_dir: str | Path = ".",
                   model_id_hint: str | None = None) -> Dict[str, Any]:
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
                # model_id_hint is used only for relatedness / version-follow
                txt = _fetch_pdf_text(u) if u.lower().endswith(".pdf") else _fetch_html_text(u, model_id_hint)
                if txt.strip():
                    if (not model_id_hint) or _looks_related_to_model(txt, u, model_id_hint, min_hits=1):
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
                        if (not model_id_hint) or _looks_related_to_model(txt, raw_pdf, model_id_hint, min_hits=1):
                            report_texts.append({"arxiv_id": raw_pdf, "full_text": txt})
                            seen.add(raw_pdf)
                except Exception as e:
                    print("⚠️ report fetch failed:", raw_pdf, e)

        # Save GH report file
        gh_rep_path = _save_reports_for_repo(repo_full_name, output_dir, report_texts)

        # Try merging with HF reports → combined reports_fulltext_{model}.json
        model_id_for_merge = model_id_hint
        if not model_id_for_merge:
            # infer from README HF links; merge only if exactly one model id
            ids = re.findall(r"https?://huggingface\.co/([\w\-\._]+/[\w\-\._]+)", readme or "", re.IGNORECASE)
            ids = [i.lower() for i in ids if not i.lower().startswith("collections/")]
            ids = list(dict.fromkeys(ids))
            if len(ids) == 1:
                model_id_for_merge = ids[0]

        if model_id_for_merge:
            _merge_hf_and_gh_reports(model_id_for_merge, repo_full_name, output_dir)
        else:
            if gh_rep_path:
                print("ℹ️ Merge skipped: could not determine a single target HF model id.")

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
    # Example: give model_id_hint to enable relatedness checks + HF/GH merge
    info = github_fetcher("google-gemini/gemma-cookbook", branch="main", model_id_hint="google/gemma-2-2b-it")
    for k, v in info.items():
        print("*" * 30, k)
        print(v)
