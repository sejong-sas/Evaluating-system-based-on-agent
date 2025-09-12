# arxiv_Fetcher.py
# -----------------------------------------------------------------------------
# HF tags → arXiv PDFs (+ Tavily Dual Search + GPT verification merge)
#
# Pipeline:
#   (1) Collect arXiv IDs from Hugging Face model tags  ← no GPT verification
#   (2) Run Tavily "dual search" ("paper", "technical report") to collect
#       up to 0–2 arXiv candidates → verify each candidate with GPT using
#       a version-tolerance rule
#   (3) Merge (1) + (2 verified) → download PDFs, extract text → save JSON
#
# .env example:
#   HF_TOKEN=...                  # or HUGGINGFACE_HUB_TOKEN / HUGGINGFACE_TOKEN / HF_API_TOKEN (optional)
#   TAVILY_API_KEY=tvly-...
#   OPENAI_API_KEY=sk-...
#   USE_TAVILY_SEARCH=1           # default on (set 0 to use HF tags only)
#   OPENAI_MODEL_VERIFIER=o3-mini # default verifier model
#   MAX_CHARS_FOR_GPT=50000       # max characters passed to GPT
# -----------------------------------------------------------------------------

import os
import re
import json
import time
import requests
import fitz  # PyMuPDF
from typing import List, Dict, Optional, Set
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# ENV & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv(find_dotenv(), override=False)

def _read_env_token(*names: str) -> str:
    """Return the first non-empty env var among the given names."""
    for n in names:
        v = os.getenv(n)
        if v and v.strip():
            return v.strip()
    return ""

HF_TOKEN              = _read_env_token("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_TOKEN")
TAVILY_API_KEY        = _read_env_token("TAVILY_API_KEY")
OPENAI_API_KEY        = _read_env_token("OPENAI_API_KEY")

USE_TAVILY_SEARCH     = os.getenv("USE_TAVILY_SEARCH", "1") == "1"
OPENAI_MODEL_VERIFIER = os.getenv("OPENAI_MODEL_VERIFIER", "o3-mini")
MAX_CHARS_FOR_GPT     = int(os.getenv("MAX_CHARS_FOR_GPT", "50000"))

_UA = {"User-Agent": "Mozilla/5.0 (Model-Paper-Fetcher)"}
_HF_HEADERS = dict(_UA)
if HF_TOKEN:
    _HF_HEADERS["Authorization"] = f"Bearer {HF_TOKEN}"

def _debug_env():
    if not HF_TOKEN:
        print("ℹ️ HF token not found — gated/private models may return 401/403.")
    if USE_TAVILY_SEARCH and not TAVILY_API_KEY:
        print("❗ USE_TAVILY_SEARCH=1 but TAVILY_API_KEY is missing. Tavily search will be skipped.")
    if USE_TAVILY_SEARCH and TAVILY_API_KEY and not OPENAI_API_KEY:
        print("❗ OPENAI_API_KEY missing — Tavily candidates cannot be GPT-verified and will be skipped.")

# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────
def _get_hf(url: str, timeout: int = 20, allow_redirects: bool = True) -> requests.Response:
    """GET for Hugging Face API with token fallback → anonymous if 401/403."""
    if "Authorization" in _HF_HEADERS:
        r = requests.get(url, headers=_HF_HEADERS, timeout=timeout, allow_redirects=allow_redirects)
        if r.status_code in (401, 403):
            r = requests.get(url, headers=_UA, timeout=timeout, allow_redirects=allow_redirects)
    else:
        r = requests.get(url, headers=_UA, timeout=timeout, allow_redirects=allow_redirects)
    r.raise_for_status()
    return r

def _http_get(url: str, timeout: int = 20) -> requests.Response:
    r = requests.get(url, headers=_UA, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r

# ─────────────────────────────────────────────────────────────────────────────
# Utilities: normalize & dedup
# ─────────────────────────────────────────────────────────────────────────────
def _clean_url(u: str) -> str:
    if not isinstance(u, str):
        return ""
    s = u.strip()
    if s.count("http") >= 2:
        s = s[s.rfind("http"):]
    s = re.sub(r"[?#].*$", "", s)
    return s.strip().strip(".,;:)]}>\"'")

def _normalize_arxiv_id(s: str) -> str:
    """
    Accepts:
      - '2101.01234', '2101.01234v2'
      - 'abs/2101.01234', 'arXiv:2101.01234v3'
      - URLs like 'https://arxiv.org/abs/2101.01234?utm=...'
    Returns canonical '2101.01234' (strips version/query).
    """
    s = (_clean_url(s) or "")
    s = re.sub(r"(?i)^arxiv:", "", s)
    s = re.sub(r"(?i)^(https?://arxiv\.org/)?(abs/|pdf/)", "", s)
    s = re.sub(r"\.pdf$", "", s, flags=re.I)
    s = s.strip("}])>")
    m = re.search(r"(\d{4}\.\d{4,5})", s)
    return m.group(1) if m else s

def _dedup_str(items: List[str]) -> List[str]:
    seen, out = set(), []
    for x in items:
        x = (x or "").strip()
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# (1) HF tags → arXiv IDs  (no GPT verification)
# ─────────────────────────────────────────────────────────────────────────────
def get_all_arxiv_ids(model_id: str) -> List[str]:
    """Extract arXiv IDs from Hugging Face model tags."""
    url = f"https://huggingface.co/api/models/{model_id}"
    resp = _get_hf(url)
    data = resp.json()
    tags = data.get("tags", []) or []
    arxiv_ids: List[str] = []
    for tag in tags:
        if isinstance(tag, str) and tag.lower().startswith("arxiv:"):
            arxiv_ids.append(_normalize_arxiv_id(tag.split(":", 1)[1]))
    ids = _dedup_str([_normalize_arxiv_id(t) for t in arxiv_ids if t])
    print(f"🔎 HF tags found arXiv IDs: {ids}")
    return ids

# ─────────────────────────────────────────────────────────────────────────────
# PDF download & text extraction
# ─────────────────────────────────────────────────────────────────────────────
def download_arxiv_pdf(arxiv_id: str, save_path: Optional[str] = None, output_dir: str | Path = ".") -> str:
    """Download a PDF by arXiv ID to disk and return the saved path."""
    nid = _normalize_arxiv_id(arxiv_id)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not save_path:
        save_path = output_dir / f"arxiv_{nid}.pdf"
    pdf_url = f"https://arxiv.org/pdf/{nid}.pdf"
    r = _http_get(pdf_url, timeout=30)
    with open(save_path, "wb") as f:
        f.write(r.content)
    print(f"📄 PDF saved: {save_path}")
    return str(save_path)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract full text from a local PDF file."""
    text_parts: List[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text_parts.append(page.get_text("text"))
    return "\n".join(text_parts)

def _download_and_extract_pdf_text_in_memory(arxiv_id: str) -> Optional[str]:
    """Download a PDF by arXiv ID (in memory) and return plain text (for GPT verification)."""
    url = f"https://arxiv.org/pdf/{_normalize_arxiv_id(arxiv_id)}.pdf"
    try:
        r = requests.get(url, headers=_UA, timeout=30)
        r.raise_for_status()
        txt = ""
        with fitz.open(stream=r.content, filetype="pdf") as doc:
            for p in doc:
                txt += p.get_text()
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt
    except Exception as e:
        print(f"    - PDF processing failed: {arxiv_id} ({e})")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# (2) Tavily dual search → candidate IDs (before verification)
# ─────────────────────────────────────────────────────────────────────────────
def simplify_model_name(model_id: str) -> str:
    """Simplify a model ID to a 'series + version' form optimized for search."""
    name = model_id.split('/')[-1]
    name = re.sub(r'\d+x\d+[bm]', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\d+(\.\d+)?[bm]', '', name, flags=re.IGNORECASE)
    for w in ['instruct', 'chat', 'base', 'sft', 'it', 'gguf', 'awq', 'gptq']:
        name = re.sub(rf'[-_]?{w}\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[-_]+', ' ', name)
    name = re.sub(r'\bv(?=\d)', '', name, flags=re.IGNORECASE)
    return ' '.join(name.split())

def _tavily_top_arxiv_id(query: str) -> Optional[str]:
    """Issue a Tavily search and return the first arXiv ID found in results (if any)."""
    print(f"🔎 Tavily search: {query}")
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_API_KEY, "query": query, "max_results": 10},
            timeout=20,
            headers=_UA
        )
        r.raise_for_status()
        for res in (r.json().get("results") or []):
            url = res.get("url", "")
            if "arxiv.org" in url:
                m = re.search(r"(\d{4}\.\d{4,5})", url)
                if m:
                    print(f"  → arXiv link found: {url}")
                    return m.group(1)
        print("  → No arXiv link found in results.")
        return None
    except Exception as e:
        print(f"❌ Tavily error: {e}")
        return None

def tavily_dual_search(model_id: str) -> List[str]:
    """
    Use two queries ('paper' and 'technical report') to fetch up to 0–2 arXiv
    candidate IDs from the web. These are NOT yet verified.
    """
    if not (USE_TAVILY_SEARCH and TAVILY_API_KEY):
        return []
    simplified = simplify_model_name(model_id)
    print(f"🔄 Simplified query: '{simplified}'")
    found: Set[str] = set()
    for kw in ["paper", "technical report"]:
        aid = _tavily_top_arxiv_id(f"{simplified} {kw}")
        if aid:
            found.add(_normalize_arxiv_id(aid))
        time.sleep(0.3)
    ids = _dedup_str(list(found))
    print(f"🛰️ Tavily candidates: {ids}")
    return ids

# ─────────────────────────────────────────────────────────────────────────────
# GPT verification (only for Tavily candidates)
# ─────────────────────────────────────────────────────────────────────────────
_openai_client = None
def _get_openai():
    """Lazily initialize the OpenAI client."""
    global _openai_client
    if _openai_client is None and OPENAI_API_KEY:
        try:
            from openai import OpenAI
            _openai_client = OpenAI(api_key=OPENAI_API_KEY)
        except Exception as e:
            print(f"❌ Failed to init OpenAI SDK: {e}")
            _openai_client = None
    return _openai_client

def _verify_with_gpt(model_id: str, paper_full_text: str) -> bool:
    """
    Ask GPT: is this the official/primary technical report for the model?
    Version rule:
      - Same MAJOR (integer) version as target
      - MINOR (decimal) difference ≤ 0.4 is acceptable
        (e.g., target 3.3 → 3.0 to 3.7 are acceptable)
    Responds strictly as JSON: {"is_match": bool, "reason": "..."}.
    """
    client = _get_openai()
    if client is None:
        print("    - Skipping GPT verification (OPENAI_API_KEY missing).")
        return False

    truncated = (paper_full_text or "")[:MAX_CHARS_FOR_GPT]

    system_prompt = (
        "You are an expert AI researcher. Determine if a paper is the official or primary technical report "
        "for a target AI model based ONLY on the provided full text (truncated). "
        "VERSION RULE: A paper counts as a match if its version has the same MAJOR number as the target model "
        "and the MINOR difference is ≤ 0.4. (Example: target 3.3 → 3.0–3.7 acceptable.) "
        "Respond STRICTLY as JSON: {\"is_match\": <true|false>, \"reason\": \"...\"}."
    )
    user_prompt = (
        f"Target Model ID: \"{model_id}\"\n\n"
        f"Candidate Paper Full Text (truncated): \"{truncated}...\"\n\n"
        "Is this the official/primary paper for the target model under the VERSION RULE?"
    )

    try:
        rsp = client.chat.completions.create(
            model=OPENAI_MODEL_VERIFIER,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        content = rsp.choices[0].message.content
        data = {}
        try:
            data = json.loads(content)
        except Exception:
            # Fallback: coarse check when model returns plain text
            data = {"is_match": "true" in (content or "").lower(), "reason": content}
        is_match = bool(data.get("is_match", False))
        reason = str(data.get("reason", "") or "").strip()
        print(f"    - GPT verdict: {'✅ match' if is_match else '❌ no match'} ({reason[:200]})")
        return is_match
    except Exception as e:
        print(f"    - GPT verification error: {e}")
        return False

def verify_tavily_candidates(model_id: str, candidate_ids: List[str]) -> List[str]:
    """Download each candidate's PDF text and keep only those that pass GPT verification."""
    if not (candidate_ids and OPENAI_API_KEY):
        return []
    passed: List[str] = []
    print(f"🔬 Verifying {len(candidate_ids)} Tavily candidate(s) with GPT…")
    for aid in candidate_ids:
        print(f"  • Candidate: {aid}")
        full_text = _download_and_extract_pdf_text_in_memory(aid)
        if not full_text:
            print("    - Failed to extract text → skip")
            continue
        if _verify_with_gpt(model_id, full_text):
            passed.append(_normalize_arxiv_id(aid))
        time.sleep(0.3)
    print(f"✅ GPT-verified IDs: {passed}")
    return passed

# ─────────────────────────────────────────────────────────────────────────────
# Public: explicit IDs → full text JSON
# ─────────────────────────────────────────────────────────────────────────────
def arxiv_fetcher(arxiv_ids: List[str], save_to_file: bool = True, output_dir: str | Path = ".", label: str = "") -> List[Dict[str, str]]:
    """
    For each arXiv ID, download the PDF, extract text, and collect
    [{"arxiv_id": "...", "full_text": "..."}]. Optionally save as a single JSON.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, str]] = []
    for aid in arxiv_ids:
        try:
            pdf_path = download_arxiv_pdf(aid, output_dir=output_dir)
            full_text = extract_text_from_pdf(pdf_path)
            results.append({"arxiv_id": _normalize_arxiv_id(aid), "full_text": full_text})
        except Exception as e:
            print(f"⚠️ Failed to process arXiv {aid}: {e}")
            # Fallback: try to store abstract if PDF fails
            try:
                abs_url = f"https://arxiv.org/abs/{_normalize_arxiv_id(aid)}"
                r = _http_get(abs_url, timeout=15)
                m = re.search(r'(?is)<span class="abstract-full.*?>(.*?)</span>', r.text)
                summary = ""
                if m:
                    summary = re.sub(r"<[^>]+>", " ", m.group(1))
                    summary = re.sub(r"\s+", " ", summary).strip()
                if summary:
                    results.append({"arxiv_id": _normalize_arxiv_id(aid), "full_text": summary})
            except Exception:
                pass

    if save_to_file and label:
        base = label.replace("/", "_").lower()
        fp = output_dir / f"arxiv_fulltext_{base}.json"
        with open(fp, "w", encoding="utf-8") as f:
            json.dump({"model_id": label, "full_texts": results}, f, indent=2, ensure_ascii=False)
        print(f"✅ Full paper text saved: {fp}")
    return results

# ─────────────────────────────────────────────────────────────────────────────
# (3) Orchestrator: HF + Tavily(+GPT) → JSON
# ─────────────────────────────────────────────────────────────────────────────
def arxiv_fetcher_from_model(model_id: str, save_to_file: bool = True, output_dir: str | Path = ".") -> List[Dict[str, str]]:
    """
    Orchestrates the full pipeline:
      1) Collect HF tag arXiv IDs (no verification)
      2) Tavily dual search → candidates not already in HF → GPT verify
      3) Merge & dedup → download PDFs, extract text → save JSON
    """
    _debug_env()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) HF tags
    hf_ids = get_all_arxiv_ids(model_id)

    # 2) Tavily → exclude IDs already found in HF → GPT verification
    tavily_candidates = tavily_dual_search(model_id)
    tavily_candidates = [i for i in tavily_candidates if i not in set(hf_ids)]
    verified_tavily = verify_tavily_candidates(model_id, tavily_candidates) if tavily_candidates else []

    # 3) Merge and fetch
    all_ids = _dedup_str(hf_ids + verified_tavily)
    print(f"📦 Final merged arXiv IDs: {all_ids}")
    if not all_ids:
        print(f"❌ No arXiv ID found/verified for '{model_id}'.")
        return []

    return arxiv_fetcher(all_ids, save_to_file=save_to_file, output_dir=output_dir, label=model_id)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    mid = "meta-llama/llama-3.3-70b-instruct"  # default example
    if len(sys.argv) > 1 and sys.argv[1]:
        mid = sys.argv[1]
    outdir = "."
    if len(sys.argv) > 2 and sys.argv[2]:
        outdir = sys.argv[2]
    print("▶ Model:", mid)
    print("▶ Output folder:", outdir)
    arxiv_fetcher_from_model(mid, output_dir=outdir)
