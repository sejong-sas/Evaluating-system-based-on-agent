# arxiv_Fetcher.py
import os
import requests
import re
import json
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional
from pathlib import Path

# ─────────── HF GET with token (try token → fallback anonymous) ───────────
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

# ─────────── utils ───────────
def _clean_url(u: str) -> str:
    if not isinstance(u, str):
        return ""
    s = u.strip()
    if s.count("http") >= 2:
        idx = s.rfind("http")
        s = s[idx:]
    s = s.strip().strip(".,;:)]}>\"'")
    return s

# ─────────── arXiv helpers ───────────
def _normalize_arxiv_id(s: str) -> str:
    """
    Accepts forms like '2101.01234', '2101.01234v2', 'abs/2101.01234', 'arXiv:2101.01234v3'
    → returns '2101.01234' (version stripped).
    """
    s = _clean_url(s)
    s = re.sub(r"(?i)^arxiv:", "", s)
    s = re.sub(r"(?i)^(abs/|pdf/)", "", s)
    s = s.replace(".pdf", "")
    s = s.strip("}])>")
    m = re.match(r"(\d{4}\.\d{4,5})", s)
    return m.group(1) if m else s

def get_all_arxiv_ids(model_id: str) -> List[str]:
    """Extract all arXiv IDs from Hugging Face model tags."""
    url = f"https://huggingface.co/api/models/{model_id}"
    resp = _get_hf(url)
    resp.raise_for_status()
    data = resp.json()
    tags = data.get("tags", []) or []
    arxiv_ids: List[str] = []
    for tag in tags:
        if isinstance(tag, str) and tag.lower().startswith("arxiv:"):
            arxiv_ids.append(_normalize_arxiv_id(tag.split(":",1)[1]))
    # de-dup while preserving order
    seen, out = set(), []
    for t in arxiv_ids:
        t = _normalize_arxiv_id(t)
        if t and t not in seen:
            seen.add(t); out.append(t)
    return out

def download_arxiv_pdf(arxiv_id: str, save_path: Optional[str] = None, output_dir: str | Path = ".") -> str:
    """
    Download arXiv PDF. Tries canonical pdf URL. Uses UA + timeout.
    """
    nid = _normalize_arxiv_id(arxiv_id)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not save_path:
        save_path = output_dir / f"arxiv_{nid}.pdf"

    pdf_url = f"https://arxiv.org/pdf/{nid}.pdf"
    r = requests.get(pdf_url, headers=_UA, timeout=25, allow_redirects=True)
    r.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(r.content)
    print(f"📄 PDF saved: {save_path}")
    return str(save_path)

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Robust PDF → text using PyMuPDF.
    """
    text_parts: List[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text_parts.append(page.get_text("text"))
    return "\n".join(text_parts)

def arxiv_fetcher(arxiv_ids: List[str], save_to_file: bool = True, output_dir: str | Path = ".", label: str = "") -> List[Dict[str, str]]:
    """
    Fetch raw texts for given arXiv IDs list.
    label: optional; if provided, used in output filename suffix.
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

    if save_to_file and label:
        base = label.replace("/", "_").lower()
        fp = output_dir / f"arxiv_fulltext_{base}.json"
        with open(fp, "w", encoding="utf-8") as f:
            json.dump({"model_id": label, "full_texts": results}, f, indent=2, ensure_ascii=False)
        print(f"✅ Full paper text saved: {fp}")
    return results

def arxiv_fetcher_from_model(model_id: str, save_to_file: bool = True, output_dir: str | Path = ".") -> List[Dict[str, str]]:
    """
    From a HF model ID, find 'arxiv:' tags → pull PDFs → extract text → save:
      arxiv_fulltext_{org_model}.json
    """
    arxiv_ids = get_all_arxiv_ids(model_id)
    if not arxiv_ids:
        print(f"❌ No arXiv ID found for '{model_id}'.")
        return []
    return arxiv_fetcher(arxiv_ids, save_to_file=save_to_file, output_dir=output_dir, label=model_id)

if __name__ == "__main__":
    arxiv_fetcher_from_model("deepseek-ai/DeepSeek-R1")
