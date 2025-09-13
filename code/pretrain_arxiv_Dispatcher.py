# -*- coding: utf-8 -*-
"""
arXiv dispatcher (BASE model) with HF-paper + Web-search augmentation.
- Load local arxiv_fulltext_{base}.json / arxiv_{base}.json when available
- PLUS (optional) harvest paper/report links from HF card/README
- PLUS (optional) Tavily web search for arXiv candidates, then GPT verify with version rule:
    * same major version (integer part) AND |minor_diff| <= PRETRAIN_VERSION_TOLERANCE (default 0.4)

Outputs:
  pretrain_arxiv_{base}.json
  {
    "pretrain_method": "...",
    "pretrain_data":   "...",
    "__evidence":      [ { "source": "arxiv:<id>|web:<url>", "quote": "..." }, ... ]
  }
"""

import os, json, re, time
from typing import Dict, Any, List, Optional, Tuple, Set
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import requests

# ───────── Env & knobs ─────────
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

MODEL_NAME = os.getenv("OPENAI_MODEL_PRETRAIN_DISPATCHER", "o3-mini")
OPENAI_MODEL_PRETRAIN_VERIFIER = os.getenv("OPENAI_MODEL_PRETRAIN_VERIFIER", MODEL_NAME)
PRETRAIN_VERSION_TOLERANCE = float(os.getenv("PRETRAIN_VERSION_TOLERANCE", "0.4"))

# augmentation toggles
ENABLE_HF_PAPER_LINKS = os.getenv("PRETRAIN_ENABLE_HF_PAPER_LINKS", "1") == "1"
ENABLE_WEB_SEARCH      = os.getenv("PRETRAIN_ENABLE_WEB_SEARCH", "1") == "1"
ENABLE_GPT_VERIFY      = os.getenv("PRETRAIN_VERIFY_WITH_GPT", "1") == "1"

# evidence/recall chunking
CHUNK_CHARS   = 60_000
CHUNK_OVERLAP = 2_000
DOC_HITS_THRESHOLD = int(os.getenv("PRETRAIN_DOC_HITS", "3"))

# text caps
PER_ARTICLE_CHAR_CAP   = int(os.getenv("PR_PER_ARTICLE_CHAR_CAP", "200000"))
TOTAL_PAYLOAD_CHAR_CAP = int(os.getenv("PR_TOTAL_PAYLOAD_CHAR_CAP", "900000"))
MAX_ARTICLES_PER_CHUNK = int(os.getenv("PR_MAX_ARTICLES_PER_CHUNK", "8"))

# HTTP
USER_AGENT = {"User-Agent": "Mozilla/5.0 (pretrain-arxiv-dispatcher)"}

_client = OpenAI(api_key=API_KEY)

# ───────── Tokens/guards ─────────
_STOPWORDS = {
    "ai","llm","language","model","models","chat","instruct","base","it",
    "sft","rl","eval","preview","alpha","beta","rc","release","v"
}

def _family_tokens_from_model_id(model_id: str) -> List[str]:
    name = (model_id or "").split("/", 1)[-1].lower()
    raw  = re.split(r"[^a-z0-9.]+", name)
    toks = set()
    for t in raw:
        t = t.strip()
        if not t or t in _STOPWORDS:
            continue
        toks.add(t)
        m = re.match(r"([a-z]+)(\d+(?:\.\d+)*)$", t)
        if m:
            head, ver = m.group(1), m.group(2)
            toks.add(head)
            toks.add(head + ver.replace(".", ""))
            toks.add(ver)
            toks.add(ver.replace(".", ""))
    joined  = re.sub(r"[^a-z0-9]", "", name)
    nodigit = re.sub(r"\d+", "", joined)
    if len(joined)  >= 3: toks.add(joined)
    if len(nodigit) >= 3: toks.add(nodigit)
    return sorted(toks)

def _quote_mentions_target(q: str, model_id: str) -> bool:
    if not q:
        return False
    ql = q.lower().replace("–","-").replace("—","-")
    for t in _family_tokens_from_model_id(model_id):
        if len(t) >= 2 and t in ql:
            return True
    return False

def _section_is_about_target(text: str, model_id: str, threshold: int = DOC_HITS_THRESHOLD) -> bool:
    if not text:
        return False
    tl = text.lower()
    hits = 0
    for t in _family_tokens_from_model_id(model_id):
        if not t: continue
        hits += tl.count(t)
        if re.search(rf"\b{re.escape(t)}\b", tl):
            hits += 1
    return hits >= max(1, threshold)

def _strict_guard_text(model_id: str) -> str:
    toks = _family_tokens_from_model_id(model_id)
    return (
        "STRICT MODEL FILTER (sentence-level preferred; section-level backstop applies later)\n"
        f"- Target model: {model_id}\n"
        f"- Prefer quotes where the sentence explicitly mentions one of: {toks}.\n"
        "- If a section is clearly about the TARGET, accept quotes without explicit tokens."
    )

def _STRICT_SUMMARY_GUARD(model_id: str) -> str:
    return _strict_guard_text(model_id) + "\nUse ONLY the provided quotes."

# ───────── Utility: chunk & payload ─────────
def _chunk(t: str) -> List[str]:
    out=[]; n=len(t or ""); i=0
    while i<n:
        end=min(i+CHUNK_CHARS,n)
        out.append(t[i:end])
        if end==n: break
        i=end-CHUNK_OVERLAP if end-CHUNK_OVERLAP>i else end
    return out

def _load_json_guess(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.load(open(path, encoding="utf-8"))
    alt = Path(path.name)
    if alt.exists():
        return json.load(open(alt, encoding="utf-8"))
    raise FileNotFoundError(str(path))

def _payload_text(sections: Dict[str,str]) -> str:
    parts = []
    for tag, val in sections.items():
        if not isinstance(val, str):
            val = json.dumps(val, ensure_ascii=False)
        parts.append(f"[{tag}]\n{val}\n")
    return "\n".join(parts)

# ───────── Local arXiv sections from files ─────────
def _extract_sections(arxiv_full_json: Dict[str, Any]) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    # preferred multi-doc schema
    for it in (arxiv_full_json.get("full_texts") or []):
        aid = str(it.get("arxiv_id") or "")[:2000]
        txt = str(it.get("full_text") or it.get("pdf_text") or "")
        if not txt: continue
        tag = f"arxiv:{aid}" if aid else "arxiv"
        sections[tag] = txt
    # single-doc fallback (rare)
    if not sections and ("full_text" in arxiv_full_json or "pdf_text" in arxiv_full_json):
        aid = str(arxiv_full_json.get("arxiv_id") or "")[:2000]
        txt = str(arxiv_full_json.get("full_text") or arxiv_full_json.get("pdf_text") or "")
        if txt:
            tag = f"arxiv:{aid}" if aid else "arxiv"
            sections[tag] = txt
    return sections

# ───────── HF 카드/README에서 논문 링크 수확 ─────────
_LINK_PAT = re.compile(r"https?://[^\s)>\"]+", re.IGNORECASE)

def _hf_card_and_readme(hf_id: str, max_len: int = 120_000) -> str:
    txt = ""
    # HF model card API
    try:
        r = requests.get(f"https://huggingface.co/api/models/{hf_id}?full=true",
                         timeout=15, headers=USER_AGENT)
        if r.ok:
            card = (r.json() or {}).get("cardData", {}) or {}
            txt += (card.get("content") or "")[:max_len]
    except Exception:
        pass
    # README raw
    for br in ("main", "master"):
        try:
            rr = requests.get(f"https://huggingface.co/{hf_id}/raw/{br}/README.md",
                              timeout=12, headers=USER_AGENT)
            if rr.status_code == 200:
                txt += "\n\n" + rr.text[:max_len]
                break
        except Exception:
            pass
    return txt

def _looks_report_like(u: str) -> bool:
    ul = (u or "").lower()
    return (
        ul.endswith(".pdf") or "arxiv.org" in ul or
        "technical-report" in ul or "tech-report" in ul or
        "whitepaper" in ul or "white-paper" in ul or
        "/paper" in ul or "/blog" in ul or "blog." in ul or
        "/research" in ul or "research." in ul or
        "/docs" in ul or "docs." in ul
    )

_ARXIV_ID_PAT = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")

def _extract_arxiv_id(s: str) -> Optional[str]:
    m = _ARXIV_ID_PAT.search(s or "")
    return m.group(1) if m else None

# ───────── PDF/HTML fetch ─────────
def _fetch_pdf_text(url: str, cap: int) -> str:
    try:
        import fitz  # PyMuPDF
    except Exception:
        return ""
    try:
        r = requests.get(url, timeout=25, headers=USER_AGENT)
        r.raise_for_status()
        with fitz.open(stream=r.content, filetype="pdf") as doc:
            t = "\n".join(p.get_text() for p in doc)
        t = re.sub(r"\s+", " ", t)
        return t[:cap]
    except Exception:
        return ""

def _fetch_html_text(url: str, cap: int) -> str:
    try:
        r = requests.get(url, timeout=20, headers=USER_AGENT)
        r.raise_for_status()
        h = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", r.text)
        t = re.sub(r"(?is)<[^>]+>", " ", h)
        t = re.sub(r"\s+", " ", t)
        return t[:cap]
    except Exception:
        return ""

# ───────── Version-aware GPT verifier ─────────
def _simplify_model_name(model_id: str) -> str:
    name = model_id.split("/")[-1]
    name = re.sub(r"\d+x\d+[bm]", "", name, flags=re.I)
    name = re.sub(r"\d+(\.\d+)?[bm]", "", name, flags=re.I)
    for w in ("instruct","chat","base","sft","it","gguf","awq","gptq"):
        name = re.sub(rf"[-_]?{w}\b", "", name, flags=re.I)
    name = re.sub(r"[-_]+"," ", name)
    name = re.sub(r"\bv(?=\d)","", name, flags=re.I)
    return " ".join(name.split())

def _verify_with_gpt(model_id: str, paper_full_text: str) -> bool:
    if not ENABLE_GPT_VERIFY:
        return True
    if not API_KEY:
        return False
    truncated = (paper_full_text or "")[: min(len(paper_full_text), 50_000)]
    system_prompt = (
        "You are an expert AI researcher. Decide if a given paper is the official/primary "
        "technical report for the TARGET model using ONLY the provided full text. "
        "Version rule: it's a match if the major (integer) version is the same as the target's, "
        f"and the absolute difference in the minor (decimal) version is <= {PRETRAIN_VERSION_TOLERANCE}. "
        "Example: target 3.3 → 3.0..3.7 acceptable. Respond JSON: "
        '{"is_match": boolean, "reason": "brief"}'
    )
    user_prompt = f"Target Model ID: \"{model_id}\"\n\nFull Text (truncated): \"{truncated}...\""
    try:
        rsp = _client.chat.completions.create(
            model=OPENAI_MODEL_PRETRAIN_VERIFIER,
            reasoning_effort="medium",
            response_format={"type":"json_object"},
            messages=[{"role":"system","content":system_prompt},
                      {"role":"user","content":user_prompt}]
        )
        obj = json.loads(rsp.choices[0].message.content)
        return bool(obj.get("is_match", False))
    except Exception:
        return False

# ───────── Tavily search for arXiv ids ─────────
def _search_arxiv_ids_with_tavily(model_id: str, max_results: int = 5) -> Set[str]:
    ids: Set[str] = set()
    if not (ENABLE_WEB_SEARCH and TAVILY_API_KEY):
        return ids
    simplified = _simplify_model_name(model_id)
    for kw in ("paper", "technical report"):
        q = f"{simplified} {kw}"
        try:
            r = requests.post("https://api.tavily.com/search",
                              json={"api_key": TAVILY_API_KEY, "query": q, "max_results": max_results},
                              timeout=20)
            r.raise_for_status()
            for it in r.json().get("results", []):
                u = it.get("url","")
                m = _extract_arxiv_id(u)
                if m:
                    ids.add(m)
        except Exception:
            continue
    return ids

def _download_arxiv_pdf_text(arxiv_id: str) -> str:
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    return _fetch_pdf_text(url, PER_ARTICLE_CHAR_CAP)

# ───────── Augmentation: HF links + Web search → verified texts ─────────
def _augment_sections_with_hf_and_web(model_id: str, sections: Dict[str,str]) -> Dict[str,str]:
    # HF card/README links
    if ENABLE_HF_PAPER_LINKS:
        md = _hf_card_and_readme(model_id)
        if md:
            urls = list(dict.fromkeys(_LINK_PAT.findall(md)))
            for u in urls:
                if not _looks_report_like(u):
                    continue
                # prefer arXiv ids; fall back to direct pdf/html
                aid = _extract_arxiv_id(u)
                text = ""
                tag = ""
                if aid:
                    if f"arxiv:{aid}" in sections:
                        continue
                    text = _download_arxiv_pdf_text(aid)
                    tag = f"arxiv:{aid}"
                else:
                    text = _fetch_pdf_text(u, PER_ARTICLE_CHAR_CAP) if u.lower().endswith(".pdf") else _fetch_html_text(u, PER_ARTICLE_CHAR_CAP)
                    tag = f"web:{u}"
                if text and _verify_with_gpt(model_id, text):
                    sections[tag] = text

    # Web search (Tavily → arXiv ids → download → verify)
    if ENABLE_WEB_SEARCH and TAVILY_API_KEY:
        ids = _search_arxiv_ids_with_tavily(model_id, max_results=5)
        for aid in ids:
            tag = f"arxiv:{aid}"
            if tag in sections:
                continue
            txt = _download_arxiv_pdf_text(aid)
            if txt and _verify_with_gpt(model_id, txt):
                sections[tag] = txt

    return sections

# ───────── Prompts ─────────
_RECALL_SYS = """
You are an assistant for AI model evaluation.
Using only the provided payload, extract evidence about the TARGET model's:
- pre-training METHOD (how pre-training was done)
- pre-training DATA   (with what data)

Return JSON ONLY with EXACTLY these keys:
{
  "pretrain_method_evidence": [ { "source": "arxiv:<id>|web:<url>", "quote": "..." } ],
  "pretrain_data_evidence":   [ { "source": "arxiv:<id>|web:<url>", "quote": "..." } ]
}

Rules:
- 'quote' must be a verbatim sentence copied from the payload.
- 'source' must be one of the provided section tags (e.g., [arxiv:<id>] or [web:<url>]).
- If no evidence, use [].
""".strip()

_SUMMARY_SYS = """
Write concise but detailed English summaries using ONLY the provided quotes.
Return JSON ONLY:
{ "pretrain_method": "...", "pretrain_data": "..." }
If no information, write "No information".
""".strip()

# ───────── Recall → filter → summarize ─────────
def _recall(sections: Dict[str,str], model_id: str) -> Dict[str, List[Dict[str,str]]]:
    text = _payload_text(sections)
    agg_method: List[Dict[str,str]] = []
    agg_data:   List[Dict[str,str]] = []

    for ch in _chunk(text):
        msg = _strict_guard_text(model_id) + "\n\n" + _RECALL_SYS + "\n\n=== PAYLOAD ===\n" + ch
        try:
            rsp = _client.chat.completions.create(
                model=MODEL_NAME,
                reasoning_effort="medium",
                response_format={"type":"json_object"},
                messages=[
                    {"role":"system","content":"Return JSON only."},
                    {"role":"user","content":msg}
                ],
            )
            obj = json.loads(rsp.choices[0].message.content.strip())
        except Exception:
            obj = {}

        for k, bucket in (("pretrain_method_evidence", agg_method),
                          ("pretrain_data_evidence",   agg_data)):
            arr = obj.get(k, []) or []
            if isinstance(arr, list):
                for e in arr:
                    if not isinstance(e, dict): continue
                    src = str(e.get("source","")).strip()
                    qt  = str(e.get("quote","")).strip()
                    if not src or not qt: continue
                    if src not in sections:
                        continue
                    # Accept if sentence mentions target OR the whole section is on-topic
                    accept = _quote_mentions_target(qt, model_id) or _section_is_about_target(sections[src], model_id)
                    if not accept:
                        continue
                    bucket.append({"source": src, "quote": qt})

    # dedup
    def _dedup(lst: List[Dict[str,str]]) -> List[Dict[str,str]]:
        seen=set(); out=[]
        for e in lst:
            key=(e["source"], e["quote"])
            if key in seen: continue
            seen.add(key); out.append(e)
        return out

    return {
        "pretrain_method_evidence": _dedup(agg_method),
        "pretrain_data_evidence":   _dedup(agg_data),
    }

def _summarize(evs: Dict[str, List[Dict[str,str]]], model_id: str) -> Dict[str,str]:
    quotes = {
        "pretrain_method": [e["quote"] for e in evs.get("pretrain_method_evidence",[])],
        "pretrain_data":   [e["quote"] for e in evs.get("pretrain_data_evidence",[])],
    }
    try:
        rsp = _client.chat.completions.create(
            model=MODEL_NAME,
            reasoning_effort="medium",
            response_format={"type":"json_object"},
            messages=[
                {"role":"system","content":_STRICT_SUMMARY_GUARD(model_id)},
                {"role":"user","content":_SUMMARY_SYS + "\n\n=== QUOTES ===\n" +
                                       json.dumps(quotes, ensure_ascii=False, indent=2)},
            ],
        )
        return json.loads(rsp.choices[0].message.content.strip())
    except Exception:
        return {"pretrain_method":"No information","pretrain_data":"No information"}

# ───────── Public API ─────────
def filter_pretrain_arxiv(model_id: str, save: bool = True, output_dir: str | Path = ".") -> Dict[str, Any]:
    """
    model_id should be the BASE (pretrained) model id.
    Steps:
      1) Load local arxiv_fulltext/arxiv jsons if present
      2) Augment with HF card/README links and/or Tavily web search (optional)
      3) Recall 3-1/4-1 evidence only → summarize
    """
    base = model_id.replace("/", "_").lower()

    # 1) Prefer fulltext file, then single-file fallback
    cand = [
        Path(output_dir) / f"arxiv_fulltext_{base}.json",
        Path(output_dir) / f"arxiv_{base}.json",
    ]
    sections: Dict[str,str] = {}
    src = None
    for p in cand:
        try:
            src = _load_json_guess(p)
            sections = _extract_sections(src)
            break
        except FileNotFoundError:
            continue
    # if no local files, still can proceed with augmentation-only
    if not sections:
        sections = {}

    # 2) Augment from HF card/README & Web search (with version-aware GPT verify)
    sections = _augment_sections_with_hf_and_web(model_id, sections)

    if not sections:
        raise FileNotFoundError("No local arXiv texts and no augmented sources found.")

    # 3) Recall & summarize
    evs = _recall(sections, model_id)
    summ = _summarize(evs, model_id)

    all_evs = (evs.get("pretrain_method_evidence", []) +
               evs.get("pretrain_data_evidence", []))
    result = {
        "pretrain_method": summ.get("pretrain_method","No information"),
        "pretrain_data":   summ.get("pretrain_data","No information"),
        "__evidence":      all_evs,
    }

    out = Path(output_dir) / f"pretrain_arxiv_{base}.json"
    if save:
        json.dump(result, open(out,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"✅ Saved: {out}")
    return result

# CLI
if __name__ == "__main__":
    mid = os.environ.get("PRETRAIN_BASE_ID", "bigscience/bloom")
    print("▶ Base model to run:", mid)
    filter_pretrain_arxiv(mid)
