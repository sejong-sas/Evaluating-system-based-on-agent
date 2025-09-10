# pretrain_arxiv_Dispatcher.py
"""
arXiv dispatcher dedicated to pretraining items (BASE model).
Relaxed guard:
- Accept a quote if (a) the sentence mentions target tokens OR
  (b) the *section/doc* is clearly about the target model.

Inputs (search in output_dir, then CWD fallback):
  arxiv_fulltext_{base_model}.json   # preferred
  └─ { "full_texts": [ { "arxiv_id": "...", "full_text": "..." }, ... ] }
  (optional) arxiv_{base_model}.json # same schema or single doc

Output:
  pretrain_arxiv_{base_model}.json
  {
    "pretrain_method": "...",
    "pretrain_data":   "...",
    "__evidence":      [ { "source": "arxiv:<id>|sections:<id>", "quote": "..." }, ... ]
  }
"""

import os, json, re
from typing import Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ───────── Config ─────────
CHUNK_CHARS   = 60_000
CHUNK_OVERLAP = 2_000
DOC_HITS_THRESHOLD = int(os.getenv("PRETRAIN_DOC_HITS", "3"))

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
MODEL_NAME = os.getenv("OPENAI_MODEL_PRETRAIN_DISPATCHER", "o3-mini")
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
        f"- Prefer quotes where the sentence explicitly mentions one of: {toks}."
    )

def _STRICT_SUMMARY_GUARD(model_id: str) -> str:
    return _strict_guard_text(model_id) + "\nUse ONLY the provided quotes."

# ───────── Payload helpers ─────────
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

def _payload_text(sections: Dict[str,str]) -> str:
    parts = []
    for tag, val in sections.items():
        if not isinstance(val, str):
            val = json.dumps(val, ensure_ascii=False)
        parts.append(f"[{tag}]\n{val}\n")
    return "\n".join(parts)

# ───────── Prompts ─────────
_RECALL_SYS = """
You are an assistant for AI model evaluation.
Using only the provided arXiv payload, extract evidence about the TARGET model's:
- pre-training METHOD (how pre-training was done)
- pre-training DATA (with what data)

Return JSON ONLY with EXACTLY these keys:
{
  "pretrain_method_evidence": [ { "source": "arxiv:<id>", "quote": "..." } ],
  "pretrain_data_evidence":   [ { "source": "arxiv:<id>", "quote": "..." } ]
}

Rules:
- 'quote' must be a verbatim sentence copied from the payload.
- 'source' must be the section tag (e.g., [arxiv:<id>]).
- If no evidence, use [].
""".strip()

_SUMMARY_SYS = """
Write concise but detailed English summaries using ONLY the provided quotes.
Return JSON ONLY:
{ "pretrain_method": "...", "pretrain_data": "..." }
If no information, write "No information".
""".strip()

# ───────── Core (recall → filter → summarize) ─────────
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
            arr = obj.get(k, [])
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
    """
    base = model_id.replace("/", "_").lower()
    # Prefer fulltext file, then single-file fallback
    cand = [
        Path(output_dir) / f"arxiv_fulltext_{base}.json",
        Path(output_dir) / f"arxiv_{base}.json",
    ]
    src = None
    for p in cand:
        try:
            src = _load_json_guess(p)
            break
        except FileNotFoundError:
            continue
    if src is None:
        raise FileNotFoundError([str(x) for x in cand])

    sections = _extract_sections(src)
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
    mid = os.environ.get("PRETRAIN_BASE_ID", "bigscience/bloomz-560m")
    print("▶ Base model to run:", mid)
    filter_pretrain_arxiv(mid)
