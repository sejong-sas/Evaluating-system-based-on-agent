# pretrain_arxiv_Dispatcher.py
"""
Extract pre-training method/data **for the target (base) model only** from arXiv full paper text.
- Input:  arxiv_fulltext_{base}.json or arxiv_{base}.json
- Output: pretrain_arxiv_{base}.json
- Strict model guard: collect/use **only** sentences that explicitly mention the target model tokens.
"""

import os, json, re, hashlib
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI

# ─────────────────────────── Config ───────────────────────────
CHUNK_CHARS   = 60_000
CHUNK_OVERLAP = 2_000

load_dotenv()
MODEL_NAME = os.getenv("OPENAI_MODEL_ARXIV_DISPATCHER", "o3-mini")
_cli = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─────────────────────────── Helpers ───────────────────────────
_PARA_SPLIT = re.compile(r"\n\s*\n+")

def _split_paragraphs(text: str) -> List[str]:
    if not text:
        return []
    if _PARA_SPLIT.search(text):
        return [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]
    return [text.strip()]

def _chunk(text: str) -> List[str]:
    out=[]; n=len(text); i=0
    while i<n:
        end=min(i+CHUNK_CHARS,n); out.append(text[i:end])
        if end==n: break
        i=end-CHUNK_OVERLAP if end-CHUNK_OVERLAP>i else end
    return out

def _find_full(base: str, root: Path) -> Optional[Path]:
    for name in (f"arxiv_fulltext_{base}.json", f"arxiv_{base}.json"):
        p = root / name
        if p.exists(): return p
    # fallback to project root
    for name in (f"arxiv_fulltext_{base}.json", f"arxiv_{base}.json"):
        p = Path(name)
        if p.exists(): return p
    return None

def _normalize_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()

def _dedup_list_str(items: List[str]) -> List[str]:
    seen=set(); out=[]
    for s in items:
        k = hashlib.sha1(_normalize_for_hash(s).encode("utf-8")).hexdigest()
        if k in seen: continue
        seen.add(k); out.append(s)
    return out

# ─────────────────────────── Model-token guard ───────────────────────────
def _canonical_model_tokens(model_id: str) -> List[str]:
    """
    Extract relatively stable tokens from model id.
    - Keep tokens len>=3
    - Add collapsed form (remove non-alnum), and no-digit form
    - Drop generic suffixes (base/it/instruct/chat/model)
    """
    name = (model_id or "").split("/", 1)[-1].lower()
    raw = re.split(r"[^a-z0-9.]+", name)
    alts = set()
    for t in raw:
        t = t.strip()
        if len(t) >= 3 and t not in {"base","it","instruct","chat","model"}:
            alts.add(t)
    collapsed = re.sub(r"[^a-z0-9]", "", name)
    nodigit   = re.sub(r"\d+", "", collapsed)
    if len(collapsed) >= 3: alts.add(collapsed)
    if len(nodigit)   >= 3: alts.add(nodigit)
    # e.g., llama3 / llama3.1 → {llama, llama3, 31/optional}
    for t in list(alts):
        m = re.match(r"([a-z]+)(\d+(?:\.\d+)*)$", t)
        if m:
            alts.add(m.group(1))
            alts.add(m.group(1)+m.group(2).replace(".",""))
    return sorted(alts)

def _contains_any_token(text: str, toks: List[str]) -> bool:
    tl = (text or "").lower().replace("–","-").replace("—","-")
    return any(t for t in toks if t and t in tl)

_T_PRETRAIN = (
    "pretraining", "pre-training", "pre train", "pretrained", "pre-trained",
    "corpus", "dataset", "data mixture", "mixture of data", "tokens", "webtext",
    "crawl", "filtering", "dedup", "training data", "pretraining data", "pre-training data",
    "method", "objective", "loss", "optimizer", "schedule", "steps", "compute"
)

def _paragraphs_for_target(text: str, model_tokens: List[str]) -> List[str]:
    """
    Prefer paragraphs mentioning BOTH model tokens and pretraining-ish keywords.
    If none, keep paragraphs that mention model tokens.
    If still none, fallback to full text (last resort).
    """
    paras = _split_paragraphs(text)
    if not paras: return []

    def has_kw(p: str) -> bool:
        pl = p.lower()
        return any(k in pl for k in _T_PRETRAIN)

    cand1 = [p for p in paras if _contains_any_token(p, model_tokens) and has_kw(p)]
    if cand1:
        return cand1
    cand2 = [p for p in paras if _contains_any_token(p, model_tokens)]
    if cand2:
        return cand2
    # final fallback: keep everything
    return paras

# ─────────────────────────── I/O ───────────────────────────
def _load_fulltext(p: Path) -> str:
    j = json.load(open(p, encoding="utf-8"))
    # Our storage format: full_texts(list) or full_text(str)
    if isinstance(j.get("full_texts"), list):
        texts = [t.get("full_text","") if isinstance(t, dict) else str(t) for t in j["full_texts"]]
        return "\n\n".join(texts)
    return j.get("full_text","")

# ─────────────────────────── Prompts ───────────────────────────
def _sys_prompt(model_id: str, model_tokens: List[str]) -> str:
    return (
        "You analyze arXiv full-text to extract the **pre-training method** and **pre-training data**\n"
        "for the **TARGET model only**. STRICT RULES:\n"
        f"- TARGET model: {model_id}\n"
        f"- Accept/use a sentence ONLY if it explicitly mentions one of: {model_tokens}\n"
        "- If the paper mixes multiple models or earlier versions, ignore those unless the TARGET is named in the same sentence.\n"
        "- If there is no evidence for the TARGET, return \"No information\" and an empty evidence list.\n"
        "- Use only the provided payload; do not invent content.\n\n"
        'Return **JSON only**:\n'
        '{\n'
        '  "pretrain_method": "string (or \\"No information\\")",\n'
        '  "pretrain_data": "string (or \\"No information\\")",\n'
        '  "__evidence": ["verbatim sentence mentioning TARGET", ...]\n'
        "}\n"
    )

def _user_prompt(chunk_text: str) -> str:
    return (
        "Read the following paper text and reply with JSON only. "
        "Use only sentences that explicitly name the TARGET tokens.\n\n"
        f"{chunk_text}"
    )

# ─────────────────────────── Main ───────────────────────────
def filter_pretrain_arxiv(model_id: str, save: bool=True, output_dir: str | Path = ".") -> Dict:
    """
    Analyze arXiv texts **only for the specified (base) model**.
    """
    base=model_id.replace("/","_").lower()
    root=Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    p_in=_find_full(base, root)
    if not p_in:
        print("⚠️ No arXiv JSON found"); return {}

    raw_txt=_load_fulltext(p_in)
    toks = _canonical_model_tokens(model_id)

    # Narrow to likely relevant paragraphs first
    target_paras = _paragraphs_for_target(raw_txt, toks)
    target_text  = "\n\n".join(target_paras)

    # As a hint, also try starting near 'pre-training' section if present
    m = re.search(r"(?is)(pre[- ]?training|pretraining).*", target_text)
    target = m.group(0) if m else target_text

    # Call LLM over chunks  (⚠️ o3-mini: no temperature/top_p/etc.)
    sys_msg = _sys_prompt(model_id, toks)
    results=[]
    for i, ch in enumerate(_chunk(target), 1):
        try:
            rsp=_cli.chat.completions.create(
                model=MODEL_NAME,
                reasoning_effort="medium",
                response_format={"type":"json_object"},
                messages=[
                    {"role":"system","content":sys_msg},
                    {"role":"user","content":_user_prompt(ch)}
                ]
            )
            data = json.loads(rsp.choices[0].message.content)
            results.append(data)
        except Exception as e:
            print(f"⚠️ arXiv-pretrain chunk {i} failed:", e)

    # Merge results
    out = {"pretrain_method":"No information","pretrain_data":"No information","__evidence":[]}
    for r in results:
        if isinstance(r, dict):
            pm = (r.get("pretrain_method") or "").strip()
            pd = (r.get("pretrain_data") or "").strip()
            ev = r.get("__evidence") or []

            # prefer the *longest* non-default text seen so far
            if pm and pm.lower() != "no information" and (len(pm) > len(out["pretrain_method"]) or out["pretrain_method"]=="No information"):
                out["pretrain_method"] = pm
            if pd and pd.lower() != "no information" and (len(pd) > len(out["pretrain_data"]) or out["pretrain_data"]=="No information"):
                out["pretrain_data"] = pd

            if isinstance(ev, list):
                out["__evidence"].extend([str(x) for x in ev if isinstance(x, (str,))])

    # Post-filter evidence to ensure model tokens are present, then dedup
    out["__evidence"] = _dedup_list_str([q for q in out["__evidence"] if _contains_any_token(q, toks)])

    # If no valid evidence remains, downgrade to "No information"
    if not out["__evidence"]:
        out["pretrain_method"] = "No information"
        out["pretrain_data"]   = "No information"

    # Save
    p_out = root / f"pretrain_arxiv_{base}.json"
    if save:
        json.dump(out, open(p_out,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"✅ Saved: {p_out}")
    return out

# CLI
if __name__ == "__main__":
    import sys
    mid = "bigscience/bloom-560m"  # example: must be the *base* model id
    if len(sys.argv)>1 and sys.argv[1]:
        mid=sys.argv[1]
    print("▶ Base model to run:", mid)
    filter_pretrain_arxiv(mid)
