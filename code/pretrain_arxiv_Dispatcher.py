# pretrain_arxiv_Dispatcher.py
"""
Extract pre-training method/data from full arXiv paper text
Input: arxiv_fulltext_{base}.json or arxiv_{base}.json
Output: pretrain_arxiv_{base}.json
"""
import os, json, re
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI

CHUNK_CHARS   = 60_000
CHUNK_OVERLAP = 2_000
MODEL_NAME    = os.getenv("OPENAI_MODEL_ARXIV_DISPATCHER", "o3-mini")

load_dotenv(); _cli = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYS = (
  "From the full paper, find the model's pre-training method and pre-training data, and "
  'return **only a JSON object** in English. '
  'Format: {"pretrain_method": str, "pretrain_data": str, "__evidence":[str,...]}'
)

def _chunk(text: str) -> List[str]:
    out=[]; n=len(text); i=0
    while i<n:
        end=min(i+CHUNK_CHARS,n); out.append(text[i:end])
        if end==n: break
        i=end-CHUNK_OVERLAP
    return out

def _find_full(base: str, root: Path) -> Optional[Path]:
    for name in (f"arxiv_fulltext_{base}.json", f"arxiv_{base}.json"):
        p = root / name
        if p.exists(): return p
    return None

def _load_fulltext(p: Path) -> str:
    j = json.load(open(p, encoding="utf-8"))
    # Our storage format: full_texts(list) or full_text(str)
    if isinstance(j.get("full_texts"), list):
        texts = [t.get("full_text","") if isinstance(t, dict) else str(t) for t in j["full_texts"]]
        return "\n\n".join(texts)
    return j.get("full_text","")

def filter_pretrain_arxiv(model_id: str, save: bool=True, output_dir: str | Path = ".") -> Dict:
    base=model_id.replace("/","_").lower()
    root=Path(output_dir)
    p_in=_find_full(base, root) or _find_full(base, Path("."))
    if not p_in:
        print("⚠️ No arXiv JSON found"); return {}
    txt=_load_fulltext(p_in)

    # Prefer the pre-training section first
    m = re.search(r"(?is)(pre[- ]?training|pretraining).*", txt)
    target = m.group(0) if m else txt

    results=[]
    for i, ch in enumerate(_chunk(target), 1):
        try:
            rsp=_cli.chat.completions.create(
                model=MODEL_NAME,
                reasoning_effort="medium",
                response_format={"type":"json_object"},
                messages=[
                    {"role":"system","content":SYS},
                    {"role":"user","content":f"Read the text below and reply with JSON only.\n\n{ch}"}
                ]
            )
            results.append(json.loads(rsp.choices[0].message.content))
        except Exception as e:
            print(f"⚠️ arXiv-pretrain chunk {i} failed:", e)

    out = {"pretrain_method":"No information","pretrain_data":"No information","__evidence":[]}
    for r in results:
        if r.get("pretrain_method"): out["pretrain_method"]=r["pretrain_method"]
        if r.get("pretrain_data"):   out["pretrain_data"]=r["pretrain_data"]
        if isinstance(r.get("__evidence"), list): out["__evidence"].extend(r["__evidence"])

    p_out = root / f"pretrain_arxiv_{base}.json"
    if save:
        json.dump(out, open(p_out,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"✅ Saved: {p_out}")
    return out
