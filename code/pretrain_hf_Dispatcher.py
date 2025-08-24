# pretrain_hf_Dispatcher.py
"""
HF dispatcher dedicated to pretraining items
Targets ONLY the specified (base) model; drops quotes about other/earlier models.

Input : huggingface_{base}.json  (includes readme/cardData.content)
Output: pretrain_hf_{base}.json
        {
          "pretrain_method": str,
          "pretrain_data":   str,
          "__evidence":      [ { "source": "readme|card_content|model_id", "quote": str }, ... ]
        }
"""

import os, json, re
from typing import Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ─────────────── Config ───────────────
CHUNK_CHARS   = 60_000
CHUNK_OVERLAP = 2_000

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
# o3-mini (reasoning) 기본값. temperature 등 샘플링 파라미터는 절대 사용하지 않음.
MODEL_NAME = os.getenv("OPENAI_MODEL_HF_DISPATCHER", "o3-mini")

_client = OpenAI(api_key=API_KEY)

# ─────────────── Strict model guard ───────────────
def _family_tokens_from_model_id(model_id: str) -> List[str]:
    """
    Extract stable tokens from the model id for family matching.
    Examples:
      "google/gemma-3-27b-it"      -> ["gemma","gemma3","3","27b"]
      "meta-llama/Llama-3.1-8B"    -> ["llama","llama3","31","3.1","8b"]
    """
    name = (model_id or "").split("/", 1)[-1].lower()
    raw  = re.split(r"[^a-z0-9.]+", name)
    toks = set()
    for t in raw:
        t = t.strip()
        if not t or t in {"base","it","instruct","chat","model"}:
            continue
        toks.add(t)
        m = re.match(r"([a-z]+)(\d+(?:\.\d+)*)$", t)  # llama3 / llama3.1
        if m:
            head, ver = m.group(1), m.group(2)
            toks.add(head)
            toks.add(head + ver.replace(".", ""))   # llama31
            toks.add(ver)
            toks.add(ver.replace(".", ""))          # 31
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

def _strict_guard_text(model_id: str) -> str:
    toks = _family_tokens_from_model_id(model_id)
    return (
        "STRICT MODEL FILTER\n"
        f"- Target model: {model_id}\n"
        f"- Accept a quote ONLY if the sentence explicitly mentions one of: {toks}.\n"
        "- Reject sentences about other models or earlier/other versions unless the TARGET is named in the same sentence.\n"
        "- If a document mixes multiple models, keep only sentences that also contain the TARGET tokens.\n"
        "- If in doubt, DROP the quote."
    )

def _STRICT_SUMMARY_GUARD(model_id: str) -> str:
    return _strict_guard_text(model_id) + "\nUse ONLY the provided quotes."

# ─────────────── Payload helpers ───────────────
def _chunk(t: str) -> List[str]:
    out=[]; n=len(t or ""); i=0
    while i<n:
        end=min(i+CHUNK_CHARS,n)
        out.append(t[i:end])
        if end==n: break
        i=end-CHUNK_OVERLAP if end-CHUNK_OVERLAP>i else end
    return out

def _extract_context(j: Dict[str, Any]) -> Dict[str,str]:
    """
    Return labeled sections so the model can tag sources.
    """
    model_id = str(j.get("model_id") or j.get("id") or "")
    readme   = j.get("readme") or ""
    card     = j.get("cardData") or {}
    card_md  = (card.get("content") if isinstance(card, dict) else "") or ""
    return {
        "model_id":    model_id,
        "readme":      str(readme),
        "card_content":str(card_md),
    }

def _payload_text(sections: Dict[str,str]) -> str:
    parts = []
    for tag in ("model_id","readme","card_content"):
        val = sections.get(tag,"")
        if not isinstance(val, str):
            val = json.dumps(val, ensure_ascii=False)
        parts.append(f"[{tag}]\n{val}\n")
    return "\n".join(parts)

# ─────────────── Prompts ───────────────
_RECALL_SYS = """
You are an assistant for AI model evaluation.
Using only the provided Hugging Face text (original quotes), extract evidence about the TARGET model's:
- pre-training METHOD (how pre-training was done)
- pre-training DATA (with what data)

Return JSON ONLY with EXACTLY these keys:
{
  "pretrain_method_evidence": [ { "source": "model_id|readme|card_content", "quote": "..." } ],
  "pretrain_data_evidence":   [ { "source": "model_id|readme|card_content", "quote": "..." } ]
}

Rules:
- 'quote' must be a verbatim sentence copied from the payload.
- 'source' must be one of [model_id], [readme], [card_content].
- If no evidence, use [].
""".strip()

_SUMMARY_SYS = """
Write concise but detailed English summaries using ONLY the provided quotes.
Return JSON ONLY:
{ "pretrain_method": "...", "pretrain_data": "..." }
If no information, write "No information".
""".strip()

# ─────────────── Core (recall → filter → summarize) ───────────────
def _recall(sections: Dict[str,str], model_id: str) -> Dict[str, List[Dict[str,str]]]:
    text = _payload_text(sections)
    agg_method: List[Dict[str,str]] = []
    agg_data:   List[Dict[str,str]] = []

    for ch in _chunk(text):
        msg = _strict_guard_text(model_id) + "\n\n" + _RECALL_SYS + "\n\n=== PAYLOAD ===\n" + ch
        try:
            rsp = _client.chat.completions.create(
                model=MODEL_NAME,
                reasoning_effort="medium",                # o3-mini OK
                response_format={"type":"json_object"},   # JSON 강제
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
                    if src not in {"model_id","readme","card_content"}: continue
                    # quote-level strict filter
                    if not _quote_mentions_target(qt, model_id): continue
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

# ─────────────── Public entry ───────────────
def filter_pretrain_hf(model_id: str, save: bool = True, output_dir: str | Path = ".") -> Dict[str, Any]:
    """
    model_id should be the BASE (pretrained) model id discovered by the agent.
    Only quotes that explicitly mention this target (or its stable family tokens) are used.
    """
    base = model_id.replace("/", "_").lower()
    path_in = Path(output_dir) / f"huggingface_{base}.json"
    if not path_in.exists():
        alt = Path(f"huggingface_{base}.json")
        if not alt.exists():
            raise FileNotFoundError(str(path_in))
        path_in = alt

    hf_json  = json.load(open(path_in, encoding="utf-8"))
    sections = _extract_context(hf_json)

    # Step 1) evidence
    evs = _recall(sections, model_id)

    # Step 2) summary from quotes only
    summ = _summarize(evs, model_id)

    # Merge: summary + combined evidence
    all_evs = (evs.get("pretrain_method_evidence", []) +
               evs.get("pretrain_data_evidence", []))

    result = {
        "pretrain_method": summ.get("pretrain_method","No information"),
        "pretrain_data":   summ.get("pretrain_data","No information"),
        "__evidence":      all_evs,
    }

    out = Path(output_dir) / f"pretrain_hf_{base}.json"
    if save:
        json.dump(result, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"✅ Saved {out}")
    return result

# CLI
if __name__ == "__main__":
    import sys
    mid = "bigscience/bloomz-560m"
    if len(sys.argv) > 1 and sys.argv[1]:
        mid = sys.argv[1]
    print("▶ Base model to run:", mid)
    filter_pretrain_hf(mid)
