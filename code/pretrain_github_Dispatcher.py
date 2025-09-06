# pretrain_github_Dispatcher.py
"""
Extract pre-training information **only for the target (agent-detected base) model**
from GitHub README / docs, with a strict model guard.

Input : github_{base}.json
Output: pretrain_gh_{base}.json
"""
import os, json, re
from typing import Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ───────────────────────── Env & model ─────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
# 전용 환경변수 (없으면 o3-mini)
MODEL_NAME = os.getenv("OPENAI_MODEL_PRETRAIN_GH_DISPATCHER", "o3-mini")
_cli = OpenAI(api_key=OPENAI_API_KEY)

# ───────────────────────── Knobs ─────────────────────────
CHUNK_CHARS   = 60_000
CHUNK_OVERLAP = 2_000
EVIDENCE_LIMIT = 300

# ───────────────────────── Helpers ─────────────────────────
_PARA_SPLIT = re.compile(r"\n\s*\n+")

def _chunk(text: str) -> List[str]:
    n, i, out = len(text), 0, []
    while i < n:
        end = min(i + CHUNK_CHARS, n)
        out.append(text[i:end])
        if end == n: break
        i = end - CHUNK_OVERLAP if end - CHUNK_OVERLAP > i else end
    return out

def _canonical_model_tokens(model_id: str) -> List[str]:
    """Stable tokens for strict model guarding (family/variants)."""
    name = (model_id or "").split("/", 1)[-1].lower()
    raw = re.split(r"[^a-z0-9.]+", name)
    alts = set()
    for t in raw:
        t = t.strip()
        if not t: 
            continue
        if t in {"base","it","instruct","chat","model"}:  # generic
            continue
        if len(t) >= 3:
            alts.add(t)
        m = re.match(r"([a-z]+)(\d+(?:\.\d+)*)$", t)
        if m:
            alts.add(m.group(1))
            alts.add(m.group(1)+m.group(2).replace(".",""))
    joined  = re.sub(r"[^a-z0-9]", "", name)
    nodigit = re.sub(r"\d+", "", joined)
    if len(joined)  >= 3: alts.add(joined)
    if len(nodigit) >= 3: alts.add(nodigit)
    return sorted(alts)

def _filter_text_for_model(text: str, toks: List[str]) -> str:
    """Keep only paragraphs that mention any target token."""
    if not text or not toks: return ""
    keep = []
    blocs = _PARA_SPLIT.split(text) if _PARA_SPLIT.search(text or "") else [text]
    for p in blocs:
        pl = p.lower()
        if any(t in pl for t in toks):
            keep.append(p.strip())
    return "\n\n".join(keep)

def _sys_prompt(model_id: str, toks: List[str]) -> str:
    return (
        "From the given GitHub README/docs, summarize the model's pre-training method and pre-training data, "
        "and return it **in JSON format**.\n\n"
        "Rules:\n"
        f"- STRICT MODEL FILTER — Target model: {model_id}\n"
        f"- Accept and quote ONLY sentences that explicitly mention one of these tokens: {toks}.\n"
        "- If a sentence talks about a different model, an earlier/later version, or another family member "
        "without naming the TARGET in the same sentence, ignore it.\n"
        "- Use only verbatim quotes for evidence (no paraphrasing).\n"
        '- If no valid evidence remains after filtering, return: '
        '{"pretrain_method":"No information","pretrain_data":"No information","__evidence":[]}.\n\n'
        'Return JSON only with this schema: '
        '{"pretrain_method": str, "pretrain_data": str, "__evidence":[str,...]}'
    )

def _is_reasoning_model(name: str) -> bool:
    n = (name or "").lower()
    return n.startswith(("o1", "o3"))

def _chat_json_jsononly(model_name: str, system: str, user: str) -> Dict[str, Any]:
    """
    o1/o3 계열: reasoning_effort만 사용 (temperature/top_p 등 샘플링 파라미터 금지)
    그 외: 샘플링 파라미터도 넣지 않고 호출(보편 안전)
    """
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "response_format": {"type": "json_object"},
    }
    if _is_reasoning_model(model_name):
        payload["reasoning_effort"] = "medium"
    r = _cli.chat.completions.create(**payload)
    try:
        return json.loads(r.choices[0].message.content.strip())
    except Exception:
        return {}

def _dedup_evidence_strs(evs: List[str], limit: int = EVIDENCE_LIMIT) -> List[str]:
    seen, out = set(), []
    for s in evs or []:
        if not isinstance(s, str): 
            continue
        q = s.strip()
        if not q or q in seen:
            continue
        seen.add(q)
        out.append(q)
        if len(out) >= limit:
            break
    return out

# ───────────────────────── Main ─────────────────────────
def filter_pretrain_gh(model_id: str,
                       save: bool = True,
                       output_dir: str | Path = ".") -> Dict[str, Any]:
    base = model_id.replace("/", "_").lower()
    root = Path(output_dir)
    pin  = root / f"github_{base}.json"
    if not pin.exists():
        # fallback to project root
        alt = Path(f"github_{base}.json")
        if alt.exists():
            pin = alt
    if not pin.exists():
        print("⚠️ No GH JSON:", pin)
        out = {"pretrain_method":"No information","pretrain_data":"No information","__evidence":[]}
        if save:
            pout = root / f"pretrain_gh_{base}.json"
            json.dump(out, open(pout, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            print("✅ Saved:", pout)
        return out

    j = json.load(open(pin, encoding="utf-8"))
    readme = (j.get("readme") or "")[:240_000]

    toks = _canonical_model_tokens(model_id)
    filtered = _filter_text_for_model(readme, toks)
    work_text = filtered if filtered.strip() else "NO_TARGET_MENTIONS"

    sys = _sys_prompt(model_id, toks)
    results: List[Dict[str, Any]] = []
    for idx, ch in enumerate(_chunk(work_text), 1):
        try:
            ans = _chat_json_jsononly(MODEL_NAME, sys, ch)
            if ans:
                results.append(ans)
        except Exception as e:
            print(f"⚠️ GH-pretrain chunk {idx} failed:", e)

    out: Dict[str, Any] = {"pretrain_method":"No information","pretrain_data":"No information","__evidence":[]}
    for r in results:
        if not isinstance(r, dict):
            continue
        pm = r.get("pretrain_method")
        pd = r.get("pretrain_data")
        ev = r.get("__evidence", [])
        if isinstance(pm, str) and pm.strip():
            out["pretrain_method"] = pm.strip()
        if isinstance(pd, str) and pd.strip():
            out["pretrain_data"] = pd.strip()
        if isinstance(ev, list):
            out["__evidence"].extend([x for x in ev if isinstance(x, str)])

    out["__evidence"] = _dedup_evidence_strs(out["__evidence"], EVIDENCE_LIMIT)

    p_out = root / f"pretrain_gh_{base}.json"
    if save:
        json.dump(out, open(p_out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print("✅ Saved:", p_out)
    return out

# CLI
if __name__ == "__main__":
    mid = "bigscience/bloom-560m"
    print("▶ Base model for pretrain (GitHub):", mid)
    filter_pretrain_gh(mid)
