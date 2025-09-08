# huggingface_Dispatcher.py
# High-Recall 2-Pass  +  evidence({source, quote})  →  long summary
# - store evidence as an array of objects
# - summaries must use quotes only
# - remove __evidence_sources field
# - Input: huggingface_{base}.json (Fetcher output)

import os
import json
import re
import hashlib
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

# ──────────────────────────────── Environment ────────────────────────────────
load_dotenv()
_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
_client = OpenAI(api_key=_api_key)

# ──────────────────────────── 16 evaluation item labels ─────────────────────────
LABELS = {
    "1-1": "1-1 (Weights)",                  "1-2": "1-2 (Code)",
    "1-3": "1-3 (License)",                  "1-4": "1-4 (Paper)",
    "1-5": "1-5 (Architecture)",             "1-6": "1-6 (Tokenizer)",
    "2-1": "2-1 (Hardware)",                 "2-2": "2-2 (Software)",
    "2-3": "2-3 (API)",
    "3-1": "3-1 (Pre-training)",             "3-2": "3-2 (Fine-tuning)",
    "3-3": "3-3 (Reinforcement Learning)",
    "4-1": "4-1 (Pre-training Data)",
    "4-2": "4-2 (Fine-tuning Data)",
    "4-3": "4-3 (Reinforcement Learning Data)",
    "4-4": "4-4 (Data Filtering)",
}

EVAL_DESCRIPTIONS = {
    LABELS["1-1"]: "All information about the availability, location, and access method of model weights, including whether anyone can download them",
    LABELS["1-2"]: "All information about whether TRAINING code is public (distinguish training vs inference/serving). Specify which stages (pre-training / fine-tuning / RL) are public.",
    LABELS["1-3"]: "All information about license type and explicit grants/restrictions for use, modification, redistribution, commercial use (include exact quoted phrases).",
    LABELS["1-4"]: "All information about official papers, technical reports, blogs, and links related to the model",
    LABELS["1-5"]: "All information about model architecture (layers, params, hyperparameters) and design details",
    LABELS["1-6"]: "All information about tokenizer used, its name/structure, and whether downloadable",
    LABELS["2-1"]: "All information about training hardware (H100, TPU, etc.), quantity, and compute scale",
    LABELS["2-2"]: "All information about the software stack used to TRAIN (frameworks, libraries, versions, configs/flags)",
    LABELS["2-3"]: "All information about an accessible API (GPT/Gemini-like), docs/examples, public availability",
    LABELS["3-1"]: "All information about pre-training methodology, procedures, data flow, key hyperparameters",
    LABELS["3-2"]: "All information about fine-tuning methods, goals, data usage, reproducible pipeline",
    LABELS["3-3"]: "All information about RLHF/DPO/PPO, methods/procedures/params",
    LABELS["4-1"]: "All information about types/sources/licenses/quantities of pre-training data",
    LABELS["4-2"]: "All information about sources/composition/examples/public availability of fine-tuning datasets",
    LABELS["4-3"]: "All information about composition/accessibility/sources/generation of RL datasets",
    LABELS["4-4"]: "All information about data filtering/cleaning criteria/procedures/impacts",
}

# ─────────────────────────────── Grouping ───────────────────────────────
ITEM_GROUPS: List[List[str]] = [
    ["1-1", "1-2", "1-3", "1-4"],
    ["1-5", "1-6", "2-1", "2-2"],
    ["2-3", "3-1", "3-2", "3-3"],
    ["4-1", "4-2", "4-3", "4-4"],
]

# ──────────────────────────── Hyperparameters ────────────────────────────
CHUNK_CHARS = 60_000
CHUNK_OVERLAP = 2_000
EVIDENCE_LIMIT_PER_KEY = 300
MODEL_NAME = os.getenv("OPENAI_MODEL_HF_DISPATCHER", "o3")

# Optional: post-filter by model tokens (0=off, 1=on). Safer default OFF for HF readmes.
HF_APPLY_MODEL_GUARD = os.getenv("HF_APPLY_MODEL_GUARD", "0") == "1"

# ─────────────────────────────── Utils ───────────────────────────────
_PARA_SPLIT = re.compile(r"\n\s*\n+")  # for potential dedup helpers
_STOPWORDS = {
    "ai","llm","language","nlp","ml","model","models","base","chat","instruct","instruction",
    "sft","rl","rlhf","eval","evaluation","bench","benchmark","dev","test","demo","preview",
    "alpha","beta","rc","hf","release","v","v1","v2","v3","v4","v5","it"
}

def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def _chunk_text(s: str, chunk: int, overlap: int) -> List[str]:
    out, n, i = [], len(s), 0
    while i < n:
        end = min(i + chunk, n)
        out.append(s[i:end])
        if end == n:
            break
        i = end - overlap if end - overlap > i else end
    return out

def _dedup_evidences(evs: List[Dict[str, str]], limit: int) -> List[Dict[str, str]]:
    seen, out = set(), []
    for ev in evs:
        if not (isinstance(ev, dict)
                and isinstance(ev.get("source"), str)
                and isinstance(ev.get("quote"), str)):
            continue
        src = ev["source"].strip()
        qt  = ev["quote"].strip()
        if not src or not qt:
            continue
        key = (src, qt)
        if key in seen:
            continue
        seen.add(key)
        out.append({"source": src, "quote": qt})
        if len(out) >= limit:
            break
    return out

def _group_desc_map(ids: List[str]) -> Dict[str, str]:
    return {LABELS[i]: EVAL_DESCRIPTIONS[LABELS[i]] for i in ids}

def _family_tokens_from_model_id(model_id: str) -> set[str]:
    name = (model_id or "").split("/", 1)[-1].lower()
    raw = re.split(r"[^a-z0-9.]+", name)
    base: set[str] = set()
    for tt in (t.strip() for t in raw):
        if not tt: 
            continue
        if tt in _STOPWORDS:
            continue
        if len(tt) >= 2:
            base.add(tt)
        m = re.match(r"([a-z]+)(\d+(?:\.\d+)*)$", tt)
        if m:
            base.add(m.group(1))
            base.add(m.group(1)+m.group(2).replace(".",""))
            base.add(m.group(2))
            base.add(m.group(2).replace(".",""))
    joined = re.sub(r"[^a-z0-9]", "", name)
    nodigit = re.sub(r"\d+", "", joined)
    if len(joined) >= 3: base.add(joined)
    if len(nodigit) >= 3: base.add(nodigit)
    return base

def _model_guard_text(model_id: str) -> str:
    toks = sorted(_family_tokens_from_model_id(model_id))
    return (
        "STRICT MODEL FILTER\n"
        f"- Target model: {model_id}\n"
        f"- Accept a quote ONLY if the sentence explicitly mentions one of: {toks}.\n"
        "- If a section mixes multiple models or earlier/other versions, keep only sentences that also include TARGET tokens.\n"
        "- If in doubt, DROP the quote.\n"
        "- If the TARGET is named in the immediately previous sentence and the current sentence contains the fact, include BOTH sentences together as one quote."
    )

def _quote_mentions_target(q: str, model_id: str) -> bool:
    if not q: return False
    ql = q.lower().replace("–","-").replace("—","-")
    for t in _family_tokens_from_model_id(model_id):
        if len(t) >= 2 and t in ql:
            return True
    return False

# ─────────────────────────────── Prompts ───────────────────────────────
_BASE_RECALL_SYS = """
You are an expert at extracting AI model openness evaluation information from a Hugging Face repository.
Use only the provided payload (original text).
Return evidence for each item as an array of objects:
  [{ "source": "...", "quote": "..." }, ...]
- "source": a payload tag (e.g., [readme], [files], [py_files/train.py], [config], [generation_config])
- "quote" : a verbatim sentence copied from that section (no edits/summaries)
STRICT TARGET POLICY:
- Prefer quotes where the sentence itself explicitly names a TARGET token.
- If the TARGET name is in the immediately previous sentence and the current sentence carries the fact,
  include BOTH sentences together as one quote (previous + current).
If there is no evidence, return [].
Output a JSON object only.
""".strip()

_BASE_SUMMARY_SYS = """
Using the provided quotes only, write long and detailed summaries for each item.
Output a JSON object only.
""".strip()

_USAGE_SYS = """
You are a classifier. Decide whether the released MODEL actually USED:
- Fine-tuning (SFT/Instruction/Adapters/etc.)
- Reinforcement Learning (RLHF/DPO/PPO/etc.)
STRICT RULES:
- Do NOT infer "used" from generic advice like "you can fine-tune this model".
- "used" only if quotes explicitly state the authors performed that stage.
- "not_used" only if quotes explicitly deny it.
- Otherwise "unknown".
If quotes describe ONLY supervised finetuning and ZERO RL signals, classify RL as "not_used".
Answer JSON only:
{ "fine_tuning": "used|not_used|unknown", "rl": "used|not_used|unknown" }
""".strip()

def _build_recall_inst(group: List[str], model_id: str) -> str:
    desc = _json(_group_desc_map(group))
    skeleton = _json({LABELS[k]: [] for k in group})
    return (
        _model_guard_text(model_id) + "\n"
        "Items in this group:\n" + desc + "\n"
        "Return a JSON object with EXACTLY these keys (arrays of {source,quote}):\n" + skeleton
    )

def _build_summary_inst(group: List[str], model_id: str) -> str:
    desc = _json(_group_desc_map(group))
    targets = sorted(_family_tokens_from_model_id(model_id))
    return (
        _model_guard_text(model_id) + "\n"
        "Items in this group:\n" + desc + "\n"
        "Return a JSON object with EXACTLY these keys (string summaries):\n" +
        _json({LABELS[k]: "" for k in group}) +
        "\nUse ONLY the provided quotes."
    )

# ─────────────────────────────── GPT call ───────────────────────────────
def _chat_json(system: str, user: str) -> Dict[str, Any]:
    resp = _client.chat.completions.create(
        model=MODEL_NAME,
        reasoning_effort="medium",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    try:
        return json.loads(resp.choices[0].message.content.strip())
    except Exception:
        return {}

# ───────────────────────────── Payload builder ─────────────────────────────
def _make_group_payload(hf: Dict, idx: int) -> Dict:
    py_src = hf.get("py_files", {}) or {}
    # cap number/length to keep token usage sane
    py_items = list(py_src.items())[:20]  # up to 20 files
    py_files = {str(fn): (str(src)[:20_000] if isinstance(src, str) else "")
                for fn, src in py_items}
    return {
        "model_id":          hf.get("model_id", ""),
        "files":             (hf.get("files", []) or [])[:2000],
        "readme":            hf.get("readme", "") or "",
        "license_file":      hf.get("license_file", "") or "",
        "config":            hf.get("config", "") or "",
        "generation_config": hf.get("generation_config", "") or "",
        "py_files":          py_files,
    }

def _payload_to_text(p: Dict) -> str:
    parts = [
        f"[model_id]\n{p.get('model_id','')}\n",
        f"[readme]\n{p.get('readme','')}\n",
        f"[license_file]\n{p.get('license_file','')}\n",
        f"[config]\n{p.get('config','')}\n",
        f"[generation_config]\n{p.get('generation_config','')}\n",
    ]
    for fn, code in p.get("py_files", {}).items():
        parts.append(f"[py_files/{fn}]\n{code}\n")
    if p.get("files"):
        parts.append("[files]\n" + "\n".join(map(str, p["files"])) + "\n")
    return "\n".join(parts)

# ───────────────────────────── Evidence collection ─────────────────────────────
_ALLOWED_PREFIX = ("model_id", "readme", "license_file", "config",
                   "generation_config", "files", "py_files/")

def _is_valid_source(src: str) -> bool:
    return isinstance(src, str) and src.strip().lower().strip("[]").startswith(_ALLOWED_PREFIX)

def _filter_evidence_by_model(ev: Dict[str, List[Dict[str,str]]], model_id: str) -> Dict[str, List[Dict[str,str]]]:
    """Optional stricter guard (disabled by default). Keeps only valid sources; if enabled, also require TARGET tokens in quote."""
    out: Dict[str, List[Dict[str,str]]] = {}
    for lbl, arr in ev.items():
        kept = []
        for e in arr or []:
            src = (e.get("source") or "").strip()
            qt  = (e.get("quote")  or "").strip()
            if not src or not qt: continue
            if not _is_valid_source(src): continue
            if HF_APPLY_MODEL_GUARD:
                if not _quote_mentions_target(qt, model_id):
                    continue
            kept.append({"source": src, "quote": qt})
        out[lbl] = _dedup_evidences(kept, EVIDENCE_LIMIT_PER_KEY)
    return out

def _recall_collect(group: List[str], text: str, model_id: str) -> Dict[str, List[Dict[str, str]]]:
    chunks = _chunk_text(text, CHUNK_CHARS, CHUNK_OVERLAP)
    out: Dict[str, List[Dict[str, str]]] = {LABELS[k]: [] for k in group}

    for chunk in chunks:
        ans = _chat_json(_BASE_RECALL_SYS, _build_recall_inst(group, model_id) +
                         "\n=== PAYLOAD ===\n" + chunk)

        for k in group:
            lbl = LABELS[k]
            evs = ans.get(lbl, [])
            if not isinstance(evs, list):
                continue
            out[lbl].extend(evs)

    # post-validation + (optional) model-guard + dedup
    return _filter_evidence_by_model(out, model_id)

# ─────────────────────────────── Summary generation ──────────────────────────────
def _summarize(group: List[str], evid: Dict[str, List[Dict[str, str]]], model_id: str) -> Dict[str, str]:
    quotes = {LABELS[k]: [e["quote"] for e in evid.get(LABELS[k], [])] for k in group}
    ans = _chat_json(_BASE_SUMMARY_SYS, _build_summary_inst(group, model_id) +
                     "\n=== QUOTES ===\n" + _json(quotes))
    return {LABELS[k]: (ans.get(LABELS[k], "") or "") for k in group}

# ─────────────────────────────── Merge utils ───────────────────────────────
def _merge_for_final(summary: Dict[str, str],
                     evid: Dict[str, List[Dict[str, str]]]) -> Dict[str, Any]:
    final = {}
    for lbl, txt in summary.items():
        final[lbl] = txt.strip()
        final[f"{lbl}__evidence"] = _dedup_evidences(evid.get(lbl, []) or [], EVIDENCE_LIMIT_PER_KEY)
    return final

def _merge_dicts(ds: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged = {}
    for d in ds:
        merged.update(d)
    return merged

# ───────────────────────────── RL not-used rule (general) ─────────────────────────────
_T_TOKENS: Tuple[str, ...] = ("finetune","fine-tuning","instruction-tune","sft","xp3","xp3mt","mtf","lora","qlora","supervised fine-tuning")
_RL_TOKENS: Tuple[str, ...] = ("rlhf","reinforcement learning","dpo","ppo","reward model","preference model","human feedback","rlaif","kl penalty")

def _contains_any(text: str, toks: Tuple[str, ...]) -> bool:
    tl = (text or "").lower().replace("–","-").replace("—","-")
    return any(t in tl for t in toks)

def _all_quotes(merged: dict) -> str:
    buf: List[str] = []
    for k, v in merged.items():
        if isinstance(k, str) and k.endswith("__evidence"):
            for e in (v or []):
                if isinstance(e, dict):
                    q = e.get("quote") or ""
                    if q: buf.append(q)
    return "\n".join(buf)

def _rule_infer_rl_not_used(merged: dict) -> bool:
    q = _all_quotes(merged)
    return (_contains_any(q, _T_TOKENS) and not _contains_any(q, _RL_TOKENS))

def _classify_usage_from_merged(merged: dict) -> dict:
    def _pull(label):
        txt = merged.get(label, "") or ""
        evs = merged.get(f"{label}__evidence", []) or []
        quotes = "\n".join([e.get("quote","") for e in evs if isinstance(e, dict)])
        return (txt + "\n" + quotes).strip()
    ft_txt = _pull("3-2 (Fine-tuning)")
    rl_txt = _pull("3-3 (Reinforcement Learning)")
    text = f"[fine_tuning]\n{ft_txt}\n\n[reinforcement]\n{rl_txt}".strip()
    usage = {"fine_tuning":"unknown","rl":"unknown"}
    if text:
        ans = _client.chat.completions.create(
            model=MODEL_NAME, reasoning_effort="medium",
            response_format={"type":"json_object"},
            messages=[{"role":"system","content":_USAGE_SYS},
                      {"role":"user","content":text[:12000]}]
        )
        try:
            out = json.loads(ans.choices[0].message.content.strip())
        except Exception:
            out = {}
        ft_s = out.get("fine_tuning","unknown"); rl_s = out.get("rl","unknown")
        if ft_s in {"used","not_used","unknown"}: usage["fine_tuning"] = ft_s
        if rl_s in {"used","not_used","unknown"}: usage["rl"] = rl_s

    if usage.get("rl") in (None, "unknown"):
        if _rule_infer_rl_not_used(merged):
            usage["rl"] = "not_used"
    return usage

# ────────────────────────────── Main function ────────────────────────────────
def filter_hf_features(model: str, save: bool = True, output_dir: str | Path = ".") -> Dict[str, Any]:
    base = model.replace("/", "_").lower()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Input JSON: prefer output_dir → fallback to project root
    path_in = output_dir / f"huggingface_{base}.json"
    if path_in.exists():
        hf = json.load(open(path_in, encoding="utf-8"))
    else:
        alt = Path(f"huggingface_{base}.json")
        if not alt.exists():
            raise FileNotFoundError(str(path_in))
        hf = json.load(open(alt, encoding="utf-8"))

    parts = []
    for idx, grp in enumerate(ITEM_GROUPS, 1):
        try:
            payload = _make_group_payload(hf, idx - 1)
            text = _payload_to_text(payload)
            evid = _recall_collect(grp, text, model)
            summ = _summarize(grp, evid, model)
            part = _merge_for_final(summ, evid)
        except Exception as e:
            print(f"⚠️ Error processing group {idx}:", e)
            part = {}

        if save:
            out_path = output_dir / f"huggingface_filtered_{base}_{idx}.json"
            json.dump(part, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            print(f"✅ Saved group {idx} result:", out_path)
        parts.append(part)

    merged = _merge_dicts(parts)

    try:
        merged["__usage"] = _classify_usage_from_merged(merged)
    except Exception as e:
        print("⚠️ Failed to classify usage:", e)
        merged["__usage"] = {"fine_tuning":"unknown","rl":"unknown"}

    if save:
        out_merged = output_dir / f"huggingface_filtered_final_{base}.json"
        json.dump(merged, open(out_merged, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("✅ Saved final merged result:", out_merged)
    return merged

# ─────────────────────────────── CLI entrypoint ──────────────────────────────
if __name__ == "__main__":
    import sys
    model_id = "bigscience/bloomz-560m"
    outdir = "."
    if len(sys.argv) >= 2 and sys.argv[1]:
        model_id = sys.argv[1]
    if len(sys.argv) >= 3 and sys.argv[2]:
        outdir = sys.argv[2]
    print("▶ Model to run:", model_id)
    print("▶ Output folder:", outdir)
    filter_hf_features(model_id, output_dir=outdir)
