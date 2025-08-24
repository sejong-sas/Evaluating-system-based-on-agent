# github_Dispatcher.py
# High-Recall 2-Pass  (evidence {source, quote} → long summary)
# - store evidence as an array of objects [{source, quote}, …]
# - summaries must use quotes only
# - STRICT: collect ONLY quotes that explicitly mention the TARGET model
# - remove unnecessary fields like __evidence_sources, __sources

import os, json, re
from typing import Dict, List, Any
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

# ────────────────── Environment ──────────────────
load_dotenv()
_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
_client = OpenAI(api_key=_api_key)

# ─────────────── 16 evaluation items ───────────────
LABELS = {
    "1-1": "1-1 (Weights)",                     "1-2": "1-2 (Code)",
    "1-3": "1-3 (License)",                     "1-4": "1-4 (Paper)",
    "1-5": "1-5 (Architecture)",                "1-6": "1-6 (Tokenizer)",
    "2-1": "2-1 (Hardware)",                    "2-2": "2-2 (Software)",
    "2-3": "2-3 (API)",
    "3-1": "3-1 (Pre-training)",                "3-2": "3-2 (Fine-tuning)",
    "3-3": "3-3 (Reinforcement Learning)",
    "4-1": "4-1 (Pre-training Data)",
    "4-2": "4-2 (Fine-tuning Data)",
    "4-3": "4-3 (Reinforcement Learning Data)",
    "4-4": "4-4 (Data Filtering)",
}

# ─────────────── Item descriptions (brief) ───────────────
EVAL_DESCRIPTIONS = {
    LABELS["1-1"]: "All information about whether model weights are public, their location, access method, and if anyone can download them",
    LABELS["1-2"]: "All information about whether TRAINING code is public. Distinguish training pipeline (data prep, configs, scripts, schedules) from inference/serving-only code. Specify which parts of training are public (pre-training, fine-tuning, RL).",
    LABELS["1-3"]: "All information about the license and explicit grants/restrictions for each right: (a) use, (b) modification, (c) redistribution, (d) commercial use. Extract exact quoted lines from LICENSE/README; include license name/version and phrases like 'non-commercial', 'research only', 'no derivatives', 'no redistribution', 'evaluation only'.",
    LABELS["1-4"]: "All information about official papers, technical reports, blogs and links related to the model",
    LABELS["1-5"]: "All information about model architecture (e.g., number of layers, hyperparameters) and structural design details",
    LABELS["1-6"]: "All information about which tokenizer is used, its name/structure, and whether it is downloadable",
    LABELS["2-1"]: "All information about training hardware type (H100, TPU, etc.), quantity, and compute scale",
    LABELS["2-2"]: "All information about the software stack used to TRAIN the model (not inference/serving): core ML frameworks (e.g., PyTorch/JAX/TensorFlow), distributed-training libraries (DeepSpeed, Megatron-LM, FSDP/ZeRO), acceleration kernels (FlashAttention/xFormers/Apex), data-loading pipelines, optimizers/schedulers, exact versions, configs, and runtime flags",
    LABELS["2-3"]: "All information about the existence of an accessible API (must be an API like GPT/Gemini, not a library), docs, examples, and public availability",
    LABELS["3-1"]: "All information about pre-training methodology, procedures, data flow, and hyperparameter settings",
    LABELS["3-2"]: "All information about fine-tuning methods, goals, whether data is used, and the existence of a reproducible pipeline",
    LABELS["3-3"]: "All information about RLHF, DPO, etc., including concrete methods, procedures, and parameter settings",
    LABELS["4-1"]: "All information about types, quantities, sources, permitted use, and composition of pre-training data",
    LABELS["4-2"]: "All information about sources, composition, examples, and public availability of fine-tuning datasets",
    LABELS["4-3"]: "All information about composition, accessibility, sources, and generation of reinforcement learning datasets",
    LABELS["4-4"]: "All information about data filtering/cleaning methods, criteria used, processes, and their impact",
}

# ───────────── Groups ─────────────
ITEM_GROUPS = [
    ["1-1", "1-2", "1-3", "1-4"],
    ["1-5", "1-6", "2-1", "2-2"],
    ["2-3", "3-1", "3-2", "3-3"],
    ["4-1", "4-2", "4-3", "4-4"],
]

# ───────────── Parameters ─────────────
CHUNK_CHARS = 60_000
CHUNK_OVERLAP = 2_000
EVIDENCE_LIMIT_PER_KEY = 300
MODEL_NAME = os.getenv("OPENAI_MODEL_GH_DISPATCHER", "o3-mini")

# ───────────── Utils ─────────────
def _js(o: Any) -> str:
    return json.dumps(o, ensure_ascii=False, indent=2)

def _chunk(s: str, size: int, ov: int) -> List[str]:
    out, n, i = [], len(s), 0
    while i < n:
        end = min(i + size, n)
        out.append(s[i:end])
        if end == n: break
        i = end - ov if end - ov > i else end
    return out

def _dedup_evid(evs: List[Dict[str, str]], limit: int) -> List[Dict[str, str]]:
    seen, out = set(), []
    for ev in evs:
        if not (isinstance(ev, dict)
                and isinstance(ev.get("source"), str)
                and isinstance(ev.get("quote"), str)):
            continue
        src, qt = (ev.get("source") or "").strip(), (ev.get("quote") or "").strip()
        if not src or not qt: continue
        key = (src, qt)
        if key in seen: continue
        seen.add(key); out.append({"source": src, "quote": qt})
        if len(out) >= limit: break
    return out

def _desc(ids: List[str]) -> Dict[str, str]:
    return {LABELS[i]: EVAL_DESCRIPTIONS[LABELS[i]] for i in ids}

# === Target-model guard (공통) =========================================
def _canonical_model_tokens(model_id: str) -> list[str]:
    """모델 ID에서 비교적 안정적인 토큰/별칭 후보 추출 (짧고 흔한 토큰 제외)."""
    name = (model_id or "").split("/", 1)[-1].lower()
    raw = re.split(r"[^a-z0-9.]+", name)
    alts = set()
    for t in raw:
        t = t.strip()
        if len(t) >= 3 and t not in {"base","it","instruct","chat","model"}:
            alts.add(t)
    joined = re.sub(r"[^a-z0-9]", "", name)
    nodigit = re.sub(r"\d+", "", joined)
    if len(joined) >= 3: alts.add(joined)
    if len(nodigit) >= 3: alts.add(nodigit)
    return sorted(alts)

def _model_guard_text(model_id: str) -> str:
    toks = _canonical_model_tokens(model_id)
    return (
        "STRICT MODEL FILTER\n"
        f"- Target model: {model_id}\n"
        f"- Accept a quote ONLY if the sentence explicitly mentions one of: {toks}.\n"
        "- Reject sentences about other models or earlier/other versions unless the TARGET is named in the same sentence.\n"
        "- If a document mixes multiple models, keep only sentences that also contain the TARGET tokens.\n"
        "- If in doubt, DROP the quote.\n"
    )

def _quote_mentions_target(q: str, model_id: str) -> bool:
    if not q: return False
    ql = q.lower()
    for t in _canonical_model_tokens(model_id):
        if t and t in ql:
            return True
    return False

# ───────────── Prompts ─────────────
_BASE_RECALL_SYS = """
You are an expert at extracting AI model openness evaluation information from a GitHub repository.
Using only the payload (original text), return evidence for each item in the format:
  [{ "source": "...", "quote": "..." }, …]
· source  : one of [repo], [readme], [license_files], [files], [py_files/xxx.py]
· quote   : a verbatim sentence copied from that section (no edits/summaries)
If there is no evidence, return an empty array [].
You must output a JSON object only.
""".strip()

_BASE_SUMMARY_SYS = """
Using the provided quotes only, write long and detailed summaries for each item.
You must output a JSON object only.
""".strip()

_USAGE_SYS = """
You are a classifier. Decide whether the MODEL ITSELF (as released by the authors)
actually USED the following stages in its training pipeline:
- Fine-tuning (SFT/Instruction/Adapters/etc.)
- Reinforcement Learning (RLHF/DPO/PPO/etc.)

STRICT RULES:
- Do NOT infer "used" from generic advice like "you can fine-tune this model".
- "used" only if quotes explicitly state the authors performed that stage
  (e.g., "we fine-tuned", "post-trained on", "instruction-tuned", "SFT", "LoRA/QLoRA applied").
- "not_used" only if quotes explicitly deny it.
- Otherwise return "unknown".
- ✅ If the quotes describe ONLY supervised finetuning (e.g., xP3/MTF) and contain ZERO RL signals
  (RLHF/DPO/PPO/reward model/human feedback), classify reinforcement learning as "not_used".

Answer JSON only:
{ "fine_tuning": "used|not_used|unknown", "rl": "used|not_used|unknown" }
""".strip()

def _recall_inst(g: List[str], model_id: str) -> str:
    return (
        _model_guard_text(model_id) +
        "\nItems in this group:\n" + _js(_desc(g)) +
        "\nReturn a JSON object with EXACTLY these keys (arrays of {source,quote}):\n" +
        _js({LABELS[i]: [] for i in g})
    )

def _summ_inst(g: List[str], model_id: str) -> str:
    return (
        _model_guard_text(model_id) +
        "\nItems in this group:\n" + _js(_desc(g)) +
        "\nReturn a JSON object with EXACTLY these keys (string summaries):\n" +
        _js({LABELS[i]: "" for i in g}) +
        "\nUse ONLY the provided quotes."
    )

# ───────────── GPT ↔ JSON ─────────────
def _chat_json(sys_msg: str, usr: str) -> Dict[str, Any]:
    r = _client.chat.completions.create(
        model=MODEL_NAME, reasoning_effort="medium",
        response_format={"type": "json_object"},
        messages=[{"role":"system","content":sys_msg},
                  {"role":"user","content":usr}]
    )
    try:    return json.loads(r.choices[0].message.content.strip())
    except: return {}

# ───────────── Build payload ─────────────
def _make_payload(d: Dict, _: int) -> Dict:
    repo  = d.get("repo") or d.get("full_name") or ""
    files = (d.get("files") or [])[:3000]
    readme = (d.get("readme") or "")[:120_000]

    # license
    lic = d.get("license_files") or {}
    if isinstance(lic, dict):
        lic_text = "\n\n".join(
            f"# {k}\n{(v or '')[:20_000]}" for k, v in list(lic.items())[:5]
        )
    elif isinstance(lic, list):
        buf = []
        for it in lic[:5]:
            if isinstance(it, dict):
                name = it.get("name") or it.get("path") or "LICENSE"
                buf.append(f"# {name}\n{(it.get('content') or '')[:20_000]}")
            elif isinstance(it, str):
                buf.append(it[:20_000])
        lic_text = "\n\n".join(buf)
    else:
        lic_text = str(lic)[:20_000]

    py_files = {}
    for fn, src in (d.get("py_files") or {}).items():
        if len(py_files) >= 40: break
        py_files[fn] = (src or "")[:20_000]

    return {"repo": repo, "files": files, "readme": readme,
            "license_files": lic_text, "py_files": py_files}

def _payload_text(p: Dict) -> str:
    parts = [
        f"[repo]\n{p['repo']}\n",
        f"[readme]\n{p['readme']}\n",
        f"[license_files]\n{p['license_files']}\n",
    ]
    for fn, code in p["py_files"].items():
        parts.append(f"[py_files/{fn}]\n{code}\n")
    if p["files"]:
        parts.append("[files]\n" + "\n".join(map(str, p["files"])) + "\n")
    return "\n".join(parts)

# ───────────── Evidence filters ─────────────
_ALLOWED_PREFIX = ("repo", "readme", "license_files", "files", "py_files/")
_EXTRA_OK_PREFIX = ("section","sec.","appendix","table","figure","fig.")

def _valid_source(src: str) -> bool:
    if not isinstance(src, str): return False
    s = src.strip().lower().strip("[]")
    return s.startswith(_ALLOWED_PREFIX) or s.startswith(_EXTRA_OK_PREFIX)

def _filter_evidence_by_model(ev: Dict[str, List[Dict[str,str]]], model_id: str) -> Dict[str, List[Dict[str,str]]]:
    out: Dict[str, List[Dict[str,str]]] = {}
    for lbl, arr in ev.items():
        kept = []
        for e in arr or []:
            src = (e.get("source") or "").strip()
            qt  = (e.get("quote")  or "").strip()
            if not src or not qt: continue
            if not _valid_source(src): continue
            if not _quote_mentions_target(qt, model_id): continue
            kept.append({"source": src, "quote": qt})
        out[lbl] = _dedup_evid(kept, EVIDENCE_LIMIT_PER_KEY)
    return out

# ───────────── Step 1: collect (then post-filter) ─────────────
def _collect(g, text: str, model_id: str) -> Dict[str, List[Dict[str,str]]]:
    ev = {LABELS[k]: [] for k in g}
    for ch in _chunk(text, CHUNK_CHARS, CHUNK_OVERLAP):
        ans = _chat_json(_BASE_RECALL_SYS, _recall_inst(g, model_id) + "\n=== PAYLOAD ===\n" + ch)
        for k in g:
            arr = ans.get(LABELS[k], [])
            if isinstance(arr, list):
                ev[LABELS[k]].extend(arr)
    raw_counts = {k: len(v or []) for k, v in ev.items()}
    ev = _filter_evidence_by_model(ev, model_id)
    kept_counts = {k: len(v or []) for k, v in ev.items()}
    print("evidence counts before/after model-guard:", {"raw": raw_counts, "kept": kept_counts})
    return ev

# ───────────── Step 2: summarize ─────────────
def _summarize(g, ev: Dict[str, List[Dict[str,str]]], model_id: str) -> Dict[str, str]:
    lbls = [LABELS[k] for k in g]
    quotes = {lbl: [(e.get("quote") or "") for e in (ev.get(lbl) or [])] for lbl in lbls}
    ans = _chat_json(_BASE_SUMMARY_SYS, _summ_inst(g, model_id) + "\n=== QUOTES ===\n" + _js(quotes))
    return {lbl: (ans.get(lbl, "") or "") for lbl in lbls}

# ─── replace _merge function ───
def _merge(summary: Dict[str, Any],
           ev: Dict[str, List[Dict[str, str]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for lbl, val in summary.items():
        txt = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
        out[lbl] = txt.strip()
        out[f"{lbl}__evidence"] = ev.get(lbl, [])
    return out

def _merge_all(lst: List[Dict[str, Any]]) -> Dict[str, Any]:
    m: Dict[str, Any] = {}
    for d in lst:
        m.update(d)
    return m

# ───────────── Usage classifier (+ FT-only → RL not_used) ─────────────
_T_TOKENS = ("finetune","fine-tuning","instruction-tune","sft","xp3","xp3mt","mtf","prompted finetuning")
_RL_TOKENS = ("rlhf","reinforcement learning","dpo","ppo","reward model","preference model","human feedback","rlaif","kl penalty")

def _contains_any(text: str, toks: tuple[str, ...]) -> bool:
    tl = (text or "").lower()
    return any(t in tl for t in toks)

def _all_quotes(merged: dict) -> str:
    buf = []
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

def _classify_usage_from_merged(merged: Dict[str, Any]) -> Dict[str, str]:
    def _quotes(label: str):
        arr = merged.get(f"{label}__evidence", []) or []
        return [e.get("quote", "") for e in arr if isinstance(e, dict)]
    ft = "\n".join(_quotes("3-2 (Fine-tuning)"))
    rl = "\n".join(_quotes("3-3 (Reinforcement Learning)"))
    text = (f"[fine_tuning]\n{ft}\n\n[reinforcement]\n{rl}").strip()
    if not text:
        return {"fine_tuning": "unknown", "rl": "unknown"}
    ans = _chat_json(_USAGE_SYS, text[:12000])
    ft_s = ans.get("fine_tuning", "unknown")
    rl_s = ans.get("rl", "unknown")
    if ft_s not in {"used","not_used","unknown"}: ft_s = "unknown"
    if rl_s not in {"used","not_used","unknown"}: rl_s = "unknown"
    return {"fine_tuning": ft_s, "rl": rl_s}

# ───────────── Public function ─────────────
def filter_github_features(model: str, save: bool = True, output_dir: str | Path = ".") -> Dict[str, Any]:
    base = model.replace("/", "_")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"github_{base}.json"              # prefer input from outdir
    if not path.exists():
        alt = Path(f"github_{base}.json")                  # root fallback
        if not alt.exists():
            raise FileNotFoundError(str(path))
        gh = json.load(open(alt, encoding="utf-8"))
    else:
        gh = json.load(open(path, encoding="utf-8"))

    parts = []
    for idx, grp in enumerate(ITEM_GROUPS, 1):
        try:
            payload = _make_payload(gh, idx-1)
            text = _payload_text(payload)
            ev   = _collect(grp, text, model)
            summ = _summarize(grp, ev, model)
            part = _merge(summ, ev)
        except Exception as e:
            print(f"⚠️ Error processing group {idx}:", e)
            part = {}
        out = output_dir / f"github_filtered_{base}_{idx}.json"
        if save:
            json.dump(part, open(out,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
            print("✅ Saved group", idx, "result:", out)
        parts.append(part)

    merged = _merge_all(parts)

    try:
        usage = _classify_usage_from_merged(merged)
        if (usage.get("rl") in (None, "unknown")) and _rule_infer_rl_not_used(merged):
            usage["rl"] = "not_used"
        merged["__usage"] = usage
    except Exception as e:
        print("⚠️ Failed to classify usage:", e)
        merged["__usage"] = {"fine_tuning":"unknown","rl":"unknown"}

    if save:
        mpath = output_dir / f"github_filtered_final_{base}.json"
        json.dump(merged, open(mpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("✅ Saved final merged result:", mpath)
    return merged

# ───────────── CLI ─────────────
if __name__ == "__main__":
    import sys
    mid = "bigscience/bloomz-560m"
    if len(sys.argv) > 1 and sys.argv[1]:
        mid = sys.argv[1]
    print("▶ Model to run:", mid)
    filter_github_features(mid)
