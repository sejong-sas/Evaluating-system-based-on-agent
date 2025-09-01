# huggingface_Dispatcher.py
# High-Recall 2-Pass  +  evidence({source, quote})  →  long summary
# - store evidence as an array of objects
# - summaries must use quotes only
# - remove __evidence_sources field
# - STRICT: collect ONLY quotes that explicitly mention the TARGET model
#   (단, README의 불릿/Key:Value 라인은 모델 소개 직후 이어지는 규격 정보로 간주하여 예외 허용)

import os
import json
import re
from typing import Dict, List, Any
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
    "1-1": "1-1 (Weights)",
    "1-2": "1-2 (Code)",
    "1-3": "1-3 (License)",
    "1-4": "1-4 (Paper)",
    "1-5": "1-5 (Architecture)",
    "1-6": "1-6 (Tokenizer)",
    "2-1": "2-1 (Hardware)",
    "2-2": "2-2 (Software)",
    "2-3": "2-3 (API)",
    "3-1": "3-1 (Pre-training)",
    "3-2": "3-2 (Fine-tuning)",
    "3-3": "3-3 (Reinforcement Learning)",
    "4-1": "4-1 (Pre-training Data)",
    "4-2": "4-2 (Fine-tuning Data)",
    "4-3": "4-3 (Reinforcement Learning Data)",
    "4-4": "4-4 (Data Filtering)",
}

# ──────────────────────────── Item descriptions ────────────────────────────
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
MODEL_NAME = os.getenv("OPENAI_MODEL_HF_DISPATCHER", "o3-mini")

# ─────────────────────────────── Utils ───────────────────────────────
def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def _js(obj: Any) -> str:
    return _json(obj)

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
        src = (ev.get("source") or "").strip()
        qt  = (ev.get("quote") or "").strip()
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

def _desc(ids: List[str]) -> Dict[str, str]:
    return _group_desc_map(ids)

def _skeleton(ids: List[str]) -> Dict[str, list]:
    return {LABELS[i]: [] for i in ids}

# ─────────────────────────────── Prompts ───────────────────────────────
_BASE_RECALL_SYS = """
You are an expert at extracting AI model openness evaluation information from a Hugging Face repository.
Use only the provided payload (original text).
For each item, return an array of evidence objects.
Each evidence object must include:
- "source": a payload section tag (e.g., "readme", "files", "py_files/filename.py")
- "quote" : a verbatim sentence copied from that section (no edits/summaries)
If there is no evidence, return an empty array [].
You must return a JSON object only.
""".strip()

_BASE_SUMMARY_SYS = """
You are an expert who writes long, detailed summaries for each item using evidence quotes only.
You must return a JSON object only (no extra text).
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

# === Target-model guard (공통) =========================================
def _canonical_model_tokens(model_id: str) -> list:
    """모델 ID에서 비교적 안정적인 토큰/별칭 후보 추출 (짧고 흔한 토큰 제외)."""
    name = (model_id or "").split("/", 1)[-1].lower()
    raw = re.split(r"[^a-z0-9.]+", name)
    alts = set()
    for t in raw:
        t = t.strip()
        if len(t) >= 3 and t not in {"base","it","instruct","chat","model"}:
            alts.add(t)
    # 숫자/구두점 제거/축약형까지 추가
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
        "\nEXCEPTION:\n"
        "- Bullet or key:value lines from the README that immediately follow the model introduction may be accepted\n"
        "  even if they don't repeat the TARGET token verbatim (common in model cards: spec bullets).\n"
    )

def _recall_inst(group: List[str], model_id: str) -> str:
    return (
        _model_guard_text(model_id) +
        "\nItems in this group:\n" + _js(_desc(group)) +
        "\nReturn a JSON object with EXACTLY these keys (arrays of {source,quote}):\n" +
        _js(_skeleton(group))
    )

def _summ_inst(group: List[str], model_id: str) -> str:
    return (
        _model_guard_text(model_id) +
        "\nItems in this group:\n" + _js(_desc(group)) +
        "\nReturn a JSON object with EXACTLY these keys (string summaries):\n" +
        _js({LABELS[i]: "" for i in group}) +
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
    py_items = list(py_src.items())[:20]         # up to 20 files
    py_files = {fn: (src[:20_000] if isinstance(src, str) else "")
                for fn, src in py_items}
    return {
        "model_id":          hf.get("model_id", ""),
        "files":             hf.get("files", [])[:2000],
        "readme":            hf.get("readme", ""),
        "license_file":      hf.get("license_file", ""),
        "config":            hf.get("config", ""),
        "generation_config": hf.get("generation_config", ""),
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
# ⬇️ 확장: README/License 표기 변주와 일반적 헤딩/섹션 표기 허용
_ALLOWED_PREFIX = (
    "model_id", "readme", "readme.md", "modelcard", "card", "carddata",
    "license", "license_file", "license.md",
    "config", "generation_config", "files", "py_files/"
)
_EXTRA_OK_PREFIX = (
    "section", "sec.", "appendix", "table", "figure", "fig.", "header", "heading"
)

def _is_valid_source(src: str) -> bool:
    if not isinstance(src, str):
        return False
    s = src.strip().lower()
    s = s.strip("[]")  # allow bracketed tags like [readme]
    return s.startswith(_ALLOWED_PREFIX) or s.startswith(_EXTRA_OK_PREFIX)

# ⬇️ README 불릿/Key:Value 예외 허용
_BULLET_OR_KV = re.compile(r"^\s*(?:[-*•]\s+|[A-Za-z0-9 _/()+\.\-]+:\s+\S+)")

def _quote_mentions_target(q: str, model_id: str, *, source: str = "") -> bool:
    if not q:
        return False
    ql = q.lower()
    toks = _canonical_model_tokens(model_id)

    # 1) 원 규칙: 인용문 자체에 토큰 포함
    if any(t for t in toks if t and t in ql):
        return True

    # 2) 예외 규칙: README의 불릿/Key:Value 라인은 허용
    s = (source or "").lower()
    if s.startswith("readme") and _BULLET_OR_KV.match(q or ""):
        return True

    return False

def _filter_evidence_by_model(ev: Dict[str, List[Dict[str, str]]], model_id: str) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = {}
    for lbl, arr in ev.items():
        kept = []
        for e in arr or []:
            src = (e.get("source") or "").strip()
            qt  = (e.get("quote")  or "").strip()
            if not src or not qt:
                continue
            if not _is_valid_source(src):
                continue
            if not _quote_mentions_target(qt, model_id, source=src):
                continue
            kept.append({"source": src, "quote": qt})
        out[lbl] = _dedup_evidences(kept, EVIDENCE_LIMIT_PER_KEY)
    return out

def _rebalance_evidence(ev: Dict[str, List[Dict[str,str]]]) -> Dict[str, List[Dict[str,str]]]:
    """If 4-2 is empty but 3-2 contains dataset-ish quotes, copy a subset."""
    ft_lbl = "3-2 (Fine-tuning)"
    fd_lbl = "4-2 (Fine-tuning Data)"
    if (not ev.get(fd_lbl)) and ev.get(ft_lbl):
        kws = r"(dataset|corpus|xP3(?:mt)?|p3\b|language distribution|languages|split|examples|released|Flores200|ROOTS|license|public links?)"
        moved = [e for e in ev[ft_lbl] if isinstance(e, dict) and re.search(kws, e.get("quote",""), re.I)]
        if moved:
            ev[fd_lbl] = _dedup_evidences(moved, EVIDENCE_LIMIT_PER_KEY)
    return ev

# ⬇️ 백스톱: 가중치/라이선스 최소 근거 자동 주입
def _inject_backstops_if_empty(ev: Dict[str, List[Dict[str,str]]], payload: Dict[str, Any]) -> Dict[str, List[Dict[str,str]]]:
    files = [str(x) for x in (payload.get("files") or [])]
    # Weights (파일 목록에서 확장자 히트)
    lbl_w = "1-1 (Weights)"
    if lbl_w in ev and not ev.get(lbl_w):
        hits = [f for f in files if re.search(r"\.(?:safetensors|bin|pt|onnx|gguf|ckpt)$", f, re.I)]
        if hits:
            ev[lbl_w] = [{"source":"files","quote":hits[0]}]
    # License (license_file 첫 줄)
    lbl_l = "1-3 (License)"
    if lbl_l in ev and not ev.get(lbl_l):
        lic = (payload.get("license_file") or "").strip().splitlines()
        line = ""
        for ln in lic:
            ln = (ln or "").strip()
            if ln:
                line = ln[:240]
                break
        if line:
            ev[lbl_l] = [{"source":"license_file","quote":line}]
    return ev

def _collect(group: List[str], text: str, model_id: str) -> Dict[str, List[Dict[str, str]]]:
    chunks = _chunk_text(text, CHUNK_CHARS, CHUNK_OVERLAP)
    out: Dict[str, List[Dict[str, str]]] = {LABELS[k]: [] for k in group}

    for chunk in chunks:
        ans = _chat_json(
            _BASE_RECALL_SYS,
            _recall_inst(group, model_id) + "\n=== PAYLOAD ===\n" + chunk
        )
        for k in group:
            lbl = LABELS[k]
            evs = ans.get(lbl, [])
            if isinstance(evs, list):
                out[lbl].extend(evs)

    # post-filter: source validity + model-mention guard + dedup
    raw_counts = {lbl: len(out.get(lbl) or []) for lbl in out}
    out = _filter_evidence_by_model(out, model_id)
    out = _rebalance_evidence(out)
    kept_counts = {lbl: len(out.get(lbl) or []) for lbl in out}
    print("evidence counts before/after model-guard:", {"raw": raw_counts, "kept": kept_counts})
    return out

# ─────────────────────────────── Summary generation ──────────────────────────────
def _summarize(group: List[str], evid: Dict[str, List[Dict[str, str]]], model_id: str) -> Dict[str, str]:
    quotes = {LABELS[k]: [e["quote"] for e in (evid.get(LABELS[k]) or [])] for k in group}
    ans = _chat_json(
        _BASE_SUMMARY_SYS,
        _summ_inst(group, model_id) + "\n=== EVIDENCE_QUOTES ===\n" + _js(quotes)
    )
    return {LABELS[k]: (ans.get(LABELS[k], "") or "") for k in group}

# ─────────────────────────────── Merge utils ───────────────────────────────
def _merge_for_final(summary: Dict[str, str],
                     evid: Dict[str, List[Dict[str, str]]]) -> Dict[str, Any]:
    final = {}
    for lbl, txt in summary.items():
        final[lbl] = txt.strip()
        final[f"{lbl}__evidence"] = evid.get(lbl, [])
    return final

def _merge_dicts(ds: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged = {}
    for d in ds:
        merged.update(d)
    return merged

# ───────────────────────────── RL/FT usage classifier (+heuristic) ─────────────────────────────
_T_TOKENS = ["finetune", "fine-tuning", "instruction-tune", "sft", "xp3", "xp3mt", "mtf", "prompted finetuning"]
_RL_TOKENS = ["rlhf", "reinforcement learning", "dpo", "ppo", "reward model", "preference model", "human feedback", "rlaif", "kl penalty"]

def _contains_any(text: str, toks: List[str]) -> bool:
    tl = (text or "").lower()
    return any(t in tl for t in toks)

def _all_quotes(merged: dict) -> str:
    buf = []
    for k, v in merged.items():
        if not (isinstance(k, str) and k.endswith("__evidence")):
            continue
        for e in (v or []):
            if isinstance(e, dict):
                q = e.get("quote") or ""
                if q:
                    buf.append(q)
    return "\n".join(buf)

def _rule_infer_rl_not_used(merged: dict) -> bool:
    q = _all_quotes(merged)
    return (_contains_any(q, _T_TOKENS) and not _contains_any(q, _RL_TOKENS))

def _classify_usage_from_merged(merged: dict) -> dict:
    """whether RL / fine-tuning were used (model-card based)"""
    def _pull(label):
        txt = merged.get(label, "") or ""
        evs = merged.get(f"{label}__evidence", []) or []
        quotes = "\n".join([e.get("quote","") for e in evs if isinstance(e, dict)])
        return (txt + "\n" + quotes).strip()
    ft_txt = _pull("3-2 (Fine-tuning)")
    rl_txt = _pull("3-3 (Reinforcement Learning)")
    text = f"[fine_tuning]\n{ft_txt}\n\n[reinforcement]\n{rl_txt}".strip()
    if not text:
        return {"fine_tuning":"unknown","rl":"unknown"}
    ans = _chat_json(_USAGE_SYS, text[:12000])
    ft_s = ans.get("fine_tuning","unknown"); rl_s = ans.get("rl","unknown")
    if ft_s not in {"used","not_used","unknown"}: ft_s = "unknown"
    if rl_s not in {"used","not_used","unknown"}: rl_s = "unknown"
    return {"fine_tuning": ft_s, "rl": rl_s}

# ────────────────────────────── Main function ────────────────────────────────
def filter_hf_features(model: str, save: bool = True, output_dir: str | Path = ".") -> Dict[str, Any]:
    base = model.replace("/", "_").lower()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Input JSON: look in output_dir first; if missing, fall back to project root
    path_in = output_dir / f"huggingface_{base}.json"
    if not path_in.exists():
        alt = Path(f"huggingface_{base}.json")
        if not alt.exists():
            raise FileNotFoundError(str(path_in))
        hf = json.load(open(alt, encoding="utf-8"))
    else:
        hf = json.load(open(path_in, encoding="utf-8"))

    parts = []
    for idx, grp in enumerate(ITEM_GROUPS, 1):
        try:
            payload = _make_group_payload(hf, idx - 1)
            text = _payload_to_text(payload)
            evid = _collect(grp, text, model)
            # ⬇️ 백스톱 주입 (가중치/라이선스 최소 근거)
            evid = _inject_backstops_if_empty(evid, payload)
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

    # usage classification (+ heuristic: FT-only → RL not_used)
    try:
        usage = _classify_usage_from_merged(merged)
        if (usage.get("rl") in (None, "unknown")) and _rule_infer_rl_not_used(merged):
            usage["rl"] = "not_used"
        merged["__usage"] = usage
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

    # Usage: python huggingface_Dispatcher.py <org/model> [output_dir]
    if len(sys.argv) >= 2 and sys.argv[1]:
        model_id = sys.argv[1]
    if len(sys.argv) >= 3 and sys.argv[2]:
        outdir = sys.argv[2]

    print("▶ Model to run:", model_id)
    print("▶ Output folder:", outdir)
    filter_hf_features(model_id, output_dir=outdir)
