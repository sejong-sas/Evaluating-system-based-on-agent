# huggingface_Dispatcher.py
# High-Recall 2-Pass  +  evidence({source, quote})  →  long summary
# - store evidence as an array of objects
# - summaries must use quotes only
# - remove __evidence_sources field

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
    LABELS["1-1"]: "All information about the availability, location, and access method of model weights, including whether anyone can download them",
    LABELS["1-2"]: "All information about whether the code for training and running the model is public, and which parts are public",
    LABELS["1-3"]: "All information about license existence/type and granted rights (use, modification, distribution, commercial use)",
    LABELS["1-4"]: "All information about official papers, technical reports, blogs, and links related to the model",
    LABELS["1-5"]: "All information about model architecture (number of layers, hyperparameters, etc.) and design details",
    LABELS["1-6"]: "All information about which tokenizer is used, its name/structure, and whether it is downloadable",
    LABELS["2-1"]: "All information about the hardware used for training (H100, TPU, etc.), quantity, and compute scale",
    LABELS["2-2"]: "All information about software used for training (frameworks, libraries), versions, and settings",
    LABELS["2-3"]: "All information about the existence of an accessible API (e.g., GPT API, Gemini API; libraries do not count), docs, examples, and public availability",
    LABELS["3-1"]: "All information about pre-training methodology, procedure, data flow, and hyperparameter settings",
    LABELS["3-2"]: "All information about fine-tuning methods, goals, whether data is used, and existence of a reproducible pipeline",
    LABELS["3-3"]: "All information about the use of RLHF, DPO, etc., including concrete methods, procedures, and parameters",
    LABELS["4-1"]: "All information about the types, quantities, sources, allowed use, and composition of pre-training data",
    LABELS["4-2"]: "All information about the source, composition, examples, and public availability of fine-tuning datasets",
    LABELS["4-3"]: "All information about the composition, accessibility, sources, and generation of reinforcement learning datasets",
    LABELS["4-4"]: "All information about data filtering/cleaning methods, criteria used, processes, and their impacts",
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

# (Note) Not used now, but can be used for tag validation
_SRC_TAG_RE = re.compile(r'^\s*\[([^\]]+)\]\s*.*$')

# ─────────────────────────────── Utils ───────────────────────────────
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
You are a classifier. Based only on the input text (quotes/summaries), determine whether this model actually used
Fine-tuning / Reinforcement Learning.
JSON only:
{ "fine_tuning": "used|not_used|unknown", "rl": "used|not_used|unknown" }
"""

def _classify_usage_from_merged(merged: dict) -> dict:  # whether RL / fine-tuning were used
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


def _build_recall_inst(group: List[str]) -> str:
    desc = _json(_group_desc_map(group))
    example = _json({
        LABELS[k]: [
            {"source": "readme", "quote": "Example original sentence 1"},
            {"source": "py_files/train.py", "quote": "Example original sentence 2"}
        ] for k in group
    })
    return (
        f"Definitions for this group:\n{desc}\n"
        "For each key, return an array of evidence objects. Example schema:\n"
        f"{example}"
    )

def _build_summary_inst(group: List[str]) -> str:
    desc = _json(_group_desc_map(group))
    return (
        f"Definitions for this group:\n{desc}\n"
        "You will receive an array of quotes next. Write long summaries for each item."
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
_ALLOWED_PREFIX = ("model_id", "readme", "license_file", "config",
                   "generation_config", "files", "py_files/")

def _is_valid_source(src: str) -> bool:
    return isinstance(src, str) and src.startswith(_ALLOWED_PREFIX)

def _recall_collect(group: List[str], text: str) -> Dict[str, List[Dict[str, str]]]:
    chunks = _chunk_text(text, CHUNK_CHARS, CHUNK_OVERLAP)
    out: Dict[str, List[Dict[str, str]]] = {LABELS[k]: [] for k in group}

    for chunk in chunks:
        ans = _chat_json(_BASE_RECALL_SYS, _build_recall_inst(group) +
                         "\n=== PAYLOAD ===\n" + chunk)

        for k in group:
            lbl = LABELS[k]
            evs = ans.get(lbl, [])
            if not isinstance(evs, list):
                continue
            # type validation + dedup
            out[lbl].extend(_dedup_evidences(evs, EVIDENCE_LIMIT_PER_KEY))

    return out

# ─────────────────────────────── Summary generation ──────────────────────────────
def _summarize(group: List[str], evid: Dict[str, List[Dict[str, str]]]) -> Dict[str, str]:
    quotes = {
        LABELS[k]: [e["quote"] for e in evid[LABELS[k]]]
        for k in group
    }
    ans = _chat_json(_BASE_SUMMARY_SYS, _build_summary_inst(group) +
                     "\n=== EVIDENCE_QUOTES ===\n" + _json(quotes))
    return {LABELS[k]: ans.get(LABELS[k], "") for k in group}

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
            evid = _recall_collect(grp, text)
            summ = _summarize(grp, evid)
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

    # Usage: python huggingface_Dispatcher.py <org/model> [output_dir]
    if len(sys.argv) >= 2 and sys.argv[1]:
        model_id = sys.argv[1]
    if len(sys.argv) >= 3 and sys.argv[2]:
        outdir = sys.argv[2]

    print("▶ Model to run:", model_id)
    print("▶ Output folder:", outdir)
    filter_hf_features(model_id, output_dir=outdir)
