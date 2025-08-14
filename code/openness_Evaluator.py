# openness_Evaluator.py
# ─────────────────────────────────────────────────────────
# • If a Hugging Face model exists: auto 1 point for 1-1, 1-5, 1-6
# • The remaining 13 items are evaluated by GPT
# • For training methodology (3-1~3-3), prioritize arxiv_Dispatcher JSON
# • Enforce GPT response schema: {"scores": {...}, "total_score": ...}
# ─────────────────────────────────────────────────────────

import os, json
from typing import Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

load_dotenv()
_API_KEY = os.getenv("OPENAI_API_KEY")
if not _API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
client = OpenAI(api_key=_API_KEY)

# ─────────── Full evaluation criteria ───────────
CRITERIA_TEXT = """
## 1. Model Basic Openness – 6 items
### 1-1. Weights – If hosted on Hugging Face, always Open
- Open (1): Model weights are publicly available without permission
- Semi-Open (0.5): Model weights are available after approval
- Closed (0): Model weights are not available for use
### 1-2. Code – If there are .py files on Hugging Face, always Open
- Open (1): Full code for training and implementation is public
- Semi-Open (0.5): Only part of the training/implementation code is public
- Closed (0): Training/implementation code is not public
### 1-3. License
- Open (1): No restrictions on use, modification, redistribution, and commercial use (e.g., MIT, Apache)
- Semi-Open (0.5): One or more restrictions on use, modification, redistribution, or commercial use
- Closed (0): Three or more restrictions exist, or no applicable license
### 1-4. Paper
- Open (1): Official paper or technical report exists
- Semi-Open (0.5): Website or blog post exists
- Closed (0): No related document
### 1-5. Architecture – If hosted on Hugging Face, always Open
- Open (1): Model structure and hyperparameters are fully disclosed
- Semi-Open (0.5): Only the model structure is disclosed
- Closed (0): Model structure info is not disclosed
### 1-6. Tokenizer – If hosted on Hugging Face, always Open
- Open (1): Tokenizer used is explicitly disclosed
- Semi-Open (0.5): A downloadable tokenizer exists
- Closed (0): Tokenizer info is not disclosed

## 2. Accessibility and Reproducibility – 3 items
### 2-1. Hardware
- Open (1): Training hardware type and quantity fully disclosed
- Semi-Open (0.5): Only hardware type disclosed
- Closed (0): No hardware info
### 2-2. Software
- Open (1): Full software specifications for training disclosed
- Semi-Open (0.5): Only partial disclosure
- Closed (0): No information
### 2-3. API
- Open (1): Public API available
- Semi-Open (0.5): Planned to be made public
- Closed (0): No API

## 3. Training Methodology Openness – 3 items
### 3-1. Pre-training
- Open (1): Detailed disclosure sufficient for reproduction
- Semi-Open (0.5): Only partial methods mentioned
- Closed (0): Methods not disclosed
### 3-2. Fine-tuning
- Open (1): Methodology fully disclosed
- Semi-Open (0.5): Partial disclosure
- Closed (0): Not disclosed/N/A
### 3-3. Reinforcement Learning
- Open (1): RLHF, DPO, etc. disclosed in detail
- Semi-Open (0.5): Partial disclosure
- Closed (0): Not disclosed/N/A

## 4. Data Openness – 4 items
### 4-1. Pre-training Data
- Open (1): Quantity and sources fully disclosed
- Semi-Open (0.5): Only types disclosed
- Closed (0): Not disclosed
### 4-2. Fine-tuning Data
- Open (1): Data fully disclosed
- Semi-Open (0.5): Partial disclosure
- Closed (0): Not disclosed/N/A
### 4-3. Reinforcement Learning Data
- Open (1): Data fully disclosed
- Semi-Open (0.5): Partial disclosure
- Closed (0): Not disclosed/N/A
### 4-4. Data Filtering
- Open (1): Filtering methodology and contents fully disclosed
- Semi-Open (0.5): Partial disclosure
- Closed (0): Not disclosed
""".strip()

# ─────────── GPT system prompt ───────────
EVALUATION_PROMPT = f"""
{CRITERIA_TEXT}

❗️For Training Methodology Openness (3-1 ~ 3-3), consult the JSON produced by arxiv_Dispatcher (paper information) **first**.
Hugging Face and GitHub information may be used only as secondary references.

Also, because the model exists on Hugging Face, **the following three items are already Open (1 point)**:
  • 1-1 Weights • 1-5 Architecture • 1-6 Tokenizer
→ Do not include these three items in the 'scores'.

Return a single JSON block exactly like the schema below:

{{
  "scores": {{
    "1-2 Code": {{ "score": 1,   "reason": "..." }},
    ...
  }},
  "total_score": 12.5
}}
Do not include any extra comments, backticks, or unnecessary text.
- For items 3-1 and 4-1: if a data.pretrain list exists, first look for evidence in that list (the pretrained model info).  # added
""".strip()

# ─────────── Auto-Open items (1 point) ───────────
AUTO_OPEN_LABELS = {
    "1-1 Weights":     "Model weights are published on Hugging Face",
    "1-5 Architecture": "Architecture info disclosed on the Hugging Face card",
    "1-6 Tokenizer":   "Tokenizer info disclosed on the Hugging Face card/config",
}

# ─────────── Pretrained JSON loader helper ───────────
def _load_pretrain_parts(base_id: str | None, base_dir: Path) -> list[dict]:
    """
    Load pretrain_hf|gh|arxiv_{base}.json (3 types) → return as a list
    """
    if not base_id:
        return []
    b = base_id.replace("/", "_").lower()
    out = []
    for src in ["hf", "gh", "arxiv"]:
        p = base_dir / f"pretrain_{src}_{b}.json"
        if p.exists() and p.stat().st_size:
            try:
                out.append(json.load(open(p, encoding="utf-8")))
            except json.JSONDecodeError:
                print("⚠️ JSON parsing failed:", p)
    return out

def _auto_scores(hf_json: Dict[str, Any]) -> Dict[str, Dict]:
    return {lbl: {"score": 1, "reason": reason}
            for lbl, reason in AUTO_OPEN_LABELS.items()} if hf_json else {}

# ─────────── GPT evaluation function ───────────
def _gpt_evaluate(model: str,
                  hf: Dict, gh: Dict, ax: Dict,
                  pretrain: list[Dict]) -> Dict[str, Dict]: 
    payload = {
        "model": model,
        "data": {"pretrain": pretrain, "huggingface": hf, "github": gh, "arxiv": ax}
    }
    rsp = client.chat.completions.create(
        model="o3-mini",
        reasoning_effort="medium",
        response_format={"type":"json_object"},
        messages=[
            {"role":"system","content":EVALUATION_PROMPT},
            {"role":"user",  "content":json.dumps(payload, ensure_ascii=False)}
        ]
    )
    raw = json.loads(rsp.choices[0].message.content.strip())
    scores_dict = raw.get("scores", raw)      # flexible parsing
    out = {}
    for k, v in scores_dict.items():
        if isinstance(v, dict):
            out[k] = {"score": v.get("score", 0), "reason": v.get("reason","")}
        elif isinstance(v, (int, float)):
            out[k] = {"score": v, "reason": ""}
    return out

# ─────────── Main evaluation helpers ───────────
def _read_usage_from_dispatch(hf: dict, gh: dict, ax: dict) -> dict:
    # Priority: arXiv > HF > GH  (switch to a voting rule if desired)
    for src in (ax, hf, gh):
        u = (src or {}).get("__usage")
        if isinstance(u, dict):
            ft = u.get("fine_tuning", "unknown")
            rl = u.get("rl", "unknown")
            if ft in {"used","not_used"} or rl in {"used","not_used"}:
                return {"fine_tuning": ft, "rl": rl}
    # If everything is absent or unknown → unknown
    return {"fine_tuning":"unknown", "rl":"unknown"}

def _aggregate_usage(hf: dict, gh: dict, ax: dict) -> dict:
    def norm(x): return x if x in {"used","not_used"} else "unknown"
    votes = []
    for src in (ax, hf, gh):
        u = (src or {}).get("__usage") or {}
        votes.append({
            "ft": norm(u.get("fine_tuning")),
            "rl": norm(u.get("rl")),
        })

    def decide(key):
        vals = [v[key] for v in votes if v[key] != "unknown"]
        if not vals:
            return "unknown"
        if "used" in vals:
            return "used"
        return "not_used"

    return {"fine_tuning": decide("ft"), "rl": decide("rl")}

def evaluate_openness(model_name: str,
                      hf_json=None, gh_json=None, arxiv_json=None, pretrain_parts=None) -> Dict:
    hf, gh, ax = hf_json or {}, gh_json or {}, arxiv_json or {}
    pretrain = pretrain_parts or []

    scores = _gpt_evaluate(model_name, hf, gh, ax, pretrain)
    scores.update(_auto_scores(hf))  # 1-1/1-5/1-6

    # ★ Decide only from usage inserted by Dispatchers
    usage = _aggregate_usage(hf_json, gh_json, arxiv_json)

    exclude = []
    if usage["fine_tuning"] == "not_used":
        exclude += ["3-2", "4-2"]
    if usage["rl"] == "not_used":
        exclude += ["3-3", "4-3"]

    included = {k:v for k,v in scores.items()
                if not any(k.startswith(p) for p in exclude)}

    raw_sum = sum(v.get("score",0) for v in included.values())
    denom  = max(len(included), 1)
    final_10 = round(raw_sum * (10.0 / denom), 3)

    return {
        "model": model_name,
        "scores": scores,
        "included_scores": included,
        "final_score_10pt": final_10,
        "meta": {
            "usage_from_dispatch": usage,
            "excluded": [k for k in scores if k not in included],
            "denominator": denom,
            "raw_sum": raw_sum,
            "scale": f"10/{denom}"
        }
    }


# ─────────── File loader & CLI ───────────
def _load(p):
    if os.path.exists(p) and os.path.getsize(p):
        try:
            return json.load(open(p,encoding="utf-8"))
        except json.JSONDecodeError:
            print("⚠️ JSON parsing failed:", p)
    return {}

def evaluate_openness_from_files(model_name: str,
                                 base_dir: str | Path = ".",
                                 base_model_id: str | None = None):
    base = model_name.replace("/", "_").lower()
    base_dir = Path(base_dir)

    # Prefer the given folder; if missing, fall back to project root
    def _load_from_base(filename: str):
        p = base_dir / filename
        if p.exists() and p.stat().st_size:
            try:
                return json.load(open(p, encoding="utf-8"))
            except json.JSONDecodeError:
                print("⚠️ JSON parsing failed:", p)
        # root fallback
        if os.path.exists(filename) and os.path.getsize(filename):
            try:
                return json.load(open(filename, encoding="utf-8"))
            except json.JSONDecodeError:
                print("⚠️ JSON parsing failed:", filename)
        return {}

    hf = _load_from_base(f"huggingface_filtered_final_{base}.json")
    gh = _load_from_base(f"github_filtered_final_{base}.json")
    ax = _load_from_base(f"arxiv_filtered_final_{base}.json")

    pretrain_parts = _load_pretrain_parts(base_model_id, base_dir)

    res = evaluate_openness(model_name, hf_json=hf,gh_json=gh, arxiv_json=ax, pretrain_parts=pretrain_parts)
    out = base_dir / f"openness_score_{base}.json"
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("📝 Saved evaluation result:", out)
    return res


if __name__ == "__main__":
    evaluate_openness_from_files("bigscience/bloomz-560m")
