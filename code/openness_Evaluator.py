# openness_Evaluator.py  — STRICT + Code-as-Training transparency + Pretrain sources merge
import os, json, re
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

load_dotenv()
_API_KEY = os.getenv("OPENAI_API_KEY")
if not _API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
client = OpenAI(api_key=_API_KEY)

# ─────────── Label maps (dispatcher ↔ scoring keys) ───────────
LBL_MAP_ARX_TO_SCORE = {
    "3-1 (Pre-training)": "3-1 Pre-training",
    "3-2 (Fine-tuning)": "3-2 Fine-tuning",
    "3-3 (Reinforcement Learning)": "3-3 Reinforcement Learning",
    "4-1 (Pre-training Data)": "4-1 Pre-training Data",
    "4-2 (Fine-tuning Data)": "4-2 Fine-tuning Data",
    "4-3 (Reinforcement Learning Data)": "4-3 Reinforcement Learning Data",
    "4-4 (Data Filtering)": "4-4 Data Filtering",
}
STRICT_KEYS = set(LBL_MAP_ARX_TO_SCORE.values())

# ─────────── Evaluation rubric text (STRICT) ───────────
CRITERIA_TEXT = """
## 1. Model Basic Openness – 6 items
### 1-1. Weights
- Open (1): Model weights are publicly available without permission
- Semi-Open (0.5): Model weights are available after approval
- Closed (0): Model weights are not available for use

### 1-2. Code (TRAINING vs inference/serving)
- Open (1): End-to-end TRAINING code is public and sufficient to reproduce training (data prep, configs, scripts, schedules).
- Semi-Open (0.5): Some TRAINING code is public (e.g., fine-tuning scripts, adapters, RL scripts) but not a full pipeline.
- Closed (0): Only inference/serving/evaluation code is public, or no code at all.

### 1-3. License
- Open (1): License explicitly allows **all four** rights — use, modification, redistribution, and commercial use — with no additional restrictions (e.g., MIT, Apache-2.0).
- Semi-Open (0.5): **One or two** rights are restricted (e.g., non-commercial / no redistribution / no derivatives / research-only), or terms partially restrict usage.
- Closed (0): **Three or more** rights are restricted, **or** there is **no license** / undefined custom terms that substantially limit usage.

- Open (1): A peer-reviewed paper or an official technical report that is **specifically about the TARGET model** (exact version/variant).
- Semi-Open (0.5): A model blog/announcement page or a model card that **primarily reports results** but is not a full technical report.
- Closed (0): No document **for this model**, or the document is about a **different model/family** or a **previous version**.

### 1-5. Architecture
- Open (1): Model structure and hyperparameters are fully disclosed
- Semi-Open (0.5): Only partial disclosure
- Closed (0): Not disclosed

### 1-6. Tokenizer
- Open (1): Tokenizer details disclosed (name/structure) and/or downloadable
- Semi-Open (0.5): Partial disclosure
- Closed (0): Not disclosed

## 2. Accessibility and Reproducibility – 3 items
### 2-1. Hardware
- Open (1): Training hardware type and quantity fully disclosed
- Semi-Open (0.5): Only type or partial quantities
- Closed (0): No info

### 2-2. Software
- Open (1):
   (a) Full training software stack is disclosed with open components and versions, typically including:
       base framework (e.g., PyTorch/JAX + version), orchestration/distributed (e.g., DeepSpeed, Megatron/FSDP, NeMo),
       precision/optimizer libs (e.g., Apex/bitsandbytes) and CUDA/TPU/XLA versions; and
   (b) No proprietary/internal-only framework governs training.
- Semi-Open (0.5):
   (a) An additional open component beyond the base framework is disclosed (e.g., DeepSpeed, Megatron, FSDP, Apex, XLA, Pathways), but
       versions and/or the full training stack are incomplete; or
   (b) Mixed use of open + internal systems (e.g., “JAX + Pathways”).
- Closed (0):
   (a) Only a generic framework is mentioned (e.g., just “PyTorch”/“JAX”/“TensorFlow”), or
   (b) A proprietary/internal training framework is used (e.g., “HAI-LLM”, “proprietary/internal framework”), or
   (c) Only inference/benchmark environment is described without training stack.

### 2-3. API
- Open (1): Public API available
- Semi-Open (0.5): Planned to be made public
- Closed (0): No API

## 3. Training Methodology – STRICT (quotes required)
### 3-1. Pre-training
- Open (1): Fully reproducible method (pipeline, objectives, schedules, all hyperparameters)
- Semi-Open (0.5): Partial method disclosure
- Closed (0): No method disclosed (default if no quotes)

### 3-2. Fine-tuning
- Open (1): Fully reproducible fine-tuning details
- Semi-Open (0.5): Partial details
- Closed (0): Not disclosed/N/A (default if no quotes)

### 3-3. Reinforcement Learning
- Open (1): RL methods detailed for reproduction
- Semi-Open (0.5): Partial disclosure
- Closed (0): Not disclosed/N/A (default if no quotes)

## 4. Data Openness – STRICT (quotes required)
### 4-1. Pre-training Data
- Open (1): Full disclosure (sources, quantities/proportions, licensing, access) to rebuild corpus
- Semi-Open (0.5): Partial disclosure (types, language mix %, subsets)
- Closed (0): No info (default if no quotes)

### 4-2. Fine-tuning Data
- Open (1): Full disclosure (names, sizes, availability)
- Semi-Open (0.5): Partial disclosure
- Closed (0): No info (default if no quotes)

### 4-3. Reinforcement Learning Data
- Open (1): Full disclosure (source, composition, sizes, availability)
- Semi-Open (0.5): Partial disclosure
- Closed (0): No info (default if no quotes)

### 4-4. Data Filtering
- Open (1): Full disclosure of filtering/cleaning criteria and impact
- Semi-Open (0.5): Partial disclosure
- Closed (0): No info (default if no quotes)
""".strip()

EVALUATION_PROMPT = (
    CRITERIA_TEXT
    + "\n\nHARD RULES:\n"
    + "1) For items 3-1~3-3 and 4-1~4-4, you MUST base your decision ONLY on provided quotes.\n"
    + "2) If there is NO quote evidence for those items, the item is CLOSED (0).\n"
    + "3) SEMI-OPEN (0.5) requires some direct evidence; OPEN (1) requires fully reproducible methods or fully disclosed datasets.\n"
    + "4) You may use auxiliary JSON for context, but NEVER override rule (1).\n"
    + "5) Do NOT treat data scale/types/hardware alone as methodology disclosure. Without concrete method details\n"
    + "   (objectives, training schedules, key hyperparameters, or an executable pipeline), score it CLOSED (0).\n"
    + "6) For **1-4 Paper**, apply the rubric above **ONLY to documents about the TARGET model** (payload.model). "
    + "   Blogs/model cards are Semi-Open; papers/tech reports about other models or older versions are Closed.\n\n"
    + "Return JSON:\n"
    + "{\n"
    + '  "scores": {\n'
    + '    "1-2 Code": { "score": 0|0.5|1, "reason": "..." },\n'
    + '    "1-3 License": { "score": 0|0.5|1, "reason": "..." },\n'
    + '    ...\n'
    + '    "4-4 Data Filtering": { "score": 0|0.5|1, "reason": "..." }\n'
    + "  },\n"
    + '  "total_score": <number>\n'
    + "}\n"
).strip()

# ---- Methodology hint dictionaries (tight policy) ----
METHOD_HINTS = {
    "3-1 Pre-training": (
        "causal lm","next-token","masked lm","span corruption","autoregressive",
        "optimizer","adam","adamw","sgd","learning rate","lr ","warmup",
        "scheduler","cosine","polynomial","decay",
        "batch size","global batch","micro batch","grad accumulate",
        "steps","epochs","update steps",
        "context length","sequence length","seq len","max length",
        "deepspeed","fsdp","megatron","tensor parallel","pipeline parallel",
        "mixed precision","fp16","bf16"
    ),
    "3-2 Fine-tuning": (
        "supervised fine-tuning","sft","instruction tuning","lora","qlora",
        "peft","adapter","prefix-tuning","prompt tuning","orpo",
        "learning rate","optimizer","adam","adamw","warmup","scheduler",
        "epochs","steps","batch size","grad accumulate",
        "cross-entropy","label smoothing","loss function"
    ),
    "3-3 Reinforcement Learning": (
        "rlhf","rlaif","ppo","trpo","a2c","actor-critic","policy gradient",
        "reward model","preference model","kl penalty","beta","reference model",
        "dpo","ipo","kto"
    ),
}

def _has_method_hints(quotes: List[str], key: str) -> bool:
    hints = METHOD_HINTS.get(key, ())
    if not quotes or not hints:
        return False
    text = " \n".join(q.lower() for q in quotes)
    return any(h in text for h in hints)

# ─────────── API evidence detector (endpoint/key) ───────────
API_PATTERNS_POS = (
    "api key", "apikey", "authorization: bearer", "bearer ",
    "curl -x", "curl -h", "curl -d", "curl -H", "curl https://", "post https://", "get https://",
    "endpoint", "rest api", "openapi", "swagger", "api reference",
)
API_PATTERNS_NEG = (
    "python api", "torch api", "transformers api", "sdk", "library api",
)

def _collect_quotes_from(src: Dict[str, Any], keys: List[str]) -> List[str]:
    out: List[str] = []
    if not isinstance(src, dict):
        return out
    for key in keys:
        evs = src.get(key) or []
        if isinstance(evs, list):
            for e in evs:
                if isinstance(e, dict):
                    q = e.get("quote", "")
                    if isinstance(q, str) and q.strip():
                        out.append(q.strip())
    return out

def _detect_public_api_from_sources(hf: Dict, gh: Dict, ax: Dict, rp: Dict) -> bool:
    keys = ["2-3 (API)__evidence", "2-3 API__evidence"]
    quotes = []
    for src in (hf, gh, ax, rp):
        quotes.extend(_collect_quotes_from(src or {}, keys))
    txt = "\n".join(q.lower() for q in quotes)
    if not txt:
        return False
    if any(n in txt for n in API_PATTERNS_NEG) and not any(p in txt for p in API_PATTERNS_POS):
        return False
    return any(p in txt for p in API_PATTERNS_POS)

# ─────────── Pretrain merge helpers ───────────
def _merge_pretrain_parts(pretrain_parts: Dict[str, Dict[str, Any]] | None) -> Dict[str, Any]:
    """
    Normalize/merge pretrain_* dispatchers (hf/github/arxiv/reports) into a simple bundle:
    {
      "pretrain_method": "<concat summaries>",
      "pretrain_data":   "<concat summaries>",
      "__evidence":      ["quote", ...]   # may be generic; used to support 3-1/4-1
    }
    Also supports legacy structures where __evidence is a dict per label.
    """
    if not pretrain_parts:
        return {}
    texts_m, texts_d, quotes = [], [], []
    for k in ("hf", "github", "arxiv", "reports", "gh", "ax"):
        obj = (pretrain_parts.get(k) or {})
        pm = obj.get("pretrain_method") or ""
        pd = obj.get("pretrain_data") or ""
        ev = obj.get("__evidence") or []
        # string summaries
        if isinstance(pm, str) and pm.strip():
            texts_m.append(pm.strip())
        if isinstance(pd, str) and pd.strip():
            texts_d.append(pd.strip())
        # evidence could be list[str] or dict[label->list[dict]]
        if isinstance(ev, list):
            for q in ev:
                if isinstance(q, str) and q.strip():
                    quotes.append(q.strip())
        elif isinstance(ev, dict):
            for v in ev.values():
                if isinstance(v, list):
                    for e in v:
                        if isinstance(e, dict):
                            q = (e.get("quote") or "").strip()
                            if q:
                                quotes.append(q)
    if not (texts_m or texts_d or quotes):
        return {}
    return {
        "pretrain_method": ("\n\n".join(texts_m)).strip(),
        "pretrain_data":   ("\n\n".join(texts_d)).strip(),
        "__evidence":      quotes
    }

# ─────────── Utilities ───────────
def _auto_open_items(hf_json: Dict[str, Any]) -> Dict[str, Dict]:
    if not hf_json:
        return {}
    return {
        "1-1 Weights":     {"score": 1, "reason": "Weights hosted on Hugging Face or equivalent"},
        "1-5 Architecture": {"score": 1, "reason": "Architecture disclosed on the model card/config"},
        "1-6 Tokenizer":   {"score": 1, "reason": "Tokenizer details available on the model card/config"},
    }

def _collect_evidence_maps(ax: Dict, hf: Dict, gh: Dict,
                           rp: Dict | None = None,
                           pretrain_bundle: Dict | None = None) -> Dict[str, Dict[str, List[str]]]:
    out: Dict[str, Dict[str, List[str]]] = {}
    for arx_lbl, score_lbl in LBL_MAP_ARX_TO_SCORE.items():
        quotes: List[str] = []
        # arXiv/reports/hf/gh evidence arrays
        evs = (ax or {}).get(f"{arx_lbl}__evidence") or []
        if isinstance(evs, list):
            quotes.extend([e.get("quote","") for e in evs if isinstance(e, dict) and e.get("quote")])
        for src in (hf, gh):
            if isinstance(src, dict):
                for k in (f"{arx_lbl}__evidence", f"{score_lbl}__evidence"):
                    evs2 = src.get(k) or []
                    if isinstance(evs2, list):
                        quotes.extend([e.get("quote","") for e in evs2 if isinstance(e, dict) and e.get("quote")])
        if isinstance(rp, dict):
            for k in (f"{arx_lbl}__evidence", f"{score_lbl}__evidence"):
                evs3 = rp.get(k) or []
                if isinstance(evs3, list):
                    quotes.extend([e.get("quote","") for e in evs3 if isinstance(e, dict) and e.get("quote")])

        # 🔗 inject pretrain-bundle quotes for 3-1 & 4-1
        if pretrain_bundle and isinstance(pretrain_bundle.get("__evidence"), list):
            if score_lbl in ("3-1 Pre-training", "4-1 Pre-training Data"):
                quotes.extend(pretrain_bundle.get("__evidence", []))

        quotes = [q for q in quotes if isinstance(q, str) and q.strip()]
        # optional: attach pretrain summaries in the "summary" (context only)
        summ = (ax or {}).get(arx_lbl, "") or ""
        if pretrain_bundle:
            if score_lbl == "3-1 Pre-training" and pretrain_bundle.get("pretrain_method"):
                summ = (summ + "\n\n" + pretrain_bundle["pretrain_method"]).strip()
            if score_lbl == "4-1 Pre-training Data" and pretrain_bundle.get("pretrain_data"):
                summ = (summ + "\n\n" + pretrain_bundle["pretrain_data"]).strip()

        out[score_lbl] = {"quotes": quotes, "summary": summ}
    return out

def _aggregate_usage(ax: dict, hf: dict, gh: dict, rp: dict | None = None) -> Dict[str,str]:
    def norm(x): return x if x in {"used","not_used"} else "unknown"
    vals = []
    for src in (ax, hf, gh, rp):
        u = (src or {}).get("__usage") or {}
        vals.append({"ft":norm(u.get("fine_tuning")), "rl":norm(u.get("rl"))})
    def decide(key):
        vs = [v[key] for v in vals if v[key]!="unknown"]
        if not vs: return "unknown"
        if "used" in vs: return "used"
        return "not_used"
    return {"fine_tuning": decide("ft"), "rl": decide("rl")}

# ─────────── Code openness detection (training vs inference) ───────────
_TRAIN_FILE_HINTS = (
    "train.py","training.py","pretrain","pre-train","pretraining",
    "run_clm.py","run_squad.py","run_translation.py","run_sft.py",
    "scripts/train","examples/train","training/","trainer","finetune.py",
    "fine_tune.py","sft","dpo","ppo","rlhf","accelerate launch","deepspeed","torchrun"
)
_PARTIAL_ONLY_HINTS = (
    "finetune","fine-tune","lora","qlora","peft","adapter","sft","dpo","ppo","rlhf"
)
_INFER_ONLY_HINTS = (
    "inference","infer","generate.py","generation.py","server","api","gradio","demo","app.py","serve","endpoint"
)

def _detect_code_openness(hf_json: Dict[str, Any]) -> Tuple[float, str]:
    if not hf_json:
        return 0.0, "No Hugging Face metadata"
    files = [f for f in (hf_json.get("files") or []) if isinstance(f, str)]
    py_files = list((hf_json.get("py_files") or {}).keys())
    readme = (hf_json.get("readme") or "").lower()
    joined = " ".join(files + py_files).lower() + " " + readme

    has_train = any(h in joined for h in _TRAIN_FILE_HINTS)
    has_partial = any(h in joined for h in _PARTIAL_ONLY_HINTS)
    has_infer_only = any(h in joined for h in _INFER_ONLY_HINTS)

    if has_train:
        return 1.0, "Training pipeline code detected (files/README contain training scripts or instructions)."
    if has_partial:
        return 0.5, "Only partial training code detected (fine-tune/adapter/RL scripts) without full pipeline."
    if py_files or files:
        if has_infer_only:
            return 0.0, "Only inference/serving/evaluation code detected; no training pipeline."
        return 0.0, "Repository lacks explicit training code hints; treated as Closed."
    return 0.0, "No code files."

# ─────────── GPT evaluation for strict items ───────────
def _gpt_evaluate(model: str,
                  hf: Dict, gh: Dict, ax: Dict,
                  evidence_map: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict]:
    payload = {
        "model": model,
        "evidence": evidence_map,               # quotes for strict items
        "data": {"huggingface": hf, "github": gh, "arxiv": ax},
    }
    rsp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL_EVALUATOR", "o3-mini"),
        reasoning_effort="medium",
        response_format={"type":"json_object"},
        messages=[
            {"role":"system","content":EVALUATION_PROMPT},
            {"role":"user",  "content":json.dumps(payload, ensure_ascii=False)}
        ]
    )
    try:
        raw = json.loads(rsp.choices[0].message.content.strip())
    except Exception:
        raw = {"scores": {}}
    scores_dict = raw.get("scores", raw)
    out: Dict[str, Dict] = {}
    for k, v in scores_dict.items():
        if isinstance(v, dict):
            out[k] = {"score": float(v.get("score", 0)), "reason": v.get("reason","")}
        elif isinstance(v, (int, float)):
            out[k] = {"score": float(v), "reason": ""}
    return out

# ─────────── Main evaluation ───────────
def evaluate_openness(model_name: str,
                      hf_json=None, gh_json=None, arxiv_json=None, reports_json=None, pretrain_parts=None) -> Dict:
    hf, gh, ax = hf_json or {}, gh_json or {}, arxiv_json or {}
    rp = reports_json or {}

    # 0) Merge pretrain sources (optional)
    pre_bundle = _merge_pretrain_parts(pretrain_parts)

    # 1) Strict evidence map for 3-x / 4-x (+inject pretrain bundle)
    evmap = _collect_evidence_maps(ax, hf, gh, rp, pretrain_bundle=pre_bundle)

    # 2) Strict GPT scoring for the rest
    scores = _gpt_evaluate(model_name, hf, gh, ax, evmap)

    # 3) Auto-open (1-1 / 1-5 / 1-6)
    scores.update(_auto_open_items(hf))

    # 4) Code (1-2) heuristic overwrite (training-centric)
    code_score, code_reason = _detect_code_openness(hf)
    scores["1-2 Code"] = {"score": code_score, "reason": code_reason}

    # 5) Usage aggregation → exclusion base
    usage = _aggregate_usage(ax, hf, gh, rp)

    # Heuristic: "you can fine-tune" 류 문구만 있고 저자 수행 근거 없으면 not_used
    def _has_phrase(src_dict: Dict[str, Any], *phrases: str) -> bool:
        if not isinstance(src_dict, dict):
            return False
        txt = ""
        for v in src_dict.values():
            if isinstance(v, str):
                txt += " " + v.lower()
        return any(p in txt for p in phrases)

    if usage.get("fine_tuning","unknown") == "unknown":
        has_generic = any(_has_phrase(s,
                           "you can fine-tune", "you may fine-tune", "to fine-tune the model",
                           "fine-tune this model") for s in (hf, gh, ax, rp))
        has_auth_run = any(_has_phrase(s, "we fine-tuned", "we finetuned", "we instruction-tuned",
                                       "we performed sft", "we applied lora", "we applied qlora")
                           for s in (hf, gh, ax, rp))
        if has_generic and not has_auth_run:
            usage["fine_tuning"] = "not_used"

    exclude_prefix = []
    if usage["fine_tuning"] == "not_used":
        exclude_prefix += ["3-2", "4-2"]
    if usage["rl"] == "not_used":
        exclude_prefix += ["3-3", "4-3"]

    # 6) STRICT guardrail: if no quotes for strict keys → force CLOSED
    for strict_key in STRICT_KEYS:
        if strict_key not in scores:
            continue
        has_quotes = bool(evmap.get(strict_key, {}).get("quotes"))
        if not has_quotes:
            reason = scores[strict_key].get("reason","").strip()
            note = "No direct quote evidence; defaulting to Closed by strict rule."
            scores[strict_key] = {"score": 0.0, "reason": (reason + " " + note).strip()}

    # 6.1) 4-1 Pre-training Data: if quotes exist → minimum 0.5 (partial disclosure)
    key_41 = "4-1 Pre-training Data"
    if key_41 in scores:
        has_quotes_41 = bool(evmap.get(key_41, {}).get("quotes"))
        if has_quotes_41 and float(scores.get(key_41, {}).get("score", 0)) < 0.5:
            scores[key_41] = {
                "score": 0.5,
                "reason": (scores.get(key_41, {}).get("reason","") + " Adjusted to Semi-Open: quotes indicate partial disclosure.").strip()
            }

    # 6.2) Tight methodology guardrail:
    for meth_key in ("3-1 Pre-training", "3-2 Fine-tuning", "3-3 Reinforcement Learning"):
        used_flag = {
            "3-1 Pre-training": "used",  # pretraining은 모델이 존재하면 사용된 것으로 간주
            "3-2 Fine-tuning": usage.get("fine_tuning", "unknown"),
            "3-3 Reinforcement Learning": usage.get("rl", "unknown"),
        }
        if used_flag[meth_key] == "not_used":
            continue
        qts = evmap.get(meth_key, {}).get("quotes", [])
        if not _has_method_hints(qts, meth_key):
            label_nice = {"3-1 Pre-training":"pre-training",
                          "3-2 Fine-tuning":"fine-tuning",
                          "3-3 Reinforcement Learning":"reinforcement learning"}[meth_key]
            scores[meth_key] = {
                "score": 0.0,
                "reason": (
                    f"No concrete {label_nice} method details (objectives/schedules/hyperparameters/pipeline). "
                    "Only data scale/types/hardware were mentioned → Closed (0) by tight policy."
                )
            }

    # 6.3) 2-3 API: no endpoint/key evidence → force 0
    if "2-3 API" in scores:
        if not _detect_public_api_from_sources(hf, gh, ax, rp):
            scores["2-3 API"] = {"score": 0.0, "reason": "No concrete public API evidence (no endpoint/key/example)."}

    # 7) Code=Open ⇒ methodology(3-1/3-2/3-3) auto-Open (unless excluded by usage)
    if scores.get("1-2 Code", {}).get("score", 0) >= 1.0:
        for prefix in ["3-1", "3-2", "3-3"]:
            if any(prefix == ex for ex in exclude_prefix):
                continue
            for key in list(scores.keys()):
                if key.startswith(prefix):
                    scores[key] = {"score": 1.0, "reason": "Full training code published; methods reproducible (auto-escalation)."}
                    break

    # 8) Build included set after exclusion
    included = {k:v for k,v in scores.items()
                if not any(k.startswith(p) for p in exclude_prefix)}

    # 9) Aggregate to 10-point scale
    raw_sum = sum(float(v.get("score",0)) for v in included.values())
    denom  = max(len(included), 1)
    final_10 = round(raw_sum * (10.0 / denom), 3)

    meta_obj = {
        "usage_from_dispatch": usage,
        "excluded": [k for k in scores if k not in included],
        "denominator": denom,
        "raw_sum": raw_sum,
        "scale": f"10/{denom}",
        "code_detection_reason": code_reason,
        "pretrain_sources_used": bool(pre_bundle)
    }
    return {
        "model": model_name,
        "scores": scores,
        "included_scores": included,
        "final_score_10pt": final_10,
        "meta": meta_obj
    }

# ─────────── File loader & CLI ───────────
def evaluate_openness_from_files(model_name: str,
                                 base_dir: str | Path = ".",
                                 base_model_id: str | None = None):
    base = model_name.replace("/", "_").lower()
    base_dir = Path(base_dir)

    def _load_from_base(filename: str):
        p = base_dir / filename
        if p.exists() and p.stat().st_size:
            try:
                return json.load(open(p, encoding="utf-8"))
            except json.JSONDecodeError:
                print("⚠️ JSON parsing failed:", p)
        if os.path.exists(filename) and os.path.getsize(filename):
            try:
                return json.load(open(filename, encoding="utf-8"))
            except json.JSONDecodeError:
                print("⚠️ JSON parsing failed:", filename)
        return {}

    hf = _load_from_base(f"huggingface_filtered_final_{base}.json")
    gh = _load_from_base(f"github_filtered_final_{base}.json")
    ax = _load_from_base(f"arxiv_filtered_final_{base}.json")
    rp = _load_from_base(f"reports_filtered_final_{base}.json")

    # Load pretrain bundles if base_model_id is present
    pretrain = {"hf": {}, "github": {}, "arxiv": {}, "reports": {}}
    if base_model_id:
        pbase = base_model_id.replace("/", "_").lower()
        # support both naming schemes
        pretrain["hf"]      = _load_from_base(f"pretrain_hf_{pbase}.json") or _load_from_base(f"pretrain_huggingface_{pbase}.json")
        pretrain["github"]  = _load_from_base(f"pretrain_gh_{pbase}.json") or _load_from_base(f"pretrain_github_{pbase}.json")
        pretrain["arxiv"]   = _load_from_base(f"pretrain_arxiv_{pbase}.json")
        pretrain["reports"] = _load_from_base(f"pretrain_reports_{pbase}.json")

    res = evaluate_openness(model_name, hf_json=hf, gh_json=gh, arxiv_json=ax,
                            reports_json=rp, pretrain_parts=pretrain)
    out = base_dir / f"openness_score_{base}.json"
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("📝 Saved evaluation result:", out)
    return res


if __name__ == "__main__":
    evaluate_openness_from_files("bigscience/bloomz-560m")
