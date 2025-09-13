# openness_Evaluator.py — STRICT + Code-as-Training transparency + Pretrain sources merge + API web-search boost
# (Update)
# - License (1-3) and Data Filtering (4-4): GPT-only scoring (no rule-based overrides)
# - API (2-3): Owner official domain OR reputable third-party aggregators count as Open
import os, json, re
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
from urllib.parse import urlparse

load_dotenv()
_API_KEY = os.getenv("OPENAI_API_KEY")
if not _API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
client = OpenAI(api_key=_API_KEY)

# ─────────── Web Search (Preview) config — used ONLY for 2-3 API ───────────
USE_OPENAI_WEB_SEARCH = os.getenv("USE_OPENAI_WEB_SEARCH", "1") == "1"
OPENAI_MODEL_WEB      = os.getenv("OPENAI_MODEL_WEB", "gpt-4.1-mini")                # Responses API + web_search tool
CHAT_SEARCH_MODEL     = os.getenv("CHAT_SEARCH_MODEL", "gpt-4o-mini-search-preview")  # Chat Completions search-preview
WEB_SEARCH_PROVIDER   = os.getenv("WEB_SEARCH_PROVIDER", "auto").lower()              # auto|responses|chat
API_WEBSEARCH_MAX_URLS = int(os.getenv("API_WEBSEARCH_MAX_URLS", "6"))

# URL regex (shared)
_URL_RE = re.compile(r"https?://[^\s)>\]\"'}]+", re.I)

def _is_apiish_url(u: str) -> bool:
    """Loose filter for API-doc/endpoint-ish URLs."""
    if not u:
        return False
    lu = u.lower()
    if any(x in lu for x in ("swagger", "openapi", "readme.io", "/v1/", "/v2/")):
        return True
    if "api." in lu or "/api" in lu:
        return True
    if any(x in lu for x in ("docs.", "/docs", "developer.", "/developers", "/reference")):
        return True
    # Allow GH only if docs-ish path
    if "github.com" in lu and any(x in lu for x in ("openapi", "swagger", "api.md", "/docs", "reference")):
        return True
    return False

# ─────────── Official-or-3rd-party heuristics (generic; no per-model hardcoding) ───────────
# Reputable third-party aggregators that we accept as "Open" when they expose real HTTP endpoints/doc refs
_REPUTABLE_THIRDPARTY_HOSTS = (
    "openrouter.ai", "openrouter", "together.ai", "togethercomputer",
    "replicate.com", "replicate.dev", "perplexity.ai", "fireworks.ai",
    "huggingface.co", "api-inference.huggingface.co", "hf.space"
)

_GENERIC_HOST_STOP = {"www", "api", "apis", "docs", "developer", "developers", "dev", "ref", "reference"}

def _extract_host(u: str) -> str:
    try:
        return (urlparse(u).hostname or "").lower()
    except Exception:
        return ""

def _org_tokens_from_model_name(model_name: str) -> List[str]:
    """
    Derive organization tokens from the part before '/' in model_name.
    e.g., 'meta-llama/Llama-3-8B' → ['meta', 'llama', 'metallama']
    Works generically; no per-model hardcoding.
    """
    org = (model_name or "").split("/", 1)[0].lower()
    if not org: return []
    bits = re.split(r"[^a-z0-9]+", org)
    toks = set()
    for b in bits:
        b = b.strip()
        if not b: continue
        if b in {"ai", "ml", "research", "labs", "lab", "team", "inc", "llc", "foundation", "org", "company"}:
            continue
        if len(b) >= 3:
            toks.add(b)
    # joined variant
    comp = re.sub(r"[^a-z0-9]+", "", org)
    if len(comp) >= 5: toks.add(comp)
    return sorted(toks)

def _host_is_reputable_third_party(host: str) -> bool:
    h = (host or "").lower()
    return any(tp in h for tp in _REPUTABLE_THIRDPARTY_HOSTS)

def _host_contains_any_token(host: str, tokens: List[str]) -> bool:
    if not host or not tokens: return False
    parts = [p for p in host.split(".") if p and p not in _GENERIC_HOST_STOP]
    hflat = ".".join(parts)
    return any(t for t in tokens if t in hflat)

def _is_official_or_thirdparty_api_url(u: str, org_tokens: List[str]) -> bool:
    """
    URL counts if:
      - It looks API-ish, AND
      - (host contains one of the org tokens) OR (host is a reputable third-party aggregator).
    """
    if not _is_apiish_url(u):
        return False
    host = _extract_host(u)
    if not host:
        return False
    if _host_is_reputable_third_party(host):
        return True
    return _host_contains_any_token(host, org_tokens)

# ─────────── Web search helpers ───────────
def _websearch_api_urls_via_responses(model_name: str, org_tokens: List[str], limit: int) -> List[str]:
    """Responses API + web_search tool."""
    token_hint = ", ".join(org_tokens[:3]) if org_tokens else ""
    prompt = (
        f'Return up to {limit} API documentation URLs for "{model_name}".\n'
        f'- Prefer URLs hosted on the model owner’s **official domain** (domain contains one of: {token_hint}); list these first.\n'
        f'- **Also include reputable third-party aggregators** (Hugging Face Inference, OpenRouter, Together, Replicate, Fireworks, Perplexity) when they expose HTTP endpoints/OpenAPI/Swagger or developer API reference for this model.\n'
        f'- Exclude SDK-only pages and community posts.\n'
        f'- Output raw URLs, one per line.'
    ).strip()
    try:
        rsp = client.responses.create(
            model=OPENAI_MODEL_WEB,
            tools=[{"type": "web_search"}],
            tool_choice="auto",
            input=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        text = getattr(rsp, "output_text", "") or str(rsp)
        urls = [u.strip() for u in _URL_RE.findall(text)]
        urls = [u for u in urls if _is_official_or_thirdparty_api_url(u, org_tokens)]
        seen, out = set(), []
        for u in urls:
            if u not in seen:
                seen.add(u); out.append(u)
            if len(out) >= limit:
                break
        return out
    except Exception:
        return []

def _websearch_api_urls_via_chat(model_name: str, org_tokens: List[str], limit: int) -> List[str]:
    """Chat Completions + search-preview model."""
    token_hint = ", ".join(org_tokens[:3]) if org_tokens else ""
    try:
        res = client.chat.completions.create(
            model=CHAT_SEARCH_MODEL,
            messages=[{"role": "user",
                       "content": (
                            f'Return up to {limit} API documentation URLs for "{model_name}".\n'
                            f'- Prefer owner’s official domain (contains: {token_hint}); list first.\n'
                            f'- **Also include reputable third-party aggregators** exposing real endpoints/docs.\n'
                            f'- Exclude SDK-only pages and community posts.\n'
                            f'- Output raw URLs, one per line.'
                       )}],
            temperature=0.2,
        )
        text = (res.choices[0].message.content or "")
        urls = [u.strip() for u in _URL_RE.findall(text)]
        urls = [u for u in urls if _is_official_or_thirdparty_api_url(u, org_tokens)]
        seen, out = set(), []
        for u in urls:
            if u not in seen:
                seen.add(u); out.append(u)
            if len(out) >= limit:
                break
        return out
    except Exception:
        return []

def _websearch_api_urls(model_name: str, org_tokens: List[str], limit: int = API_WEBSEARCH_MAX_URLS) -> List[str]:
    """Pick a provider (auto/responses/chat) to fetch API-ish URLs."""
    if not USE_OPENAI_WEB_SEARCH:
        return []
    providers = (["responses", "chat"] if WEB_SEARCH_PROVIDER == "auto" else [WEB_SEARCH_PROVIDER])
    for p in providers:
        urls = _websearch_api_urls_via_responses(model_name, org_tokens, limit) if p == "responses" \
               else _websearch_api_urls_via_chat(model_name, org_tokens, limit)
        if urls:
            return urls
    return []

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

ALL_SCORE_KEYS = [
    "1-1 Weights", "1-2 Code", "1-3 License", "1-4 Paper",
    "1-5 Architecture", "1-6 Tokenizer",
    "2-1 Hardware", "2-2 Software", "2-3 API",
    "3-1 Pre-training", "3-2 Fine-tuning", "3-3 Reinforcement Learning",
    "4-1 Pre-training Data", "4-2 Fine-tuning Data",
    "4-3 Reinforcement Learning Data", "4-4 Data Filtering",
]

# ─────────── Evaluation rubric text (STRICT) ───────────
CRITERIA_TEXT = """
## 1. Model Basic Openness – 6 items
### 1-1. Weights
- Open (1): Model weights are publicly available without permission
- Semi-Open (0.5): Model weights are available after approval
- Closed (0): Model weights are not available for use

### 1-2. Code (TRAINING vs inference/serving)
- Open (1): End-to-end TRAINING code is public and sufficient to reproduce training (data prep, configs, scripts, schedules) **and actual training pipeline files exist** (e.g., train.py, training/, scripts/train, pretrain*.py, run_*.py).
- Semi-Open (0.5): Some TRAINING code is public (e.g., SFT/LoRA/QLoRA, RL scripts) but not a full pipeline.
- Closed (0): Only inference/serving/eval code or none.

### 1-3. License
- Open (1): License explicitly allows **all four** rights — use, modification, redistribution, and commercial use — with no major restrictions (MIT/Apache-2.0/BSD/MPL/Qwen ).
- Semi-Open (0.5): **One or two** rights restricted (non-commercial / no-redistribution / no-derivatives / research-only, OpenRAIL family, CC-BY-NC).
    **If the license has restrictions but they are so lenient that they don’t apply to ordinary users (e.g., only targeting large corporate competitors), then it should still be considered “Open” (like in the case of Qwen).
- Closed (0): **Three or more** rights restricted, or no license / undefined restrictive terms.

### 1-4. Paper
- Open (1): Peer-reviewed or official tech report **specifically about the TARGET model** (exact version/variant).
- Semi-Open (0.5): Model blog/announcement/model card with results but not a full tech report.
- Closed (0): No doc for this model, or about other model/version.

### 1-5. Architecture
- Open (1): Structure and hyperparameters fully disclosed
- Semi-Open (0.5): Partial disclosure
- Closed (0): Not disclosed

### 1-6. Tokenizer
- Open (1): Tokenizer details disclosed and/or downloadable
- Semi-Open (0.5): Partial disclosure
- Closed (0): Not disclosed

## 2. Accessibility and Reproducibility – 3 items
### 2-1. Hardware  (QUOTES REQUIRED)
- Open (1): Quotes disclose **both** training hardware type(s) **and** quantities.
- Semi-Open (0.5): Only one side (type or quantity/compute) or partial quantities.
- Closed (0): No quoted info about training hardware.

### 2-2. Software  (QUOTES REQUIRED; TRAINING stack only)
- Open (1): Quotes disclose the **full training software stack** used to train the model, typically including: base framework (with version), distributed/orchestration (e.g., DeepSpeed, Megatron, FSDP, NeMo, XLA/TPU), precision/optimizer libs (Apex, bitsandbytes, FlashAttention, etc.), data pipeline tools, and **versions/configs** sufficient to reconstruct the stack.
- Semi-Open (0.5): Quotes disclose **some training components beyond the base framework**. 
- Closed (0): a framework that is not available to general users or is not open code. Only generic framework name, or inference/serving stack only, or no quoted info (ex HAI-LLM,torch/jax/transfomer/python)

### 2-3. API — **Official OR reputable third-party**
- Open (1): Public API provided by the model owner **or** a reputable third-party aggregator (OpenRouter, Together, Replicate, Fireworks, Perplexity, Hugging Face Inference), with developer/docs URLs.
- Semi-Open (0.5): The owner has officially announced a “coming soon” public API, or only a waitlist/approval application is available (developers cannot obtain a key yet). Alternatively, documentation exists but the credential issuance path is unclear (e.g., “Contact us only,” partners-only).
- Closed (0): No public API, or no way to obtain credentials is provided at all (internal-only, partners by private contract, UI embed only with no developer endpoints/keys, etc.).
## 3. Training Methodology – STRICT (quotes required)
### 3-1. Pre-training
- Open (1): Fully reproducible method (pipeline, objectives, schedules, all hyperparameters)
- Semi-Open (0.5): Partial method disclosure
- Closed (0): No method disclosed (default if no quotes)

### 3-2. Fine-tuning
- Open (1): Fully reproducible fine-tuning details
- Semi-Open (0.5): Partial details
- Closed (0): Not disclosed/N/A (default if no quotes, )

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
- Open (1):Discusses large-scale data collection or types of filtering in the paper or Discloses concrete filtering/cleaning pipeline and impact (e.g. dedup/minhash/simhash, toxicity/NSFW filters or moderation models, language-ID, PII removal, quality/perplexity thresholds, multi-stage pipeline, removal ratios,llama guard).
- Semi-Open (0.5): Mentions of filtering/cleaning exist but lack full details (generic “we filtered/cleaned/removed…”, or single category without thresholds/ratios/pipeline) .
- Closed (0): No info (default if no quotes)
""".strip()

EVALUATION_PROMPT = (
    CRITERIA_TEXT
    + "\n\nHARD RULES:\n"
    + "1) For items **2-1 Hardware, 2-2 Software, 3-1~3-3, and 4-1~4-4**, base decisions **ONLY on provided quotes**.\n"
    + "   · If there is NO quote evidence for these items, the item is **CLOSED (0)**.\n"
    + "   · For 2-2 Software, ignore bare mentions of base frameworks only (e.g., just “PyTorch/JAX”) but the framework that is not available to general users or is not open code case is closed  — Closed unless training components beyond the base framework are quoted.\n"
    + "   · Only statements about the **TRAINING** stack count for 2-2; inference/serving environments do **not** count.\n"
    + "2) SEMI-OPEN (0.5) requires some direct evidence; OPEN (1) requires fully reproducible methods/stack or fully disclosed datasets/pipelines per the rubric.\n"
    + "3) You may use auxiliary JSON for context, but NEVER override rule (1). Do **not** fill in facts from memory or general knowledge.\n"
    + "4) Do NOT treat data scale/types/hardware alone as methodology disclosure. Without concrete method details, score **Closed (0)** for 3-1/3-2/3-3.\n"
    + "5) For **1-4 Paper**, apply the rubric only to documents about the TARGET model (payload.model). Blogs/model cards are Semi-Open; papers/tech reports about other models or older versions are Closed.\n"
    + "6) For **2-3 API**, treat as **Open** if the owner hosts a public API **or** a reputable third-party aggregator exposes real endpoints/docs for this model; **Semi-Open** if only an official 'coming soon' announcement exists; otherwise **Closed**.\n\n"
    + "7) For **1-2 Code**, Open requires actual training pipeline files (e.g., train.py, training/, scripts/train, pretrain*.py, run_*.py). Partial code (e.g., SFT/LoRA/QLoRA scripts) is Semi-Open.\n\n"
    + "8) if fine_tuning: unknown case ,Old models like GPT-2 only have a pre-training. so fineReinforcement = N/A."
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

# ---- Methodology hint dictionaries (more permissive for Semi-Open) ----
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

# --- Method categories (for detecting truly reproducible OPEN cases) ---
_METHOD_CATS = {
    "objective": ("causal lm","next-token","masked lm","span corruption","autoregressive"),
    "optimizer": ("adam","adamw","sgd","adafactor"),
    "schedule":  ("warmup","scheduler","cosine","polynomial","linear decay","decay"),
    "batching":  ("batch size","global batch","micro batch","grad accumulate"),
    "duration":  ("steps","epochs","update steps"),
    "context":   ("context length","sequence length","seq len","max length"),
    "dist":      ("deepspeed","fsdp","megatron","tensor parallel","pipeline parallel"),
    "precision": ("mixed precision","fp16","bf16")
}

def _method_category_hits(quotes: List[str]) -> set:
    txt = " \n".join(q.lower() for q in quotes)
    hits = set()
    for cat, kws in _METHOD_CATS.items():
        if any(k in txt for k in kws):
            hits.add(cat)
    return hits

def _lenient_method_score_from_quotes(quotes: List[str], key: str) -> float:
    """
    Permissive rules:
      - Open (1.0): Reproducible — objective + (optimizer or schedule) + (batching or duration), and ≥4 categories in total.
      - Semi-Open (0.5): Any concrete method hints for that family (pretrain / finetune / RL).
      - Closed (0.0): No method info in quotes.
    """
    if not quotes:
        return 0.0
    cats = _method_category_hits(quotes)
    if len(cats) >= 4 and ("objective" in cats) and (("optimizer" in cats) or ("schedule" in cats)) and (("batching" in cats) or ("duration" in cats)):
        return 1.0
    if _has_method_hints(quotes, key) or len(cats) >= 1:
        return 0.5
    return 0.0

# ─────────── Shared quote collectors ───────────
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

def _collect_all_quotes_for_item(hf: Dict, gh: Dict, ax: Dict, rp: Dict, labels: List[str]) -> List[str]:
    quotes: List[str] = []
    for src in (hf or {}, gh or {}, ax or {}, rp or {}):
        quotes.extend(_collect_quotes_from(src, labels))
    return [q for q in quotes if isinstance(q, str) and q.strip()]

# ─────────── Pretrain merge helpers ───────────
def _merge_pretrain_parts(pretrain_parts: Dict[str, Dict[str, Any]] | None) -> Dict[str, Any]:
    if not pretrain_parts:
        return {}
    texts_m, texts_d, quotes = [], [], []
    for k in ("hf", "github", "arxiv", "reports", "gh", "ax"):
        obj = (pretrain_parts.get(k) or {})
        pm = obj.get("pretrain_method") or ""
        pd = obj.get("pretrain_data")   or ""
        ev = obj.get("__evidence")      or []
        if isinstance(pm, str) and pm.strip(): texts_m.append(pm.strip())
        if isinstance(pd, str) and pd.strip(): texts_d.append(pd.strip())
        if isinstance(ev, list):
            for q in ev:
                if isinstance(q, str) and q.strip(): quotes.append(q.strip())
        elif isinstance(ev, dict):
            for v in ev.values():
                if isinstance(v, list):
                    for e in v:
                        if isinstance(e, dict):
                            q = (e.get("quote") or "").strip()
                            if q: quotes.append(q)
    if not (texts_m or texts_d or quotes):
        return {}
    return {
        "pretrain_method": ("\n\n".join(texts_m)).strip(),
        "pretrain_data":   ("\n\n".join(texts_d)).strip(),
        "__evidence":      quotes
    }

# ─────────── Utilities ───────────
def _auto_open_items(hf_json: Dict[str, Any], hf_raw: Dict[str, Any] | None = None) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}

    # 1-1 Weights
    if hf_raw:
        files = [f for f in (hf_raw.get("files") or []) if isinstance(f, str)]
        if any(f.lower().endswith((".safetensors", ".bin", ".pt", ".ckpt")) for f in files):
            out["1-1 Weights"] = {"score": 1, "reason": "Weights files present in the repository (e.g., *.safetensors/bin/pt/ckpt)."}

    # 1-5 Architecture
    if isinstance(hf_json, dict) and hf_json.get("1-5 (Architecture)__evidence"):
        out["1-5 Architecture"] = {"score": 1, "reason": "Architecture evidence present in extracted quotes."}

    # 1-6 Tokenizer
    tokenizer_score = None
    tokenizer_reason = ""
    if hf_raw:
        files = [f.lower() for f in (hf_raw.get("files") or []) if isinstance(f, str)]
        tok_file_patterns = (
            "tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt",
            "sentencepiece.model", "spiece.model", "spm.model", "bpe.codes"
        )
        if any(any(pat in f for pat in tok_file_patterns) for f in files):
            tokenizer_score = 1.0
            tokenizer_reason = "Tokenizer files downloadable in repository."
    if tokenizer_score is None and isinstance(hf_json, dict) and hf_json.get("1-6 (Tokenizer)__evidence"):
        tokenizer_score = 0.5
        tokenizer_reason = "Tokenizer details disclosed in documentation, but no tokenizer files detected."
    if tokenizer_score is not None:
        out["1-6 Tokenizer"] = {"score": tokenizer_score, "reason": tokenizer_reason}

    return out

def _collect_evidence_maps(ax: Dict, hf: Dict, gh: Dict,
                           rp: Dict | None = None,
                           pretrain_bundle: Dict | None = None) -> Dict[str, Dict[str, List[str]]]:
    out: Dict[str, Dict[str, List[str]]] = {}
    for arx_lbl, score_lbl in LBL_MAP_ARX_TO_SCORE.items():
        quotes: List[str] = []
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

        # Inject pretrain bundle quotes for 3-1 & 4-1
        if pretrain_bundle and isinstance(pretrain_bundle.get("__evidence"), list):
            if score_lbl in ("3-1 Pre-training", "4-1 Pre-training Data"):
                quotes.extend(pretrain_bundle.get("__evidence", []))

        quotes = [q for q in quotes if isinstance(q, str) and q.strip()]
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

# ─────────── Code openness detection (strict file-based) ───────────
def _detect_code_openness(hf_filtered: Dict[str, Any] | None,
                          hf_raw: Dict[str, Any] | None) -> Tuple[float, str]:
    """
    Open(1.0): Training pipeline files exist in repo (train.py / training/ / scripts/train / pretrain*.py / run_*.py).
    Semi(0.5): Only partial training scripts exist (finetune*.py, *sft*.py, *lora*.py, *dpo*.py, *ppo*.py, *rlhf*.py).
    Closed(0): Others (including README-only mentions).

    If OP_EVAL_CODE_USE_FILTERED_ONLY=1 → rely on dispatcher's 1-2 (Code)__evidence when files missing.
    """
    import re
    hf_filtered = hf_filtered or {}
    hf_raw = hf_raw or {}

    files = [f for f in (hf_raw.get("files") or []) if isinstance(f, str)]
    py_files = list((hf_raw.get("py_files") or {}).keys())
    paths = " ".join(files + py_files).lower()

    TRAIN_PAT = re.compile(r'(^|/|\\)(train\.py|pretrain[^/\\]*\.py|run_[a-z0-9_]+\.py)(\b|$)')
    has_train_dir = ("training/" in paths) or ("/scripts/train" in paths) or ("\\scripts\\train" in paths)
    has_train_files = bool(TRAIN_PAT.search(paths)) or has_train_dir
    if has_train_files:
        return 1.0, "Training pipeline files exist in repository (train.py / training/ / scripts/train / pretrain*.py / run_*.py)."

    PARTIAL_PAT = re.compile(
        r'(^|/|\\)(finetune[^/\\]*\.py|fine[_-]?tune[^/\\]*\.py|[^/\\]*sft[^/\\]*\.py|[^/\\]*(lora|qlora)[^/\\]*\.py|[^/\\]*(dpo|ppo|rlhf)[^/\\]*\.py)(\b|$)'
    )
    has_partial_files = bool(PARTIAL_PAT.search(paths))

    if os.getenv("OP_EVAL_CODE_USE_FILTERED_ONLY", "0") == "1":
        ev = (hf_filtered.get("1-2 (Code)__evidence")
              or hf_filtered.get("1-2 Code__evidence")
              or [])
        if has_train_files:
            return 1.0, "Training pipeline files exist in repository."
        if not ev:
            return 0.0, "Dispatcher found no training-code evidence; treat as Closed."
        return 0.5, "Dispatcher provided training-code evidence, but no full pipeline files detected."

    if has_partial_files:
        return 0.5, "Partial training scripts exist as files but no full pipeline."
    return 0.0, "No training pipeline files; README mentions are ignored."

# ─────────── GPT evaluation for strict/non-strict items ───────────
def _gpt_evaluate(model: str,
                  hf: Dict, gh: Dict, ax: Dict,
                  evidence_map: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict]:
    """
    Default evaluator model: o3 (set OPENAI_MODEL_EVALUATOR=o3 to be explicit)
    """
    payload = {
        "model": model,
        "evidence": evidence_map,
        "data": {"huggingface": hf, "github": gh, "arxiv": ax},
    }
    target_model = os.getenv("OPENAI_MODEL_EVALUATOR", "o3")
    rsp = client.chat.completions.create(
        model=target_model,
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
    scores_dict = raw.get("scores", raw) if isinstance(raw, dict) else {}
    out: Dict[str, Dict] = {}
    for k, v in scores_dict.items():
        if isinstance(v, dict):
            out[k] = {"score": float(v.get("score", 0)), "reason": v.get("reason","")}
        elif isinstance(v, (int, float)):
            out[k] = {"score": float(v), "reason": ""}
    return out

# ─────────── API scorer with web-search boost (official OR reputable third-party) ───────────
_PLANNED_API_PAT = re.compile(r"(?i)\b(api)\b[^.\n]{0,60}\b(coming soon|will be (?:made )?public|planned|under development|launching soon)\b")

def _score_api_lenient(hf: Dict, gh: Dict, ax: Dict, rp: Dict, model_name: str) -> Tuple[float, str]:
    """
    Open (1.0): evidence of owner-hosted API or reputable third-party aggregator (quotes with URLs, or web search).
    Semi-Open (0.5): owner announces 'coming soon' in quotes.
    Closed (0.0): otherwise.
    """
    org_tokens = _org_tokens_from_model_name(model_name)

    # 1) Quotes
    quotes: List[str] = []
    for src in (hf or {}, gh or {}, ax or {}, rp or {}):
        quotes.extend(_collect_quotes_from(src, ["2-3 (API)__evidence", "2-3 API__evidence"]))

    base_score, base_reason = 0.0, "No API-related evidence."
    if quotes:
        urls_in_quotes = []
        for q in quotes:
            urls_in_quotes.extend(_URL_RE.findall(q or ""))
        ok_in_quotes = [u for u in urls_in_quotes if _is_official_or_thirdparty_api_url(u, org_tokens)]

        if ok_in_quotes:
            base_score = 1.0
            base_reason = f"API docs in quotes: {', '.join(ok_in_quotes[:3])}"
        else:
            joined = " \n".join(quotes)
            if _PLANNED_API_PAT.search(joined):
                base_score = 0.5
                base_reason = "Owner announced API 'coming soon'."
            else:
                base_score = 0.0
                base_reason = "Only generic/irrelevant mentions; no qualifying API docs."

    # 2) Web search boost
    urls = _websearch_api_urls(model_name, org_tokens, API_WEBSEARCH_MAX_URLS)
    if urls:
        reason_ws = f"Web search found API docs: {', '.join(urls[:3])}"
        final_score = max(base_score, 1.0)
        final_reason = (base_reason + "  " + reason_ws).strip()
        return final_score, final_reason
    else:
        reason_ws = "No API docs found via web search."
        final_reason = (base_reason + "  " + reason_ws).strip()
        return base_score, final_reason

# ─────────── Main evaluation ───────────
def evaluate_openness(model_name: str,
                      hf_json=None, gh_json=None, arxiv_json=None, reports_json=None,
                      pretrain_parts=None, hf_raw_json=None) -> Dict:
    hf, gh, ax = hf_json or {}, gh_json or {}, arxiv_json or {}
    rp = reports_json or {}

    # 0) Merge pretrain sources (optional, used for 3-1 / 4-1)
    pre_bundle = _merge_pretrain_parts(pretrain_parts)

    # 1) Strict evidence map for 3-x / 4-x (+inject pretrain bundle where applicable)
    evmap = _collect_evidence_maps(ax, hf, gh, rp, pretrain_bundle=pre_bundle)

    # 2) Baseline GPT scoring (we will overwrite some items with deterministic rules)
    scores = _gpt_evaluate(model_name, hf, gh, ax, evmap)

    # 3) Auto-open (1-1 / 1-5 / 1-6)
    scores.update(_auto_open_items(hf, hf_raw_json))

    # 4) Code (1-2) strict file-based detection
    code_score, code_reason = _detect_code_openness(hf, hf_raw_json)
    scores["1-2 Code"] = {"score": code_score, "reason": code_reason}

    # 4.1) License (1-3) — GPT-only. No rule-based override.

    # 5) Usage aggregation → exclusion base
    usage = _aggregate_usage(ax, hf, gh, rp)

    # Heuristic: fine-tuning 'generic claim only' → not_used
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
        if strict_key in scores:
            has_quotes = bool(evmap.get(strict_key, {}).get("quotes"))
            if not has_quotes:
                reason = scores[strict_key].get("reason","").strip()
                note = "No direct quote evidence; defaulting to Closed by strict rule."
                scores[strict_key] = {"score": 0.0, "reason": (reason + " " + note).strip()}

    # 6.1) 4-1 Pre-training Data: if quotes exist → minimum 0.5
    key_41 = "4-1 Pre-training Data"
    if key_41 in scores:
        has_quotes_41 = bool(evmap.get(key_41, {}).get("quotes"))
        if has_quotes_41 and float(scores.get(key_41, {}).get("score", 0)) < 0.5:
            scores[key_41] = {
                "score": 0.5,
                "reason": (scores.get(key_41, {}).get("reason","") + " Adjusted to Semi-Open: quotes indicate partial disclosure.").strip()
            }

    # 6.2) New lenient methodology scoring (Open/Semi-Open/Closed) for 3-1/3-2/3-3
    for meth_key in ("3-1 Pre-training", "3-2 Fine-tuning", "3-3 Reinforcement Learning"):
        used_flag = {
            "3-1 Pre-training": "used",
            "3-2 Fine-tuning": usage.get("fine_tuning", "unknown"),
            "3-3 Reinforcement Learning": usage.get("rl", "unknown"),
        }
        if used_flag[meth_key] == "not_used":
            continue
        qts = evmap.get(meth_key, {}).get("quotes", [])
        score_val = _lenient_method_score_from_quotes(qts, meth_key)
        if score_val == 1.0:
            reason = "Methodology is sufficiently detailed to reproduce training (objective/optimizer/schedule + batching/duration)."
        elif score_val == 0.5:
            reason = "Partial methodology disclosed (techniques/algorithms mentioned) but not fully reproducible."
        else:
            reason = "No concrete method details in quotes."
        scores[meth_key] = {"score": score_val, "reason": reason}

    # 6.3) Data Filtering (4-4): GPT-only. No rule-based override.

    # 6.4) 2-3 API: official OR reputable third-party
    api_score, api_reason = _score_api_lenient(hf, gh, ax, rp, model_name)
    scores["2-3 API"] = {"score": api_score, "reason": api_reason}

    # 6.5) Ensure all 16 keys exist (avoid missing items in output)
    for k in ALL_SCORE_KEYS:
        if k not in scores:
            if k in STRICT_KEYS and not evmap.get(k, {}).get("quotes"):
                scores[k] = {"score": 0.0, "reason": "No direct quote evidence; Closed by strict rule."}
            else:
                scores[k] = {"score": 0.0, "reason": "No evidence."}

    # 7) Build included set after exclusion
    included = {k:v for k,v in scores.items()
                if not any(k.startswith(p) for p in exclude_prefix)}

    # 8) Aggregate to 10-point scale
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

    # Filtered (dispatcher) outputs
    hf = _load_from_base(f"huggingface_filtered_final_{base}.json")
    gh = _load_from_base(f"github_filtered_final_{base}.json")
    ax = _load_from_base(f"arxiv_filtered_final_{base}.json")
    rp = _load_from_base(f"reports_filtered_final_{base}.json")

    # Raw HF fetcher output (for code/weights/tokenizer detection)
    hf_raw = _load_from_base(f"huggingface_{base}.json")

    # Load pretrain bundles if base_model_id is present
    pretrain = {"hf": {}, "github": {}, "arxiv": {}, "reports": {}}
    if base_model_id:
        pbase = base_model_id.replace("/", "_").lower()
        pretrain["hf"]      = _load_from_base(f"pretrain_hf_{pbase}.json") or _load_from_base(f"pretrain_huggingface_{pbase}.json")
        pretrain["github"]  = _load_from_base(f"pretrain_gh_{pbase}.json") or _load_from_base(f"pretrain_github_{pbase}.json")
        pretrain["arxiv"]   = _load_from_base(f"pretrain_arxiv_{pbase}.json")
        pretrain["reports"] = _load_from_base(f"pretrain_reports_{pbase}.json")

    res = evaluate_openness(model_name, hf_json=hf, gh_json=gh, arxiv_json=ax,
                            reports_json=rp, pretrain_parts=pretrain,
                            hf_raw_json=hf_raw)
    out = base_dir / f"openness_score_{base}.json"
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("📝 Saved evaluation result:", out)
    return res


if __name__ == "__main__":
    evaluate_openness_from_files("facebook/opt-125m", base_dir="facebook_opt-125m")

## 데이터필터링 ,라이센스 코드 제거