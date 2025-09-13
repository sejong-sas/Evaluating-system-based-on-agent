# github_Dispatcher.py
# High-Recall 2-Pass  (evidence {source, quote} → long summary)
# - store evidence as an array of objects [{source, quote}, …]
# - summaries must use quotes only
# - STRICT by default: collect ONLY quotes that explicitly mention the TARGET model
#   BUT for data/method items (4-4, 4-1, 3-1) and LICENSE (1-3) allow weak-guard when strong signals exist
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

# Tunables
MODEL_NAME = os.getenv("OPENAI_MODEL_GH_DISPATCHER", "o3")
GITHUB_README_CHAR_CAP      = int(os.getenv("GITHUB_README_CHAR_CAP", "120000"))
GITHUB_LICENSE_CHAR_CAP     = int(os.getenv("GITHUB_LICENSE_CHAR_CAP", "20000"))
GITHUB_MAX_LICENSE_PARTS    = int(os.getenv("GITHUB_MAX_LICENSE_PARTS", "5"))
GITHUB_MAX_PY_FILES         = int(os.getenv("GITHUB_MAX_PY_FILES", "40"))
GITHUB_PY_FILE_CHAR_CAP     = int(os.getenv("GITHUB_PY_FILE_CHAR_CAP", "20000"))
GITHUB_MAX_FILES_LIST       = int(os.getenv("GITHUB_MAX_FILES_LIST", "3000"))
CHUNK_CHARS                 = int(os.getenv("GITHUB_CHUNK_CHARS", "60000"))
CHUNK_OVERLAP               = int(os.getenv("GITHUB_CHUNK_OVERLAP", "2000"))
EVIDENCE_LIMIT_PER_KEY      = int(os.getenv("GITHUB_EVIDENCE_LIMIT_PER_KEY", "300"))

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

# ───────────── Item descriptions (brief) ─────────────
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
    LABELS["4-4"]: "All information about data filtering/cleaning methods, criteria used, processes, and their impact(or include llama guard)",
}

# ───────────── Groups ─────────────
ITEM_GROUPS = [
    ["1-1", "1-2", "1-3", "1-4"],
    ["1-5", "1-6", "2-1", "2-2"],
    ["2-3", "3-1", "3-2", "3-3"],
    ["4-1", "4-2", "4-3", "4-4"],
]

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

# === Target-model guard (공통, 강화판) =========================================
_STOPWORDS = {
    # generic
    "ai","llm","ml","nlp","model","models","base","chat","instruct","instruction","sft","rl","rm",
    "eval","evaluation","bench","benchmark","paper","release","repo","library","toolkit","example",
    # versions/quality
    "v","v1","v2","v3","dev","alpha","beta","rc","preview","nightly","experimental","test","demo",
    # misc
    "hf","torch","cuda","jit"
}

def _canonical_model_tokens(model_id: str) -> List[str]:
    """
    Robust tokens from model id:
      - split on non-alnum, drop short/generic tokens
      - add joined (remove non-alnum) and no-digit variants
      - add (name, name+digits) splits like llama3, llama3.1 → llama, llama3, 31
    """
    name = (model_id or "").split("/", 1)[-1].lower()
    raw = re.split(r"[^a-z0-9.]+", name)
    alts = set()
    for t in raw:
        t = t.strip()
        if not t:
            continue
        if t in _STOPWORDS:
            continue
        if len(t) >= 2:
            alts.add(t)
        m = re.match(r"([a-z]+)(\d+(?:\.\d+)*)$", t)
        if m:
            base = m.group(1); ver = m.group(2)
            alts.add(base)
            alts.add(base + ver.replace(".", ""))  # llama31
            alts.add(ver)                          # 3.1
            alts.add(ver.replace(".", ""))         # 31
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
        f"- Prefer quotes whose sentence explicitly mentions one of: {toks}.\n"
        "- Reject sentences about other models unless the TARGET is named in the same sentence.\n"
        "- Exception (data/method items 4-4, 4-1, 3-1 and LICENSE 1-3): if a sentence clearly describes filtering/data criteria or LICENSE tokens/links, accept it even without the TARGET token.\n"
        "- If in doubt, DROP the quote.\n"
    )

def _quote_mentions_target(q: str, model_id: str) -> bool:
    if not q: return False
    ql = q.lower().replace("–","-").replace("—","-")
    for t in _canonical_model_tokens(model_id):
        if t and t in ql:
            return True
    return False

# ───────────── Generalized filtering signal detectors (for weak-guard) ─────────────
_GENERIC_FILTER_KWS = (
    # categories
    "dedup", "duplicate", "near-duplicate", "minhash", "minhashlsh", "simhash", "lsh", "jaccard",
    "pii", "personally identifiable", "anonymiz", "redact", "de-identif",
    "language identification", "language-id", "langid", "cld3", "fasttext", "langdetect",
    "toxicity", "nsfw", "safety", "unsafe", "hate", "abuse", "violence", "sexual", "guardrail", "moderation",
    "quality filter", "quality score", "perplexity", "ppl", "classifier", "detector", "model-based", "llm-based",
    "contamination", "decontamination", "benchmark leakage", "advertisement", "ad spam", "spam",
    # pipeline-y words
    "cascaded filtering", "filtering pipeline", "multi-stage", "series of filtering", "stage 1", "stage 2", "step 1", "step 2",
)

_TOOL_NEAR_FILTER_PAT = re.compile(
    r'(?is)\b(?:use|using|employ(?:ing)?|leverage|apply|applied|adopt(?:ed|ing)?|'
    r'build(?:ing)?|train(?:ed|ing)?|run(?:ning)?)\b[^\.]{0,120}?'
    r'\b(?:model(?:-based)?|classifier|detector|moderation|guard(?:rail)?|'
    r'filter|sieve|cleaner|scrubber|anonymi[sz]er|redactor|safety model|content filter)\b'
)
_PROPER_TOOL_TITLE_PAT = re.compile(
    r'\b[A-Z][A-Za-z0-9_.-]{2,}\s+(?:Guard|Moderation|Shield|Filter|Detector|Classifier)\b'
)
_PERCENT_CHANGE_PAT = re.compile(
    r'(?i)\b(?:removed|discarded|dropped|filtered out|kept|retained|reduced(?: by)?)\b'
    r'[^.%]{0,60}\b\d{1,3}\s?%')
_THRESH_GENERIC_PAT = re.compile(
    r'(?i)\b(?:threshold|cut-?off|cutoff|score)\b[^\.]{0,40}?(?:>=|≤|<=|<|>|=)\s*'
    r'(?:0?\.\d+|[1-9]\d*(?:\.\d+)?)')
_METRIC_THRESH_PAT = re.compile(
    r'(?i)\b(?:jaccard|similarity|overlap|perplexity|ppl)\b[^\.]{0,40}?'
    r'(?:>=|≤|<=|<|>|=)\s*(?:0?\.\d+|[1-9]\d*(?:\.\d+)?)')
_PIPELINE_PAT = re.compile(
    r'(?i)\b(?:pipeline|multi[- ]?stage|cascad(?:ed|e)|series of filter|'
    r'stage\s*\d+|step\s*\d+)\b'
)

def _quote_has_filtering_signal(q: str) -> bool:
    if not q: return False
    ql = q.lower()
    if any(k in ql for k in _GENERIC_FILTER_KWS): return True
    if (_TOOL_NEAR_FILTER_PAT.search(q) or _PROPER_TOOL_TITLE_PAT.search(q) or
        _PERCENT_CHANGE_PAT.search(q) or _THRESH_GENERIC_PAT.search(q) or
        _METRIC_THRESH_PAT.search(q) or _PIPELINE_PAT.search(q)):
        return True
    return False

# ───────────── LICENSE detectors (weak-guard 허용) ─────────────
LICENSE_TOKENS = (
    # SPDX/common names
    "apache-2.0","apache 2.0","mit","bsd-2-clause","bsd-3-clause","bsd 2-clause","bsd 3-clause",
    "gpl-2.0","gpl-3.0","lgpl-3.0","agpl-3.0","mpl-2.0","mpl 2.0",
    "cc-by","cc by","cc-by-sa","cc by-sa","cc-by-nc","cc by-nc","cc0","unlicense",
    # model licenses
    "openrail","openrail-m","open rail","bigcode openrail","falcon-llm license","falcon llm license",
    "tii falcon-llm license","tii falcon llm license",
)

LICENSE_LINE_PAT = re.compile(r'(?im)^\s*(license|license_name)\s*:\s*([^\n]+)$')
LICENSE_LINK_PAT = re.compile(r'(?i)\b(license_link|licen[cs]e url|licen[cs]e link)\b\s*[:=]\s*([^\s]+)')
ANY_LICENSE_INLINE_PAT = re.compile(r'(?i)\blicen[cs]e\b[^:\n]*[:\-]?\s*([A-Za-z0-9 .+/\-]+)')

KNOWN_LICENSE_URLS = {
    "apache-2.0": "https://www.apache.org/licenses/LICENSE-2.0",
    "mit": "https://opensource.org/license/mit/",
    "openrail": "https://www.bigcode-project.org/docs/pages/model-license/",
    "openrail-m": "https://www.bigcode-project.org/docs/pages/model-license/",
    "falcon-llm license": "https://falconllm.tii.ae/falcon-terms-and-conditions.html",
}

def _quote_has_license_signal(q: str) -> bool:
    if not q: return False
    ql = q.lower()
    if "license" in ql and any(tok in ql for tok in LICENSE_TOKENS):
        return True
    # front-matter or explicit 'license:' lines
    if LICENSE_LINE_PAT.search(q) or LICENSE_LINK_PAT.search(q) or ANY_LICENSE_INLINE_PAT.search(q):
        return True
    return False

ALLOW_WEAK_GUARD = {
    LABELS["4-4"], LABELS["4-1"], LABELS["3-1"],  # 기존
    LABELS["1-3"],                                # ← LICENSE 약한 가드 허용
}

def _is_relevant_quote(lbl: str, quote: str, model_id: str) -> bool:
    # 1) Strong rule: sentence mentions target tokens
    if _quote_mentions_target(quote, model_id):
        return True
    # 2) Weak rule for data/method items + LICENSE
    if lbl in ALLOW_WEAK_GUARD:
        if lbl == LABELS["1-3"]:
            return _quote_has_license_signal(quote)
        return _quote_has_filtering_signal(quote)
    return False

# ───────────── Prompts ─────────────
_GH_FILTER_HINTS = (
    "dedup/near-duplicate/Minhash/SimHash/LSH/Jaccard",
    "PII redaction/anonymization; language-ID (CLD3/fastText/langdetect)",
    "toxicity/NSFW/safety; guard/moderation; model/LLM-based filtering",
    "numeric thresholds/ratios: ppl < …, score ≥ …, removed XX%",
    "phrases: cascaded filtering pipeline / multi-stage / series of filtering",
)

_BASE_RECALL_SYS = f"""
You are an expert at extracting AI model openness evaluation information from a GitHub repository.
Using only the payload (original text), return evidence for each item in the format:
  [{{ "source": "...", "quote": "..." }}, …]
· source  : one of [repo], [readme], [license_files], [files], [py_files/xxx.py]
· quote   : a verbatim sentence copied from that section (no edits/summaries)
If there is no evidence, return an empty array [].
Focus: for "4-4 (Data Filtering)" look for these signals → {", ".join(_GH_FILTER_HINTS)}
You must output a JSON object only.
""".strip()

_BASE_SUMMARY_SYS = """
Using the provided quotes only, write long and detailed summaries for each item.
- For "4-4 (Data Filtering)", prefer quotes that include concrete criteria: tool/classifier mentions,
  numeric thresholds/ratios (e.g., Jaccard 0.95, ppl < X, removed Y%), or pipeline-stage wording.
You may also use quotes that refer to models in the same series with the same major version and a minor version difference of 0.4 or less (for example, v1.2 and v1.5).
Do NOT use quotes that refer to any other models outside of this range.
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
    base = (
        _model_guard_text(model_id) +
        "\nItems in this group:\n" + _js(_desc(g)) +
        "\nReturn a JSON object with EXACTLY these keys (arrays of {source,quote}):\n" +
        _js({LABELS[i]: [] for i in g})
    )
    # Add explicit hints when 4-4 is in scope
    if "4-4" in g:
        hints = (
            "\nHINTS for 4-4 (Data Filtering):\n"
            "- dedup/near-duplicate/Minhash/SimHash/LSH/Jaccard\n"
            "- PII redaction/anonymization; language-ID (CLD3/fastText/langdetect)\n"
            "- toxicity/NSFW/safety; guard/moderation; model/LLM-based filtering\n"
            "- numeric thresholds/ratios (ppl < …, score ≥ …, removed XX%)\n"
            "- pipeline phrases: cascaded filtering pipeline / multi-stage / series of filtering\n"
            "For 4-4/4-1/3-1 you may keep a quote without the TARGET token if it clearly describes these signals."
        )
        return base + hints
    return base

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
        model=MODEL_NAME,
        reasoning_effort="medium",
        response_format={"type": "json_object"},
        messages=[{"role":"system","content":sys_msg},
                  {"role":"user","content":usr}]
    )
    try:    return json.loads(r.choices[0].message.content.strip())
    except: return {}

# ───────────── Build payload ─────────────
def _make_payload(d: Dict, _: int) -> Dict:
    repo   = d.get("repo") or d.get("full_name") or ""
    files  = (d.get("files") or [])[:GITHUB_MAX_FILES_LIST]
    readme = (d.get("readme") or "")[:GITHUB_README_CHAR_CAP]

    # license (dict or list both 지원)
    lic = d.get("license_files") or {}
    if isinstance(lic, dict):
        items = list(lic.items())[:GITHUB_MAX_LICENSE_PARTS]
        lic_text = "\n\n".join(f"# {k}\n{(v or '')[:GITHUB_LICENSE_CHAR_CAP]}" for k, v in items)
    elif isinstance(lic, list):
        buf = []
        for it in lic[:GITHUB_MAX_LICENSE_PARTS]:
            if isinstance(it, dict):
                name = it.get("name") or it.get("path") or "LICENSE"
                buf.append(f"# {name}\n{(it.get('content') or '')[:GITHUB_LICENSE_CHAR_CAP]}")
            elif isinstance(it, str):
                buf.append(it[:GITHUB_LICENSE_CHAR_CAP])
        lic_text = "\n\n".join(buf)
    else:
        lic_text = str(lic)[:GITHUB_LICENSE_CHAR_CAP]

    py_files = {}
    for fn, src in (d.get("py_files") or {}).items():
        if len(py_files) >= GITHUB_MAX_PY_FILES: break
        py_files[fn] = (src or "")[:GITHUB_PY_FILE_CHAR_CAP]

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
            if not _is_relevant_quote(lbl, qt, model_id): continue
            kept.append({"source": src, "quote": qt})
        out[lbl] = _dedup_evid(kept, EVIDENCE_LIMIT_PER_KEY)
    return out

# ───────────── Evidence re-balance (3-2 → 4-2) ─────────────
def _rebalance_evidence(ev: Dict[str, List[Dict[str,str]]]) -> Dict[str, List[Dict[str,str]]]:
    ft_lbl = "3-2 (Fine-tuning)"
    fd_lbl = "4-2 (Fine-tuning Data)"
    if (not ev.get(fd_lbl)) and ev.get(ft_lbl):
        kws = r"(dataset|corpus|xP3(?:mt)?|p3\b|language distribution|languages|split|examples|released|Flores200|ROOTS|license|public links?)"
        moved = [e for e in ev[ft_lbl] if isinstance(e, dict) and re.search(kws, e.get("quote",""), re.I)]
        if moved:
            ev[fd_lbl] = _dedup_evid(moved, EVIDENCE_LIMIT_PER_KEY)
    return ev

# ───────────── README paper backstop (1-4) ─────────────
_PAPER_URL_RE = re.compile(r"(https?://\S*(arxiv\.org|openreview\.net|doi\.org|acm\.org|ieee\.org)\S*)", re.I)
def _inject_backstop_paper(ev: Dict[str, List[Dict[str,str]]], payload: Dict[str,Any]) -> Dict[str, List[Dict[str,str]]]:
    lbl = "1-4 (Paper)"
    if ev.get(lbl):
        return ev
    readme = payload.get("readme") or ""
    m = _PAPER_URL_RE.search(readme)
    if m:
        url = m.group(1).strip()
        ev[lbl] = [{"source":"readme","quote":url}]
    return ev

# ───────────── LICENSE backstops (1-3): README / front-matter / links / known URLs ─────────────
def _scan_license_in_text(text: str) -> List[str]:
    out: List[str] = []
    if not text:
        return out
    # front-matter style
    for m in LICENSE_LINE_PAT.finditer(text):
        out.append(m.group(0).strip())
    # explicit link line
    for m in LICENSE_LINK_PAT.finditer(text):
        out.append(f"{m.group(1)}: {m.group(2)}")
    # generic inline mention
    for m in ANY_LICENSE_INLINE_PAT.finditer(text):
        line = m.group(0).strip()
        if any(tok in line.lower() for tok in LICENSE_TOKENS):
            out.append(line)
    return out

def _inject_backstop_license(ev: Dict[str, List[Dict[str,str]]], payload: Dict[str,Any]) -> Dict[str, List[Dict[str,str]]]:
    lbl = "1-3 (License)"
    if ev.get(lbl):
        return ev

    readme = payload.get("readme") or ""
    injected: List[Dict[str,str]] = []

    # 1) README/front-matter lines
    for ln in _scan_license_in_text(readme):
        injected.append({"source":"readme", "quote": ln})

    # 2) Known license URL hints from README text
    rl = readme.lower()
    for key, url in KNOWN_LICENSE_URLS.items():
        if key in rl:
            injected.append({"source":"readme", "quote": f"License reference: {url}"})

    # 3) If still empty AND license_files text exists, grab first non-empty heading/line
    if not injected:
        lic_text = payload.get("license_files") or ""
        lines = [ln.strip() for ln in lic_text.splitlines() if ln.strip()]
        for ln in lines[:20]:
            if _quote_has_license_signal(ln) or ln.lower().startswith(("license","copyright")):
                injected.append({"source":"license_files", "quote": ln})
                if len(injected) >= 5: break

    if injected:
        ev[lbl] = _dedup_evid(injected, EVIDENCE_LIMIT_PER_KEY)
    return ev

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
    ev = _rebalance_evidence(ev)
    # Backstops
    payload = None  # only for logging; backstops are injected in filter_github_features()
    kept_counts = {k: len(v or []) for k, v in ev.items()}
    print("evidence counts before/after model-guard:", {"raw": raw_counts, "kept": kept_counts})
    return ev

# ───────────── Step 2: summarize ─────────────
def _summarize(g, ev: Dict[str, List[Dict[str,str]]], model_id: str) -> Dict[str, str]:
    lbls = [LABELS[k] for k in g]
    quotes = {lbl: [(e.get("quote") or "") for e in (ev.get(lbl) or [])] for lbl in lbls}
    ans = _chat_json(_BASE_SUMMARY_SYS, _summ_inst(g, model_id) + "\n=== QUOTES ===\n" + _js(quotes))
    return {lbl: (ans.get(lbl, "") or "") for lbl in lbls}

# ─── merge ───
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

            # Backstops after post-filter:
            if LABELS["1-3"] in ev and not ev[LABELS["1-3"]]:
                ev = _inject_backstop_license(ev, payload)
            elif LABELS["1-3"] not in ev:
                ev = _inject_backstop_license(ev, payload)

            ev   = _inject_backstop_paper(ev, payload)  # README 내 논문 링크 백스톱
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
