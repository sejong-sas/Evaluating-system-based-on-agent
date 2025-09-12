# reports_Dispatcher.py
# Purpose:
#   - Merge and deduplicate "report-ish" texts gathered by fetchers
#     (HF: reports_fulltext_{hf}.json, GH: reports_fulltext_github_{owner_repo}.json)
#     PLUS arXiv fetcher outputs (arxiv_fulltext_{hf}.json / arxiv_{hf}.json).
#   - Filter out irrelevant sources to the given model (doc-level + quote-level).
#   - Run a 2-pass (evidence → summary) extraction for the 16 openness items.
#   - Save group outputs and a final merged JSON:
#       reports_filtered_{base}_{1..4}.json, reports_filtered_final_{base}.json
#
# Env:
#   OPENAI_API_KEY (required)
#   OPENAI_MODEL_REPORTS_DISPATCHER (default "o3")
#   REPORTS_FILTER_ALWAYS=1 (force doc-level relevance filter)
#   REPORTS_FILTER_THRESHOLD=20 (apply doc-filter when docs > threshold)
#   REPORTS_MIN_HITS_IN_TEXT=2 (min body hits for doc to be related)
#   REPORTS_URL_DENY_SUBSTR="blog.foo, old-version" (denylist substrings)
#   REPORTS_DISPATCHER_SECTION_CHAR_CAP / REPORTS_DISPATCHER_MAX_SECTIONS (optional caps)
#   REPORTS_LICENSE_CANONICAL_URLS="https://example.com/license1,https://example.com/license2" (optional, license-only refs)

import os, json, re, hashlib
from typing import Dict, List, Any, Tuple
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ───────────────────────────── Env ─────────────────────────────
load_dotenv()
_API_KEY = os.getenv("OPENAI_API_KEY")
if not _API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
_client = OpenAI(api_key=_API_KEY)
MODEL_NAME = os.getenv("OPENAI_MODEL_REPORTS_DISPATCHER", "o3")

# ─────────────────────── 16 evaluation items ───────────────────────
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

ITEM_GROUPS = [
    ["1-1","1-2","1-3","1-4"],
    ["1-5","1-6","2-1","2-2"],
    ["2-3","3-1","3-2","3-3"],
    ["4-1","4-2","4-3","4-4"],
]

# ───────────────────────── Parameters ─────────────────────────
CHUNK_CHARS = 60_000
CHUNK_OVERLAP = 2_000
EVIDENCE_LIMIT_PER_KEY = 300
SECTION_CHAR_CAP = int(os.getenv("REPORTS_DISPATCHER_SECTION_CHAR_CAP", "0"))  # 0=unlimited
MAX_SECTIONS = int(os.getenv("REPORTS_DISPATCHER_MAX_SECTIONS", "0"))          # 0=unlimited

REPORTS_FILTER_ALWAYS = os.getenv("REPORTS_FILTER_ALWAYS", "0") == "1"
REPORTS_FILTER_THRESHOLD = int(os.getenv("REPORTS_FILTER_THRESHOLD", "20"))
REPORTS_MIN_HITS_IN_TEXT = int(os.getenv("REPORTS_MIN_HITS_IN_TEXT", "2"))
REPORTS_URL_DENY_SUBSTR = os.getenv("REPORTS_URL_DENY_SUBSTR", "")
REPORTS_LICENSE_CANONICAL_URLS = os.getenv("REPORTS_LICENSE_CANONICAL_URLS", "")

# ───────────────────────── Utils ─────────────────────────
def _js(o): return json.dumps(o, ensure_ascii=False, indent=2)
_PARA_SPLIT = re.compile(r"\n\s*\n+")

def _normalize_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()

def _dedup_texts_by_paragraph(texts: List[str]) -> str:
    """Concatenate documents while removing duplicate paragraphs (first-seen wins)."""
    seen, out = set(), []
    for doc in texts:
        if not doc: continue
        paras = _PARA_SPLIT.split(doc.strip()) if _PARA_SPLIT.search(doc or "") else [doc.strip()]
        for p in paras:
            p = p.strip()
            if not p: continue
            key = hashlib.sha1(_normalize_for_hash(p).encode("utf-8")).hexdigest()
            if key in seen: continue
            seen.add(key); out.append(p)
    return "\n\n".join(out)

def _dedup_evs(evs: List[Dict[str,str]], limit:int):
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

# ────────────────── Target-model guard (강화) ──────────────────
_STOPWORDS = {
    "ai","llm","language","nlp","ml","model","models","base","chat","instruct","instruction",
    "sft","rl","rlhf","eval","evaluation","bench","benchmark","dev","test","demo","preview",
    "alpha","beta","rc","hf","release","v","v1","v2","v3","v4","v5","it"
}

def _family_tokens_from_model_id(model_id: str) -> set[str]:
    """
    Extract stable tokens for family/version matching.
    Ignore too-generic tokens.
    """
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
        m = re.match(r"([a-z]+)(\d+(?:\.\d+)*)$", tt)  # llama3.1 → (llama, 3.1)
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
        f"- Prefer a quote ONLY if the sentence explicitly mentions one of: {toks}.\n"
        "- If a section mixes multiple models or earlier/other versions, keep only sentences that also include TARGET tokens.\n"
        "- If the TARGET name is in the immediately previous sentence within the same paragraph and the current sentence contains the fact, include BOTH sentences together as one quote (previous + current).\n"
        "- If in doubt, DROP the quote."
    )

def _source_tag_mentions_target(src: str, model_id: str) -> bool:
    """Allow meta match by [sections/<title/url>] tags if they include TARGET tokens."""
    if not isinstance(src, str): return False
    s = src.strip().lower().replace("–","-").replace("—","-").strip("[]")
    for t in _family_tokens_from_model_id(model_id):
        if len(t) >= 2 and t in s:
            return True
    return False

def _quote_mentions_target(q: str, model_id: str) -> bool:
    if not q: return False
    ql = q.lower().replace("–","-").replace("—","-")
    for t in _family_tokens_from_model_id(model_id):
        if len(t) >= 2 and t in ql:
            return True
    return False

# ───────── Generalized filtering-signal detection (for weak-guard) ─────────
_GENERIC_FILTER_KWS = (
    # categories / signals
    "dedup", "duplicate", "near-duplicate", "minhash", "minhashlsh", "simhash", "lsh", "jaccard",
    "pii", "anonymiz", "de-identif", "redact",
    "langid", "language identification", "language-id", "language detection", "cld3", "fasttext", "langdetect",
    "toxicity", "nsfw", "safety", "unsafe", "hate", "abuse", "violence", "sexual", "guardrail", "moderation",
    "quality filter", "quality score", "perplexity", "ppl", "classifier", "detector", "model-based filter", "llm-based filter",
    "contamination", "decontamination", "benchmark leakage",
    "advertisement", "ad spam", "spam",
    # pipeline phrasing
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
    r'(?i)\b(?:jaccard|similarity|overlap|perplexity|ppl|score)\b[^\.]{0,40}?'
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

def _allow_weak_guard_labels() -> set[str]:
    """Labels where model-name is often omitted in methodology text (allow weak-guard)."""
    return { LABELS["4-4"], LABELS["4-1"], LABELS["3-1"], LABELS["1-3"] }  # + License

# ───────────── License detection (weak-guard + canonical URL injection) ─────────────
_LICENSE_NAME_PAT = re.compile(
    r'(?i)\b(?:license|licence|licensing|licensed under|license:)\b[^.\n]{0,120}?'
    r'(apache\s*2(\.0)?|apache-2\.0|mit|bsd|gpl|lgpl|agpl|mpl|mozilla public license|'
    r'cc[- ]?by(?:[- ]?sa)?|creative commons|openrail(?:-m)?|bigcode openrail(?:-m)?|'
    r'falcon[-\s]*llm\s*license|tii\s*f(alcon)?[-\s]*llm\s*license|'
    r'llama\s*2\s*community\s*license|open model license|oml)'
)

def _license_canonical_urls_from_text(text: str) -> List[str]:
    tl = (text or "").lower()
    urls: List[str] = []
    # Family-level canonical pages (no model-specific branching)
    if "openrail" in tl or "bigcode" in tl:
        urls.append("https://www.bigcode-project.org/docs/pages/model-license/")
    if "llama 2" in tl and "license" in tl:
        urls.append("https://ai.meta.com/llama/license/")
    if "falcon" in tl and "license" in tl:
        # Use TII Falcon license exemplar (generic reference)
        urls.append("https://huggingface.co/tiiuae/falcon-40b/blob/main/LICENSE")
    # User-provided extras
    extra = [u.strip() for u in re.split(r"[,\s]+", REPORTS_LICENSE_CANONICAL_URLS) if u.strip()]
    for u in extra:
        if u not in urls:
            urls.append(u)
    return urls

def _inject_license_backstop(ev: Dict[str, List[Dict[str,str]]],
                             payload_text: str) -> Dict[str, List[Dict[str,str]]]:
    """
    1) If '1-3 (License)' evidence is empty, scan payload for license phrases and inject quotes.
    2) Regardless, if the text indicates a known license family, inject canonical URL references
       as additional evidence with source=[web:<url>] (no HTTP fetch, just a canonical pointer).
    """
    lbl = LABELS["1-3"]
    text = payload_text or ""
    existing = ev.get(lbl) or []

    # (1) Phrase-based quote harvesting (weak-guard allowed)
    if not existing:
        adds: List[Dict[str,str]] = []
        for m in _LICENSE_NAME_PAT.finditer(text):
            start = max(0, m.start() - 160)
            end   = min(len(text), m.end() + 160)
            snip  = re.sub(r"[ \t]+", " ", text[start:end].strip())
            if snip:
                adds.append({"source":"[pdf_text]","quote":snip})
        if adds:
            ev[lbl] = _dedup_evs(adds, EVIDENCE_LIMIT_PER_KEY)
            existing = ev[lbl]

    # (2) Canonical URL references
    canon_urls = _license_canonical_urls_from_text(text)
    if canon_urls:
        ev.setdefault(lbl, [])
        for u in canon_urls:
            ev[lbl].append({"source": f"[web:{u}]","quote": u})
        ev[lbl] = _dedup_evs(ev[lbl], EVIDENCE_LIMIT_PER_KEY)

    return ev

# ───────────────────────── Prompts ─────────────────────────
def _desc(ids): return {LABELS[i]: EVAL_DESCRIPTIONS[LABELS[i]] for i in ids}
def _skeleton(g):  return {LABELS[i]: [] for i in g}

# Hints for 4-4
_FILTER_HINTS = (
    "dedup/near-duplicate/Minhash/SimHash/LSH/Jaccard",
    "PII anonymization/redaction; language-ID (CLD3, fastText, langdetect)",
    "toxicity/NSFW/safety; moderation/guard",
    "model/LLM-based filtering; classifier/detector",
    "numeric thresholds/ratios: ppl < …, score >= …, removed XX%",
    "pipeline phrasing: cascaded filtering pipeline / multi-stage / series of filtering"
)

_BASE_RECALL_SYS = """
You are an expert at extracting AI model openness evaluation information from *technical reports, papers, and blogs*.
Using only the payload (original text), return evidence for each item in the format
  [{ "source": "...", "quote": "..." }, …]
· source : e.g., [url:<...>], [sections/<url>], [pdf_text], [title], [abstract]
· quote  : a verbatim sentence copied from that section (no edits)

STRICT TARGET POLICY:
- Prefer quotes where the sentence itself explicitly names a TARGET token.
- If the TARGET name is in the immediately previous sentence within the same paragraph and the current sentence carries the fact,
  include BOTH sentences together as one quote (previous + current).
- Drop any content that cannot be tied to the TARGET by these rules.

If there is no evidence, return an empty array [].
You must output a JSON object only.
""".strip()

_BASE_SUMMARY_SYS = """
Using the provided quotes only, write long and detailed summaries for each item.
- For "4-4 (Data Filtering)", prefer quotes that include concrete criteria: tool/classifier mentions,
  numeric thresholds/ratios (e.g., Jaccard 0.95, ppl < X, removed Y%), or pipeline-stage wording.
You must output a JSON object only.
""".strip()

def _recall_inst(g: List[str], model_id: str) -> str:
    base = (
        _model_guard_text(model_id) +
        "\nItems in this group:\n" + _js(_desc(g)) +
        "\nReturn a JSON object with EXACTLY these keys (arrays of {source,quote}):\n" +
        _js(_skeleton(g))
    )
    if "4-4" in g:
        hints = (
            "\n\nMUST LOOK HARD FOR '4-4 (Data Filtering)':\n- Hints: " +
            "; ".join(_FILTER_HINTS) +
            "\n- Accept numeric thresholds/ratios (e.g., 0.95, 80%, ppl < 40), pipeline phrases (cascaded/multi-stage),\n"
            "  and 'model/LLM-based filtering' or 'classifier/detector' as concrete details."
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

# ───────────────────── GPT JSON call ─────────────────────
def _chat_json(sys, usr):
    r = _client.chat.completions.create(
        model=MODEL_NAME, reasoning_effort="medium",
        response_format={"type":"json_object"},
        messages=[{"role":"system","content":sys},
                  {"role":"user","content":usr}]
    )
    try:    return json.loads(r.choices[0].message.content.strip())
    except: return {}

# ───────────────────── Payload builder ─────────────────────
def _make_payload(doc: Dict[str, Any]) -> Dict[str, Any]:
    secs = []
    for it in (doc.get("full_texts") or []):
        u = str(it.get("arxiv_id") or it.get("id") or it.get("url") or "")[:2000]
        tx = str(it.get("full_text") or it.get("pdf_text") or "")
        if not tx: continue
        if SECTION_CHAR_CAP > 0: tx = tx[:SECTION_CHAR_CAP]
        secs.append({"title": u or "doc", "text": tx})

    joined = _dedup_texts_by_paragraph([s["text"] for s in secs])

    return {
        "title":    "",
        "abstract": "",
        "pdf_text": joined,
        "sections": secs[:MAX_SECTIONS] if (MAX_SECTIONS>0) else secs,
        "bib": "",
    }

def _payload_text(p: Dict[str, Any]) -> str:
    parts = []
    if p.get("title"):    parts.append(f"[title]\n{p.get('title')}\n")
    if p.get("abstract"): parts.append(f"[abstract]\n{p.get('abstract')}\n")
    parts.append(f"[pdf_text]\n{p.get('pdf_text','')}\n")
    for s in p.get("sections", []):
        tag = (s.get("title") or "doc").strip()
        parts.append(f"[sections/{tag}]\n{s.get('text','')}\n")
    return "\n".join(parts)

# ───────────────── Allowed source tags & validators ─────────────────
_ALLOWED_PREFIX = ("url:", "sections/", "pdf_text", "title", "abstract", "web:")
_EXTRA_OK_PREFIX = ("section","sec.","appendix","table","figure","fig.")
def _valid_source(src: str) -> bool:
    if not isinstance(src, str): return False
    s = src.strip().lower().strip("[]")
    return s.startswith(_ALLOWED_PREFIX) or s.startswith(_EXTRA_OK_PREFIX)

# ─────────────── Doc-level relevance filter ───────────────
def _deny_url(u: str) -> bool:
    if not REPORTS_URL_DENY_SUBSTR.strip():
        return False
    ul = (u or "").lower()
    for sub in re.split(r"[,\s]+", REPORTS_URL_DENY_SUBSTR.lower()):
        sub = sub.strip()
        if sub and sub in ul:
            return True
    return False

def _looks_related_doc(url: str, text: str, fam_tokens: set[str], min_hits: int) -> bool:
    """
    Keep doc if:
      • URL contains any family token (strong)
      • OR body contains >= min_hits occurrences of those tokens (word-boundary boosts)
    """
    ul = (url or "").lower()
    tl = (text or "").lower()
    if _deny_url(ul): return False
    for t in fam_tokens:
        if len(t) >= 3 and t in ul:
            return True
    hits = 0
    for t in fam_tokens:
        if not t: continue
        hits += tl.count(t)
        if re.search(rf"\b{re.escape(t)}\b", tl):
            hits += 1
    return hits >= max(1, min_hits)

# ───────────────── Evidence & Summary ─────────────────
def _collect(g: List[str], text: str, model_id: str):
    ev = {LABELS[k]: [] for k in g}
    i = 0; n = len(text)
    while i < n:
        end = min(i+CHUNK_CHARS, n)
        ch = text[i:end]
        i = end - CHUNK_OVERLAP if end - CHUNK_OVERLAP > i else end
        ans = _chat_json(_BASE_RECALL_SYS, _recall_inst(g, model_id)+"\n=== PAYLOAD ===\n"+ch)
        for k in g:
            arr = ans.get(LABELS[k], [])
            if isinstance(arr, list):
                ev[LABELS[k]].extend(arr)

    # quote-level 모델 필터(+source-tag), 약한 가드(4-4/4-1/3-1/1-3) 허용, dedup
    raw_counts = {LABELS[k]: len(ev.get(LABELS[k]) or []) for k in g}
    weak_labels = _allow_weak_guard_labels()
    ev2 = {}
    for k in g:
        lbl = LABELS[k]
        kept = []
        for e in (ev.get(lbl) or []):
            src = (e.get("source") or "").strip()
            qt  = (e.get("quote")  or "").strip()
            if not src or not qt: continue
            if not _valid_source(src): continue

            ok = (_quote_mentions_target(qt, model_id) or _source_tag_mentions_target(src, model_id))
            if not ok and (lbl in weak_labels):
                if lbl == LABELS["1-3"]:
                    ok = bool(_LICENSE_NAME_PAT.search(qt))
                else:
                    ok = _quote_has_filtering_signal(qt)

            if ok:
                kept.append({"source": src, "quote": qt})
        ev2[lbl] = _dedup_evs(kept, EVIDENCE_LIMIT_PER_KEY)

    kept_counts = {k: len(ev2.get(k) or []) for k in ev2}
    print("evidence counts before/after model-guard:", {"raw": raw_counts, "kept": kept_counts})
    return ev2

def _summarize(g: List[str], ev: Dict[str, List[Dict[str,str]]], model_id: str):
    quotes = {LABELS[k]: [e["quote"] for e in (ev.get(LABELS[k]) or [])] for k in g}
    ans = _chat_json(_BASE_SUMMARY_SYS, _summ_inst(g, model_id)+"\n=== QUOTES ===\n"+_js(quotes))
    return {LABELS[k]: (ans.get(LABELS[k], "") or "") for k in g}

def _merge(sum_, ev):
    return {lbl: (sum_.get(lbl,"") or "").strip() for lbl in sum_} | {
        f"{lbl}__evidence": ev.get(lbl, []) for lbl in sum_
    }

def _merge_all(lst):
    m={}
    for d in lst: m.update(d)
    return m

# ─────────────── RL usage classifier (FT-only → RL not_used 보정) ───────────────
_T_TOKENS: Tuple[str, ...] = ("finetune","fine-tuning","instruction-tune","sft","xp3","xp3mt","mtf","prompted finetuning")
_RL_TOKENS: Tuple[str, ...] = ("rlhf","reinforcement learning","dpo","ppo","reward model","preference model","human feedback","rlaif","kl penalty")

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
"""

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
        ans = _chat_json(_USAGE_SYS, text[:12000])
        ft_s = ans.get("fine_tuning","unknown"); rl_s = ans.get("rl","unknown")
        if ft_s in {"used","not_used","unknown"}: usage["fine_tuning"] = ft_s
        if rl_s in {"used","not_used","unknown"}: usage["rl"] = rl_s
    # FT-only → RL not_used 보정
    if usage.get("rl") in (None, "unknown"):
        q = _all_quotes(merged)
        if _contains_any(q, _T_TOKENS) and not _contains_any(q, _RL_TOKENS):
            usage["rl"] = "not_used"
    return usage

# ─────────────── Backstop injection (Paper URL + License) ───────────────
_PAPER_URL_RE = re.compile(r"(https?://\S*(arxiv\.org|openreview\.net|doi\.org|acm\.org|ieee\.org)\S*)", re.I)
def _inject_backstop_paper(ev: Dict[str, List[Dict[str,str]]], payload: Dict[str, Any]) -> Dict[str, List[Dict[str,str]]]:
    lbl = "1-4 (Paper)"
    if ev.get(lbl):
        return ev
    # 1) 섹션 title(URL) 사용
    for s in (payload.get("sections") or []):
        title = (s.get("title") or "").strip()
        if title.startswith("http://") or title.startswith("https://"):
            ev[lbl] = [{"source": f"sections/{title}", "quote": title}]
            return ev
    # 2) 본문에서 arxiv/doi 등의 링크 스캔
    m = _PAPER_URL_RE.search(payload.get("pdf_text") or "")
    if m:
        url = m.group(1).strip()
        ev[lbl] = [{"source":"pdf_text","quote":url}]
    return ev

# ───────────────────────── Public entry ─────────────────────────
def filter_reports_features(model: str, save: bool = True, output_dir: str | Path = ".") -> Dict[str, Any]:
    """
    Input candidates:
      - reports_fulltext_{hf}.json                (HF fetcher)
      - reports_fulltext_github_{owner_repo}.json (GH fetcher, if same model repo is known)
      - arxiv_fulltext_{hf}.json / arxiv_{hf}.json (arXiv fetcher)
    Merge → doc-level filter → dedup → quote-level filter → 2-pass extraction.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = model.replace("/", "_").lower()

    # (A) HF-based reports
    candidates: List[Path] = [output_dir / f"reports_fulltext_{base}.json"]

    # (B) GH-based reports (if GH repo identified in outdir)
    gh_meta = output_dir / f"github_{base}.json"
    if gh_meta.exists():
        try:
            gh_j = json.load(open(gh_meta, encoding="utf-8"))
            repo = (gh_j.get("repo") or gh_j.get("full_name") or "").replace("/", "_").lower()
            if repo:
                candidates.append(output_dir / f"reports_fulltext_github_{repo}.json")
        except Exception:
            pass

    # (C) arXiv papers
    candidates += [
        output_dir / f"arxiv_fulltext_{base}.json",
        output_dir / f"arxiv_{base}.json",
    ]

    # Fallback to project root if not found in output_dir
    existing: List[Path] = [p for p in candidates if p.exists()]
    if not existing:
        alt_names = [p.name for p in candidates]
        for nm in alt_names:
            if Path(nm).exists(): existing.append(Path(nm))
    if not existing:
        raise FileNotFoundError([str(c) for c in candidates])

    # Normalize into unified schema
    merged_full_texts: List[Dict[str,str]] = []
    for src in existing:
        try:
            d = json.load(open(src, encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, dict) and "full_texts" in d:
            for t in (d.get("full_texts") or []):
                if not isinstance(t, dict): continue
                url = str(t.get("arxiv_id","") or t.get("id","") or t.get("url",""))
                txt = str(t.get("full_text") or t.get("pdf_text") or "")
                if not txt.strip(): continue
                merged_full_texts.append({"arxiv_id": url, "full_text": txt})
        else:
            url = str(d.get("arxiv_id","") or d.get("id","") or "")
            txt = str(d.get("full_text") or d.get("pdf_text") or "")
            if txt.strip():
                merged_full_texts.append({"arxiv_id": url, "full_text": txt})

    # De-dup by URL key (host+path)
    def _urlkey(u:str)->str:
        m = re.match(r"^https?://([^/\s]+)(/[^?#\s]*)?", str(u).strip(), flags=re.I)
        if not m: return (u or "").strip().lower()
        host = (m.group(1) or "").lower().strip()
        path = (m.group(2) or "").rstrip("/").lower() if m.group(2) else ""
        return f"{host}{path}"

    seen_urls, dedup_by_url = set(), []
    for it in merged_full_texts:
        key = _urlkey(it.get("arxiv_id",""))
        if key in seen_urls: continue
        seen_urls.add(key); dedup_by_url.append(it)

    # Doc-level relevance
    fam = _family_tokens_from_model_id(model)
    need_filter = REPORTS_FILTER_ALWAYS or (len(dedup_by_url) > REPORTS_FILTER_THRESHOLD)
    filtered: List[Dict[str,str]] = []
    for it in dedup_by_url:
        u = str(it.get("arxiv_id",""))
        tx = str(it.get("full_text",""))
        if (not need_filter) or not fam:
            ok = True
        else:
            ok = _looks_related_doc(u, tx, fam, REPORTS_MIN_HITS_IN_TEXT)
        if ok:
            filtered.append(it)
    if need_filter:
        print(f"🔎 Relevance filter: kept {len(filtered)}/{len(dedup_by_url)} docs for '{model}'")

    # Build one payload doc
    doc_for_payload = {"full_texts": filtered}
    payload = _make_payload(doc_for_payload)
    text = _payload_text(payload)

    parts = []
    for i, grp in enumerate(ITEM_GROUPS, 1):
        try:
            ev = _collect(grp, text, model)
            # Backstops: Paper URL + License (phrase & canonical URLs)
            ev = _inject_backstop_paper(ev, payload)
            ev = _inject_license_backstop(ev, payload.get("pdf_text","") or text)
            summ = _summarize(grp, ev, model)
            part = _merge(summ, ev)
        except Exception as e:
            print(f"⚠️ Error in group {i}:", e)
            part = {}
        if save:
            fp = output_dir / f"reports_filtered_{base}_{i}.json"
            json.dump(part, open(fp,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
            print("✅ Saved group", i, ":", fp)
        parts.append(part)

    merged = _merge_all(parts)
    try:
        merged["__usage"] = _classify_usage_from_merged(merged)
    except Exception as e:
        print("⚠️ Failed to classify usage:", e)
        merged["__usage"] = {"fine_tuning":"unknown","rl":"unknown"}

    if save:
        fp = output_dir / f"reports_filtered_final_{base}.json"
        json.dump(merged, open(fp,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
        print("✅ Saved final merged:", fp)
    return merged

# ───────────────────────── CLI ─────────────────────────
if __name__ == "__main__":
    import sys
    mid="bigscience/bloomz-560m"
    if len(sys.argv)>1 and sys.argv[1]: mid=sys.argv[1]
    print("▶ Model to run:", mid)
    filter_reports_features(mid)
