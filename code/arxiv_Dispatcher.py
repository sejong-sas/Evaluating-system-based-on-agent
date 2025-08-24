# arxiv_Dispatcher.py
# High-Recall 2-Pass  (evidence {source, quote} → long summary)
# - store evidence as an array of objects
# - summaries must use quotes only
# - remove __evidence_sources / __sources
# - Input: arxiv_fulltext_{base}.json → if missing, fallback to arxiv_{base}.json
# - NOTE: This dispatcher intentionally ignores reports_fulltext_* (handled by Reports Dispatcher)

import os, json, re, hashlib
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

# ─────────── Dedup helpers ───────────
_PARA_SPLIT = re.compile(r"\n\s*\n+")  # blank-line separated paragraphs

def _normalize_for_hash(text: str) -> str:
    """Normalize text for hashing so that trivial whitespace/casing differences do not defeat deduplication."""
    return re.sub(r"\s+", " ", text).strip().lower()

def _dedup_texts_by_paragraph(texts: List[str], min_keep_len: int = 0) -> str:
    """
    Concatenate multiple documents while removing duplicate paragraphs.
    Keeps original order of first occurrences.
    """
    seen: set[str] = set()
    out: List[str] = []
    for doc in texts:
        if not doc:
            continue
        paras = _PARA_SPLIT.split(doc.strip()) if _PARA_SPLIT.search(doc) else [doc.strip()]
        for p in paras:
            p = p.strip()
            if not p:
                continue
            if min_keep_len and len(p) < min_keep_len:
                key = hashlib.sha1(_normalize_for_hash(p).encode("utf-8")).hexdigest()
                if key in seen:
                    continue
                seen.add(key); out.append(p); continue
            key = hashlib.sha1(_normalize_for_hash(p).encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
    return "\n\n".join(out)

# ─────────────────── Environment ───────────────────
load_dotenv()
key = os.getenv("OPENAI_API_KEY")
if not key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
_client = OpenAI(api_key=key)

# ─────────── 16 item labels ───────────
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

# === Target-model guard ==================================================
def _family_tokens_from_model_id(model_id: str) -> set[str]:
    """
    Extract stable tokens from the model id for family/version matching.
    Examples:
      "google/gemma-3-27b-it" -> {"gemma", "gemma3", "3", "27b"}
      "meta-llama/Llama-3.1-8B" -> {"llama", "llama3", "3.1", "31", "8b"}
    Ignore overly generic tokens like 'base', 'it', 'chat', 'model'.
    """
    name = (model_id or "").split("/", 1)[-1].lower()
    raw = re.split(r"[^a-z0-9.]+", name)
    base: set[str] = set()
    for tt in (t.strip() for t in raw):
        if not tt: 
            continue
        if tt in {"base","it","instruct","chat","hf","model"}:
            continue
        if len(tt) >= 2:
            base.add(tt)
        m = re.match(r"([a-z]+)(\d+(?:\.\d+)*)$", tt)  # llama3 / llama3.1 → (llama, 3/3.1)
        if m:
            base.add(m.group(1))
            base.add(m.group(1)+m.group(2).replace(".",""))
            base.add(m.group(2))              # "3.1"
            base.add(m.group(2).replace(".",""))  # "31"
    # joined/nodigit variants
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
        "- Reject sentences about other models or earlier/other versions unless the TARGET is named in the same sentence.\n"
        "- If a document mixes multiple models, keep only sentences that also contain the TARGET tokens.\n"
        "- If in doubt, DROP the quote.\n"
    )

def _quote_mentions_target(q: str, model_id: str) -> bool:
    if not q: return False
    ql = q.lower()
    # normalize dashes to match fine–tuning/fine—tuning
    ql = ql.replace("–", "-").replace("—", "-")
    for t in _family_tokens_from_model_id(model_id):
        if len(t) >= 2 and t in ql:
            return True
    return False

# ─────────── 4 groups ───────────
ITEM_GROUPS = [
    ["1-1","1-2","1-3","1-4"],
    ["1-5","1-6","2-1","2-2"],
    ["2-3","3-1","3-2","3-3"],
    ["4-1","4-2","4-3","4-4"],
]

# ─────────── Parameters ───────────
CHUNK_CHARS = 60_000
CHUNK_OVERLAP = 2_000
EVIDENCE_LIMIT_PER_KEY = 300
MODEL_NAME = os.getenv("OPENAI_MODEL_ARXIV_DISPATCHER", "o3-mini")  # change if not accessible

# Section limits (0 = unlimited)
SECTION_CHAR_CAP = int(os.getenv("ARXIV_DISPATCHER_SECTION_CHAR_CAP", "0"))
MAX_SECTIONS = int(os.getenv("ARXIV_DISPATCHER_MAX_SECTIONS", "0"))

# Relevance filter knobs
ARXIV_FILTER_THRESHOLD = int(os.getenv("ARXIV_FILTER_THRESHOLD", "20"))  # apply filter if docs > threshold
ARXIV_FILTER_ALWAYS = os.getenv("ARXIV_FILTER_ALWAYS", "0") == "1"       # force filtering always
ARXIV_MIN_HITS_IN_TEXT = int(os.getenv("ARXIV_MIN_HITS_IN_TEXT", "2"))   # body hits threshold
ARXIV_FAMILY_HINTS = os.getenv("ARXIV_FAMILY_HINTS", "")                 # extra tokens (comma/space-separated)
ARXIV_URL_DENY_SUBSTR = os.getenv("ARXIV_URL_DENY_SUBSTR", "")           # denylist substrings in URL

# ─────────── Utils ───────────
def _js(o): return json.dumps(o, ensure_ascii=False, indent=2)

def _chunk(s: str, n: int, ov: int) -> List[str]:
    out, L, i = [], len(s), 0
    while i < L:
        end = min(i + n, L)
        out.append(s[i:end])
        if end == L: break
        i = end - ov if end - ov > i else end
    return out

def _dedup_evs(evs: List[Dict[str,str]], limit:int):
    seen, out = set(), []
    for ev in evs:
        if not (isinstance(ev, dict)
                and isinstance(ev.get("source"), str)
                and isinstance(ev.get("quote"), str)):
            continue
        src, qt = ev["source"].strip(), ev["quote"].strip()
        if not src or not qt: continue
        key = (src, qt)
        if key in seen: continue
        seen.add(key); out.append({"source": src, "quote": qt})
        if len(out) >= limit: break
    return out

# ─────────── Prompts ───────────
_BASE_RECALL_SYS = """
You are an expert at extracting AI model openness evaluation information from arXiv source text.
Using only the payload (original text), return evidence for each item in the format
  [{ "source": "...", "quote": "..." }, …]
· source : e.g., [title], [abstract], [pdf_text], [sections/Introduction], …
· quote  : a verbatim sentence copied from that section
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

def _desc(ids): return {LABELS[i]: EVAL_DESCRIPTIONS[LABELS[i]] for i in ids}

def _skeleton(g):  # exact keys to force model output shape
    return {LABELS[i]: [] for i in g}

def _recall_inst(g: List[str], model_id: str) -> str:
    return (
        _model_guard_text(model_id) +
        "\nItems in this group:\n" + _js(_desc(g)) +
        "\nReturn a JSON object with EXACTLY these keys (arrays of {source,quote}):\n" +
        _js(_skeleton(g))
    )

def _summ_inst(g: List[str], model_id: str) -> str:
    return (
        _model_guard_text(model_id) +
        "\nItems in this group:\n" + _js(_desc(g)) +
        "\nReturn a JSON object with EXACTLY these keys (string summaries):\n" +
        _js({LABELS[i]: "" for i in g}) +
        "\nUse ONLY the provided quotes."
    )

# ─────────── GPT JSON call ───────────
def _chat_json(sys: str, usr: str):
    r = _client.chat.completions.create(
        model=MODEL_NAME, reasoning_effort="medium",
        response_format={"type":"json_object"},
        messages=[{"role":"system","content":sys},
                  {"role":"user","content":usr}]
    )
    try:
        return json.loads(r.choices[0].message.content.strip())
    except Exception:
        return {}

# ─────────── Build payload ───────────
def _make_payload(d: Dict[str, Any]) -> Dict[str, Any]:
    pid   = d.get("arxiv_id") or d.get("id") or ""
    title = d.get("title") or d.get("meta",{}).get("title") or ""
    abstr = d.get("abstract") or ""
    cats  = d.get("categories") or ""
    lic   = d.get("license") or ""
    auth  = d.get("authors")  or ""
    body  = d.get("pdf_text") or d.get("fulltext") or d.get("body") or ""

    # section text
    secs: List[Dict[str,str]] = []
    src_secs = (d.get("sections") or [])
    if MAX_SECTIONS > 0:
        src_secs = src_secs[:MAX_SECTIONS]
    for s in src_secs:
        if isinstance(s, dict):
            title_str = str(s.get("title",""))[:300]
            text_str = str(s.get("text",""))
            if SECTION_CHAR_CAP > 0:
                text_str = text_str[:SECTION_CHAR_CAP]
            secs.append({"title": title_str, "text": text_str})

    if not secs and isinstance(d.get("section_texts"), dict):
        items = list(d["section_texts"].items())
        if MAX_SECTIONS > 0:
            items = items[:MAX_SECTIONS]
        for k, v in items:
            text_str = str(v)
            if SECTION_CHAR_CAP > 0:
                text_str = text_str[:SECTION_CHAR_CAP]
            secs.append({"title": str(k)[:300], "text": text_str})

    # bibliography
    bib = d.get("references") or d.get("bib") or ""
    if isinstance(bib, list):
        bib = "\n".join(str(x) for x in bib[:3000])
    else:
        bib = str(bib)[:100_000]

    return {
        "arxiv_id": pid,
        "title":    str(title)[:2000],
        "abstract": str(abstr)[:120_000],
        "categories": str(cats)[:5000],
        "license": str(lic)[:5000],
        "authors": str(auth)[:8000],
        "pdf_text": str(body)[:],  # no truncation
        "sections": secs,
        "bib": bib,
    }

def _payload_text(p: Dict[str, Any]) -> str:
    parts = [
        f"[arxiv_id]\n{p['arxiv_id']}\n",
        f"[title]\n{p['title']}\n",
        f"[authors]\n{p['authors']}\n",
        f"[categories]\n{p['categories']}\n",
        f"[license]\n{p['license']}\n",
        f"[abstract]\n{p['abstract']}\n",
        f"[pdf_text]\n{p['pdf_text']}\n"
    ]
    for s in p["sections"]:
        parts.append(f"[sections/{s['title'] or 'Section'}]\n{s['text']}\n")
    if p["bib"]:
        parts.append("[bib]\n"+p["bib"]+"\n")
    return "\n".join(parts)

# ─────────── Collect evidence ───────────
_ALLOWED_PREFIXES = ("arxiv_id","title","abstract","pdf_text","sections/","categories","license","authors","bib")
_EXTRA_OK_PREFIXES = ("section","sec.","appendix","table","figure","fig.")

def _valid_source(src: str) -> bool:
    if not isinstance(src, str):
        return False
    s = src.strip().lower().strip("[]")  # allow bracketed tags
    return s.startswith(_ALLOWED_PREFIXES) or s.startswith(_EXTRA_OK_PREFIXES)

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
        out[lbl] = _dedup_evs(kept, EVIDENCE_LIMIT_PER_KEY)
    return out

def _rebalance_evidence(ev: Dict[str, List[Dict[str,str]]]) -> Dict[str, List[Dict[str,str]]]:
    """If 4-2 is empty but 3-2 contains dataset-ish quotes, copy a subset."""
    ft_lbl = "3-2 (Fine-tuning)"
    fd_lbl = "4-2 (Fine-tuning Data)"
    if (not ev.get(fd_lbl)) and ev.get(ft_lbl):
        kws = r"(dataset|corpus|xP3(?:mt)?|p3\b|language distribution|languages|split|examples|released|Flores200|ROOTS|license|public links?)"
        moved = [e for e in ev[ft_lbl] if isinstance(e, dict) and re.search(kws, e.get("quote",""), re.I)]
        if moved:
            ev[fd_lbl] = _dedup_evs(moved, EVIDENCE_LIMIT_PER_KEY)
    return ev

def _collect(g: List[str], text: str, model_id: str) -> Dict[str, List[Dict[str,str]]]:
    ev = {LABELS[k]: [] for k in g}
    for ch in _chunk(text, CHUNK_CHARS, CHUNK_OVERLAP):
        ans = _chat_json(_BASE_RECALL_SYS, _recall_inst(g, model_id) + "\n=== PAYLOAD ===\n" + ch)
        if not isinstance(ans, dict):
            ans = {}
        if ans:
            print("🔎 recall keys:", list(ans.keys())[:16])
        for k in g:
            lbl = LABELS[k]
            arr = ans.get(lbl, [])
            if isinstance(arr, list):
                ev[lbl].extend(arr)

    raw_counts = {lbl: len(ev.get(lbl) or []) for lbl in ev}
    ev = _filter_evidence_by_model(ev, model_id)
    kept_counts = {lbl: len(ev.get(lbl) or []) for lbl in ev}
    print("evidence counts before/after model-guard:", {"raw": raw_counts, "kept": kept_counts})

    # 재배치 휴리스틱 (특히 3-2 → 4-2)
    return _rebalance_evidence(ev)

# ─────────── Summarize ───────────
def _summarize(g: List[str], ev: Dict[str, List[Dict[str,str]]], model_id: str) -> Dict[str, str]:
    lbls = [LABELS[k] for k in g]
    quotes = {lbl: [(e.get("quote") or "") for e in (ev.get(lbl) or [])] for lbl in lbls}
    ans = _chat_json(_BASE_SUMMARY_SYS, _summ_inst(g, model_id) + "\n=== QUOTES ===\n" + _js(quotes))
    return {lbl: (ans.get(lbl, "") or "") for lbl in lbls}

# ─────────── Merge ───────────
def _merge(sum_: Dict[str,str], ev: Dict[str, List[Dict[str,str]]]) -> Dict[str, Any]:
    return {lbl: (sum_.get(lbl,"") or "").strip() for lbl in sum_} | {
        f"{lbl}__evidence": ev.get(lbl, []) for lbl in sum_
    }

def _merge_all(lst: List[Dict[str, Any]]) -> Dict[str, Any]:
    m: Dict[str, Any] = {}
    for d in lst: m.update(d)
    return m

# ─────────── Relevance filtering helpers (doc-level) ───────────
def _tok(s: str) -> List[str]:
    s = re.sub(r"[^a-z0-9.]+", " ", (s or "").lower())
    return [t for t in s.split() if t]

def _deny_url(u: str) -> bool:
    if not ARXIV_URL_DENY_SUBSTR.strip():
        return False
    u = (u or "").lower()
    for sub in re.split(r"[,\s]+", ARXIV_URL_DENY_SUBSTR.lower()):
        sub = sub.strip()
        if sub and sub in u:
            return True
    return False

def _looks_related_doc(url: str, text: str, fam_tokens: set[str], min_hits: int = 2) -> bool:
    """
    Decide if a document is related to the current model/family.
    Signals:
      - URL contains any family token (strong)
      - Body text contains >= min_hits occurrences of any family token
    """
    ul = (url or "").lower()
    tl = (text or "").lower()
    if _deny_url(ul):
        return False
    for t in fam_tokens:
        if len(t) >= 3 and t in ul:
            return True
    hits = 0
    for t in fam_tokens:
        if not t: 
            continue
        hits += tl.count(t)
        if re.search(rf"\b{re.escape(t)}\b", tl):
            hits += 1
    return hits >= max(1, min_hits)

# ─────────── RL not-used rule (general) ───────────
_T_TOKENS: Tuple[str, ...] = ("finetune","fine-tuning","instruction-tune","sft","xp3","xp3mt","mtf","prompted finetuning")
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

# ─────────── Public function ───────────
def filter_arxiv_features(model: str, save: bool = True, output_dir: str | Path = "."):
    base = model.replace("/", "_").lower()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Input: prefer outdir → fallback to project root (reports_* intentionally NOT included)
    candidates = [
        output_dir / f"arxiv_fulltext_{base}.json",
        output_dir / f"arxiv_{base}.json",
    ]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        fallbacks = [
            f"arxiv_fulltext_{base}.json",
            f"arxiv_{base}.json",
        ]
        alt = [Path(p) for p in fallbacks if Path(p).exists()]
        if not alt:
            raise FileNotFoundError([str(c) for c in candidates])
        existing = alt

    # Load & merge all inputs into a single unified list
    docs: List[Dict[str, Any]] = []
    for src in existing:
        try:
            docs.append(json.load(open(src, encoding="utf-8")))
        except Exception:
            try:
                raw = open(src, encoding="utf-8").read()
                if raw.strip():
                    docs.append({"full_texts": [{"arxiv_id": str(src), "full_text": raw}]})
            except Exception:
                pass

    # Normalization: unify to {"full_texts":[{"arxiv_id" (or URL), "full_text"}...]}
    merged_full_texts: List[Dict[str, str]] = []
    for ax in docs:
        if isinstance(ax, dict) and "full_texts" in ax:
            for t in (ax.get("full_texts") or []):
                if isinstance(t, dict):
                    merged_full_texts.append({
                        "arxiv_id": str(t.get("arxiv_id", "")).strip(),
                        "full_text": str(t.get("full_text") or t.get("pdf_text") or "")
                    })
        else:
            if isinstance(ax, dict):
                merged_full_texts.append({
                    "arxiv_id": str(ax.get("arxiv_id", "")).strip(),
                    "full_text": str(ax.get("full_text") or ax.get("pdf_text") or "")
                })

    # 2) Model relevance filter (doc-level)
    fam = _family_tokens_from_model_id(model)
    need_filter = ARXIV_FILTER_ALWAYS or (len(merged_full_texts) > ARXIV_FILTER_THRESHOLD)
    if need_filter and fam:
        before = len(merged_full_texts)
        filtered = []
        for it in merged_full_texts:
            u = str(it.get("arxiv_id",""))
            t = str(it.get("full_text",""))
            if _looks_related_doc(u, t, fam, ARXIV_MIN_HITS_IN_TEXT):
                filtered.append(it)
        if filtered:
            merged_full_texts = filtered
        print(f"🔎 Relevance filter: kept {len(merged_full_texts)}/{before} docs for '{model}'")

    # 3) Collect all raw texts first
    _all_texts = [x["full_text"] for x in merged_full_texts if x.get("full_text")]

    # 4) Deduplicate by paragraph while preserving first-seen order
    _dedup_pdf_text = _dedup_texts_by_paragraph(_all_texts, min_keep_len=0)

    # 5) Build payload doc
    doc_for_payload = {
        "arxiv_id": ";".join([x["arxiv_id"] for x in merged_full_texts if x.get("arxiv_id")]),
        "title": "",
        "abstract": "",
        "categories": "",
        "license": "",
        "authors": "",
        "pdf_text": _dedup_pdf_text,
        "sections": [
            {
                "title": (x.get("arxiv_id") or "doc")[:300],
                "text":  (x.get("full_text") or "") if SECTION_CHAR_CAP == 0
                        else (x.get("full_text") or "")[:SECTION_CHAR_CAP]
            }
            for x in merged_full_texts if x.get("full_text")
        ],
        "bib": "",
    }

    # 6) Process by group
    parts: List[Dict[str, Any]] = []
    for i, grp in enumerate(ITEM_GROUPS, 1):
        try:
            pay = _make_payload(doc_for_payload)
            text = _payload_text(pay)
            ev   = _collect(grp, text, model)
            summ = _summarize(grp, ev, model)
            part = _merge(summ, ev)
        except Exception as e:
            print(f"⚠️ Error in group {i}:", e)
            part = {}

        if save:
            fp = output_dir / f"arxiv_filtered_{base}_{i}.json"
            json.dump(part, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            print("✅ Saved group", i, ":", fp)
        parts.append(part)

    # 7) Save final merge
    merged = _merge_all(parts)

    try:
        # initial GPT-based classification
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

        # general FT-only → RL not_used rule
        if usage.get("rl") in (None, "unknown"):
            if _rule_infer_rl_not_used(merged):
                usage["rl"] = "not_used"

        merged["__usage"] = usage
    except Exception as e:
        print("⚠️ Failed to classify usage:", e)
        merged["__usage"] = {"fine_tuning":"unknown","rl":"unknown"}

    if save:
        fp = output_dir / f"arxiv_filtered_final_{base}.json"
        json.dump(merged, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("✅ Saved final merged:", fp)
    return merged

# ─────────── CLI ───────────
if __name__=="__main__":
    import sys
    mid="bigscience/bloomz-560m"
    if len(sys.argv)>1 and sys.argv[1]: mid=sys.argv[1]
    print("▶ Model to run:", mid)
    filter_arxiv_features(mid)
