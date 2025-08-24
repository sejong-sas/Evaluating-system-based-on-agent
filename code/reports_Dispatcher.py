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
#   OPENAI_MODEL_REPORTS_DISPATCHER (default "o3-mini")
#   REPORTS_FILTER_ALWAYS=1 (force doc-level relevance filter)
#   REPORTS_FILTER_THRESHOLD=20 (apply doc-filter when docs > threshold)
#   REPORTS_MIN_HITS_IN_TEXT=2 (min body hits for doc to be related)
#   REPORTS_URL_DENY_SUBSTR="blog.foo, old-version" (denylist substrings)
#   REPORTS_SECTION_CHAR_CAP / REPORTS_MAX_SECTIONS (optional caps)

import os, json, re, hashlib
from typing import Dict, List, Any, Tuple
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
_API_KEY = os.getenv("OPENAI_API_KEY")
if not _API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
_client = OpenAI(api_key=_API_KEY)

MODEL_NAME = os.getenv("OPENAI_MODEL_REPORTS_DISPATCHER", "o3-mini")

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

ITEM_GROUPS = [
    ["1-1","1-2","1-3","1-4"],
    ["1-5","1-6","2-1","2-2"],
    ["2-3","3-1","3-2","3-3"],
    ["4-1","4-2","4-3","4-4"],
]

# ─────────────── Parameters ───────────────
CHUNK_CHARS = 60_000
CHUNK_OVERLAP = 2_000
EVIDENCE_LIMIT_PER_KEY = 300
SECTION_CHAR_CAP = int(os.getenv("REPORTS_DISPATCHER_SECTION_CHAR_CAP", "0"))  # 0=unlimited
MAX_SECTIONS = int(os.getenv("REPORTS_DISPATCHER_MAX_SECTIONS", "0"))          # 0=unlimited

REPORTS_FILTER_ALWAYS = os.getenv("REPORTS_FILTER_ALWAYS", "0") == "1"
REPORTS_FILTER_THRESHOLD = int(os.getenv("REPORTS_FILTER_THRESHOLD", "20"))
REPORTS_MIN_HITS_IN_TEXT = int(os.getenv("REPORTS_MIN_HITS_IN_TEXT", "2"))
REPORTS_URL_DENY_SUBSTR = os.getenv("REPORTS_URL_DENY_SUBSTR", "")

# ─────────────── Utils ───────────────
def _js(o): return json.dumps(o, ensure_ascii=False, indent=2)

_PARA_SPLIT = re.compile(r"\n\s*\n+")

def _normalize_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()

def _dedup_texts_by_paragraph(texts: List[str]) -> str:
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
        src, qt = ev["source"].strip(), ev["quote"].strip()
        if not src or not qt: continue
        key = (src, qt)
        if key in seen: continue
        seen.add(key); out.append({"source": src, "quote": qt})
        if len(out) >= limit: break
    return out

# === Target-model guard (공통) =========================================
def _family_tokens_from_model_id(model_id: str) -> set[str]:
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
        "- Reject sentences about other models or earlier/other versions unless the TARGET is named in the same sentence.\n"
        "- If a document mixes multiple models, keep only sentences that also contain the TARGET tokens.\n"
        "- If in doubt, DROP the quote.\n"
    )

def _quote_mentions_target(q: str, model_id: str) -> bool:
    if not q: return False
    ql = q.lower().replace("–","-").replace("—","-")
    for t in _family_tokens_from_model_id(model_id):
        if len(t) >= 2 and t in ql:
            return True
    return False

# ─────────────── Prompts ───────────────
def _desc(ids): return {LABELS[i]: EVAL_DESCRIPTIONS[LABELS[i]] for i in ids}
def _skeleton(g):  return {LABELS[i]: [] for i in g}

_BASE_RECALL_SYS = """
You are an expert at extracting AI model openness evaluation information from *technical reports, papers, and blogs*.
Using only the payload (original text), return evidence for each item in the format
  [{ "source": "...", "quote": "..." }, …]
· source : e.g., [url:<...>], [sections/<url>], [pdf_text]
· quote  : a verbatim sentence copied from that section (no edits)
If there is no evidence, return an empty array [].
You must output a JSON object only.
""".strip()

_BASE_SUMMARY_SYS = """
Using the provided quotes only, write long and detailed summaries for each item.
You must output a JSON object only.
""".strip()

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

# ─────────────── GPT JSON call ───────────────
def _chat_json(sys, usr):
    r = _client.chat.completions.create(
        model=MODEL_NAME, reasoning_effort="medium",
        response_format={"type":"json_object"},
        messages=[{"role":"system","content":sys},
                  {"role":"user","content":usr}]
    )
    try:    return json.loads(r.choices[0].message.content.strip())
    except: return {}

# ─────────────── Payload builder ───────────────
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
    parts = [f"[pdf_text]\n{p.get('pdf_text','')}\n"]
    for s in p.get("sections", []):
        tag = (s.get("title") or "doc").strip()
        parts.append(f"[sections/{tag}]\n{s.get('text','')}\n")
    return "\n".join(parts)

# ─────────────── Allowed source tags & validators ───────────────
_ALLOWED_PREFIX = ("url:", "sections/", "pdf_text")
def _valid_source(src: str) -> bool:
    if not isinstance(src, str): return False
    s = src.strip().lower().strip("[]")
    return s.startswith(_ALLOWED_PREFIX)

def _filter_evidence_by_model(ev: Dict[str, List[Dict[str,str]]], model_id: str) -> Dict[str, List[Dict[str,str]]]:
    out: Dict[str, List[Dict[str,str]]] = {}
    for lbl, arr in ev.items():
        kept = []
        for e in (arr or []):
            src = (e.get("source") or "").strip()
            qt  = (e.get("quote")  or "").strip()
            if not src or not qt: continue
            if not _valid_source(src): continue
            if not _quote_mentions_target(qt, model_id): continue
            kept.append({"source": src, "quote": qt})
        out[lbl] = _dedup_evs(kept, EVIDENCE_LIMIT_PER_KEY)
    return out

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
        if re.search(rf"\b{re.escape(t)}\b", tl): hits += 1
    return hits >= max(1, min_hits)

# ─────────────── Evidence & Summary ───────────────
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
    # quote-level 모델 필터
    ev = _filter_evidence_by_model(ev, model_id)
    return ev

def _summarize(g: List[str], ev: Dict[str, List[Dict[str,str]]], model_id: str):
    quotes = {LABELS[k]: [e["quote"] for e in ev[LABELS[k]]] for k in g}
    ans = _chat_json(_BASE_SUMMARY_SYS, _summ_inst(g, model_id)+"\n=== QUOTES ===\n"+_js(quotes))
    return {LABELS[k]: ans.get(LABELS[k], "") for k in g}

def _merge(sum_, ev):
    return {lbl: (sum_.get(lbl,"") or "").strip() for lbl in sum_} | {
        f"{lbl}__evidence": ev.get(lbl, []) for lbl in sum_
    }

def _merge_all(lst):
    m={}
    for d in lst: m.update(d)
    return m

# ─────────────── RL usage classifier (with FT-only → RL not_used rule) ───────────────
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

# ─────────────── Public entry ───────────────
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
            repo = (gh_j.get("repo") or "").replace("/", "_").lower()
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

# CLI
if __name__ == "__main__":
    import sys
    mid="bigscience/bloomz-560m"
    if len(sys.argv)>1 and sys.argv[1]: mid=sys.argv[1]
    print("▶ Model to run:", mid)
    filter_reports_features(mid)
