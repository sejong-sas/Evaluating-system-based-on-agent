# reports_Dispatcher.py
# Purpose:
#   - Merge and deduplicate "report-ish" texts gathered by fetchers
#     (HF: reports_fulltext_{hf}.json, GH: reports_fulltext_github_{owner_repo}.json)
#     PLUS arXiv fetcher outputs (arxiv_fulltext_{hf}.json / arxiv_{hf}.json).
#   - Filter out irrelevant sources to the given model.
#   - Run a 2-pass (evidence → summary) extraction for the 16 openness items.
#   - Save group outputs and a final merged JSON:
#       reports_filtered_{base}_{1..4}.json, reports_filtered_final_{base}.json
#
# Notes:
#   - Keep prompts and overall style consistent with other dispatchers.
#   - Minimal external changes to the existing pipeline (see step 3 and 4).
#
# Env:
#   OPENAI_API_KEY (required)
#   OPENAI_MODEL_REPORTS_DISPATCHER (optional, default "o3-mini")

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

# ─────────────── 16 evaluation items (same as others) ───────────────
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
    LABELS["2-2"]: "All information about software used for training (frameworks, libraries), versions, and settings",
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

def _tok(s: str) -> List[str]:
    s = re.sub(r"[^a-z0-9]+", " ", (s or "").lower())
    return [t for t in s.split() if t]

def _related_to_model(url: str, text: str, hf_id: str) -> bool:
    """
    Heuristic filter:
      - Always accept arXiv entries from arxiv_fetcher.
      - Accept if URL contains model tokens (name/family).
      - OR if body contains model tokens more than once.
    """
    model = hf_id.split("/",1)[1] if "/" in hf_id else hf_id
    toks = set(_tok(model))
    if not toks: return True
    s_url = (url or "").lower()
    if any(t in s_url for t in toks):  # URL hint
        return True
    body = (text or "").lower()
    hit = sum(1 for t in toks if body.count(t) >= 2)  # at least 2 mentions of any token
    return hit > 0

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

def _desc(ids): return {LABELS[i]: EVAL_DESCRIPTIONS[LABELS[i]] for i in ids}
def _recall_inst(g): return "Items in this group:\n"+_js(_desc(g))
def _summ_inst(g):   return "Items in this group:\n"+_js(_desc(g))

# ─────────────── Prompts ───────────────
_BASE_RECALL_SYS = """
You are an expert at extracting AI model openness evaluation information from *technical reports and papers*.
Using only the payload (original text), return evidence for each item in the format
  [{ "source": "...", "quote": "..." }, …]
· source : e.g., [url:<...>], [sections/<url>], [pdf_text]
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

Answer JSON only:
{ "fine_tuning": "used|not_used|unknown", "rl": "used|not_used|unknown" }
"""


# ─────────────── Payload builder ───────────────
def _make_payload(doc: Dict[str, Any]) -> Dict[str, Any]:
    # Flatten into a big text with per-source sections
    secs = []
    for it in (doc.get("full_texts") or []):
        u = str(it.get("arxiv_id") or it.get("id") or it.get("url") or "")[:2000]
        tx = str(it.get("full_text") or it.get("pdf_text") or "")
        if not tx: continue
        if SECTION_CHAR_CAP > 0:
            tx = tx[:SECTION_CHAR_CAP]
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

# ─────────────── Core steps ───────────────
def _collect(g, text):
    ev = {LABELS[k]: [] for k in g}
    # chunked recall
    i = 0; n = len(text)
    while i < n:
        end = min(i+CHUNK_CHARS, n)
        ch = text[i:end]
        i = end - CHUNK_OVERLAP if end - CHUNK_OVERLAP > i else end
        ans = _chat_json(_BASE_RECALL_SYS, _recall_inst(g)+"\n=== PAYLOAD ===\n"+ch)
        for k in g:
            arr = ans.get(LABELS[k], [])
            if isinstance(arr, list):
                ev[LABELS[k]].extend(arr)
    for k in ev:
        ev[k] = _dedup_evs(ev[k], EVIDENCE_LIMIT_PER_KEY)
    return ev

def _summarize(g, ev):
    quotes = {LABELS[k]: [e["quote"] for e in ev[LABELS[k]]] for k in g}
    ans = _chat_json(_BASE_SUMMARY_SYS, _summ_inst(g)+"\n=== QUOTES ===\n"+_js(quotes))
    return {LABELS[k]: ans.get(LABELS[k], "") for k in g}

def _merge(sum_, ev):
    return {lbl: (sum_.get(lbl,"") or "").strip() for lbl in sum_} | {
        f"{lbl}__evidence": ev.get(lbl, []) for lbl in sum_
    }

def _merge_all(lst):
    m={}
    for d in lst: m.update(d)
    return m

def _classify_usage_from_merged(merged: dict) -> dict:
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

# ─────────────── Public entry ───────────────
def filter_reports_features(model: str, save: bool = True, output_dir: str | Path = ".") -> Dict[str, Any]:
    """
    Input candidates:
      - reports_fulltext_{hf}.json                (HF fetcher)
      - reports_fulltext_github_{owner_repo}.json (GH fetcher, if same model repo is known)
      - arxiv_fulltext_{hf}.json / arxiv_{hf}.json (arXiv fetcher)
    Merge → dedup → filter by model relevance → 2-pass extraction.
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

    # (C) arXiv papers (still considered "reports" corpus for this dispatcher)
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

    # Normalize into unified schema: {"full_texts":[{"arxiv_id":<url/id>,"full_text":<text>}...]}
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
            # older single-doc schema
            url = str(d.get("arxiv_id","") or d.get("id","") or "")
            txt = str(d.get("full_text") or d.get("pdf_text") or "")
            if txt.strip():
                merged_full_texts.append({"arxiv_id": url, "full_text": txt})

    # De-dup by URL key (host+path) first
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

    # Model relevance filter (keep arXiv items by default)
    filtered: List[Dict[str,str]] = []
    for it in dedup_by_url:
        u = str(it.get("arxiv_id",""))
        tx = str(it.get("full_text",""))
        if u.startswith("http"):  # external url
            if _related_to_model(u, tx, model):
                filtered.append(it)
        else:
            # arXiv IDs or unknown IDs (assume relevant)
            filtered.append(it)

    # Build one payload doc with original sections retained
    doc_for_payload = {"full_texts": filtered}
    payload = _make_payload(doc_for_payload)
    text = _payload_text(payload)

    parts = []
    for i, grp in enumerate(ITEM_GROUPS, 1):
        try:
            ev = _collect(grp, text)
            summ = _summarize(grp, ev)
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
