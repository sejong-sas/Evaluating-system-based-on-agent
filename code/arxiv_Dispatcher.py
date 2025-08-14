# arxiv_Dispatcher.py
# High-Recall 2-Pass  (evidence {source, quote} → long summary)
# - store evidence as an array of objects
# - summaries must use quotes only
# - remove __evidence_sources / __sources
# - Input: arxiv_fulltext_{base}.json → if missing, fallback to arxiv_{base}.json
#          (NEW) also accept reports_fulltext_{base}.json and merge multiple sources

import os, json, re
from typing import Dict, List, Any
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

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
    LABELS["1-1"]: "All information about the availability, location, and access method of model weights, including whether anyone can download them",
    LABELS["1-2"]: "All information about whether the code for training and running the model is public, and which parts are public",
    LABELS["1-3"]: "All information about the existence/type of the license and granted rights (use, modification, distribution, commercial use)",
    LABELS["1-4"]: "All information about official papers, technical reports, blogs, and links related to the model",
    LABELS["1-5"]: "All information about the model architecture (number of layers, hyperparameters, etc.) and design details",
    LABELS["1-6"]: "All information about which tokenizer is used, its name/structure, and whether it is downloadable",
    LABELS["2-1"]: "All information about training hardware type (H100, TPU, etc.), quantity, and compute scale",
    LABELS["2-2"]: "All information about software used for training (frameworks, libraries), versions, and settings",
    LABELS["2-3"]: "All information about the existence of an accessible API (must be an API like GPT/Gemini, not a library), docs, examples, and public availability",
    LABELS["3-1"]: "All information about pre-training methodology, procedures, data flow, and hyperparameter settings",
    LABELS["3-2"]: "All information about fine-tuning methods, goals, whether data is used, and the existence of a reproducible pipeline",
    LABELS["3-3"]: "All information about whether RLHF, DPO, etc. were used, including concrete methods, procedures, and parameter settings",
    LABELS["4-1"]: "All information about types, quantities, sources, permitted use, and composition of pre-training data",
    LABELS["4-2"]: "All information about sources, composition, examples, and public availability of fine-tuning datasets",
    LABELS["4-3"]: "All information about composition, accessibility, sources, and generation of reinforcement learning datasets",
    LABELS["4-4"]: "All information about data filtering/cleaning methods, criteria used, processes, and their impact",
}

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
MODEL_NAME = os.getenv("OPENAI_MODEL_ARXIV_DISPATCHER", "o3-mini")

# ─────────── Utils ───────────
def _js(o): return json.dumps(o, ensure_ascii=False, indent=2)

def _chunk(s, n, ov):
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
You are a classifier. Based only on the input text (quotes/summaries), decide whether this model actually used
Fine-tuning / Reinforcement Learning.
JSON only:
{ "fine_tuning": "used|not_used|unknown", "rl": "used|not_used|unknown" }
"""

def _classify_usage_from_merged(merged: dict) -> dict:  # whether RL or fine-tuning were used
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

def _desc(ids): return {LABELS[i]: EVAL_DESCRIPTIONS[LABELS[i]] for i in ids}  # description (token-saving)
def _recall_inst(g): return "Items in this group:\n"+_js(_desc(g))
def _summ_inst(g):   return "Items in this group:\n"+_js(_desc(g))

# ─────────── GPT JSON call ───────────
def _chat_json(sys, usr):
    r = _client.chat.completions.create(
        model=MODEL_NAME, reasoning_effort="medium",
        response_format={"type":"json_object"},
        messages=[{"role":"system","content":sys},
                  {"role":"user","content":usr}]
    )
    try:    return json.loads(r.choices[0].message.content.strip())
    except: return {}

# ─────────── Build payload ───────────
def _make_payload(d):
    # basic fields
    pid   = d.get("arxiv_id") or d.get("id") or ""
    title = d.get("title") or d.get("meta",{}).get("title") or ""
    abstr = d.get("abstract") or ""
    cats  = d.get("categories") or ""
    lic   = d.get("license") or ""
    auth  = d.get("authors")  or ""
    body  = d.get("pdf_text") or d.get("fulltext") or d.get("body") or ""

    # section text
    secs = []
    for s in (d.get("sections") or [])[:50]:
        if isinstance(s, dict):
            secs.append({"title": str(s.get("title",""))[:300],
                         "text":  str(s.get("text",""))[:20_000]})
    if not secs and isinstance(d.get("section_texts"), dict):
        for k,v in list(d["section_texts"].items())[:50]:
            secs.append({"title": str(k)[:300], "text": str(v)[:20_000]})

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
        "pdf_text": str(body)[:240_000],
        "sections": secs,
        "bib": bib,
    }

def _payload_text(p):
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
_ALLOWED = ("arxiv_id","title","abstract","pdf_text","sections/",
            "categories","license","authors","bib")

def _valid(src): return isinstance(src,str) and src.startswith(_ALLOWED)

def _collect(g, text):
    ev = {LABELS[k]: [] for k in g}
    for ch in _chunk(text, CHUNK_CHARS, CHUNK_OVERLAP):
        ans = _chat_json(_BASE_RECALL_SYS, _recall_inst(g)+"\n=== PAYLOAD ===\n"+ch)
        for k in g:
            ev[LABELS[k]].extend(ans.get(LABELS[k], []))
    for k in ev:
        ev[k] = _dedup_evs(ev[k], EVIDENCE_LIMIT_PER_KEY)
    return ev

# ─────────── Summarize ───────────
def _summarize(g, ev):
    quotes = {LABELS[k]: [e["quote"] for e in ev[LABELS[k]]] for k in g}
    ans = _chat_json(_BASE_SUMMARY_SYS, _summ_inst(g)+"\n=== QUOTES ===\n"+_js(quotes))
    return {LABELS[k]: ans.get(LABELS[k], "") for k in g}

# ─────────── Merge ───────────
def _merge(sum_, ev):
    return {lbl: sum_[lbl].strip() for lbl in sum_} | {
        f"{lbl}__evidence": ev.get(lbl, []) for lbl in sum_
    }

def _merge_all(lst):
    m={}
    for d in lst: m.update(d)
    return m

# ─────────── Public function ───────────
def filter_arxiv_features(model, save: bool = True, output_dir: str | Path = "."):
    base = model.replace("/", "_").lower()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Input: prefer outdir → fallback to project root (NOW supports reports_fulltext_ and multi-merge)
    candidates = [
        output_dir / f"arxiv_fulltext_{base}.json",
        output_dir / f"arxiv_{base}.json",
        output_dir / f"reports_fulltext_{base}.json",  # NEW: technical reports (HF/GH harvested)
    ]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        # project root fallbacks
        fallbacks = [
            f"arxiv_fulltext_{base}.json",
            f"arxiv_{base}.json",
            f"reports_fulltext_{base}.json",
        ]
        alt = [Path(p) for p in fallbacks if Path(p).exists()]
        if not alt:
            raise FileNotFoundError([str(c) for c in candidates])
        existing = alt

    # Load & merge all inputs into a single "doc_for_payload" shape
    docs = []
    for src in existing:
        docs.append(json.load(open(src, encoding="utf-8")))

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
            # older single-doc schema → map to unified
            merged_full_texts.append({
                "arxiv_id": str(ax.get("arxiv_id", "")).strip(),
                "full_text": str(ax.get("full_text") or ax.get("pdf_text") or "")
            })

    doc_for_payload = {
        "arxiv_id": ";".join([x["arxiv_id"] for x in merged_full_texts if x.get("arxiv_id")]),
        "title": "",
        "abstract": "",
        "categories": "",
        "license": "",
        "authors": "",
        # Key point: map merged text to pdf_text (to match downstream expectations)
        "pdf_text": "\n\n".join([x["full_text"] for x in merged_full_texts if x.get("full_text")]),
        "sections": [],
        "bib": "",
    }

    # 3) Process by group
    parts = []
    for i, grp in enumerate(ITEM_GROUPS, 1):
        try:
            pay = _make_payload(doc_for_payload)
            text = _payload_text(pay)
            ev = _collect(grp, text)
            summ = _summarize(grp, ev)
            part = _merge(summ, ev)
        except Exception as e:
            print(f"⚠️ Error in group {i}:", e)
            part = {}

        if save:
            fp = output_dir / f"arxiv_filtered_{base}_{i}.json"
            json.dump(part, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            print("✅ Saved group", i, ":", fp)
        parts.append(part)

    # 4) Save final merge
    merged = _merge_all(parts)

    try:
        merged["__usage"] = _classify_usage_from_merged(merged)
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
