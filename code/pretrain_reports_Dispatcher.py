# pretrain_reports_Dispatcher.py
# Collect only pre-training related reports/blog links from base HF/GH README,
# summarize 3-1 (Pre-training) & 4-1 (Pre-training Data) with quotes-only,
# and output a single json: pretrain_reports_{base}.json

import os, re, json
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
_API = os.getenv("OPENAI_API_KEY")
if not _API:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
_client = OpenAI(api_key=_API)

MODEL_NAME = os.getenv("OPENAI_MODEL_PRETRAIN_REPORTS", "o3-mini")
USER_AGENT = {"User-Agent": "Mozilla/5.0 (pretrain-reports-dispatcher)"}

# ---------- helpers ----------
def _js(o) -> str:
    return json.dumps(o, ensure_ascii=False, indent=2)

def _hf_card_and_readme(hf_id: str, max_len: int = 120_000) -> str:
    txt = ""
    try:
        r = requests.get(f"https://huggingface.co/api/models/{hf_id}?full=true", timeout=15)
        card = (r.json() or {}).get("cardData", {}) or {}
        txt += (card.get("content") or "")[:max_len]
    except Exception:
        pass
    for br in ("main", "master"):
        try:
            rr = requests.get(f"https://huggingface.co/{hf_id}/raw/{br}/README.md", timeout=12)
            if rr.status_code == 200:
                txt += "\n\n" + rr.text[:max_len]
                break
        except Exception:
            pass
    return txt

def _gh_readme(repo: str, max_len: int = 160_000) -> str:
    for br in ("main", "master"):
        try:
            url = f"https://raw.githubusercontent.com/{repo}/{br}/README.md"
            r = requests.get(url, timeout=12, headers=USER_AGENT)
            if r.status_code == 200:
                return r.text[:max_len]
        except Exception:
            pass
    return ""

_LINK_PAT = re.compile(r"https?://[^\s)>\"]+", re.IGNORECASE)

def _looks_report(u: str) -> bool:
    ul = (u or "").lower()
    return (
        ul.endswith(".pdf")
        or "technical-report" in ul or "tech-report" in ul
        or "whitepaper" in ul or "white-paper" in ul
        or "/blog" in ul or "blog." in ul
        or "/research" in ul or "research." in ul
        or "/docs" in ul or "docs." in ul
        or "/paper" in ul or "arxiv.org" in ul
    )

def _fetch_pdf(url: str) -> str:
    import fitz
    r = requests.get(url, timeout=25, headers=USER_AGENT)
    r.raise_for_status()
    with fitz.open(stream=r.content, filetype="pdf") as doc:
        return "\n".join(p.get_text() for p in doc)

def _fetch_html(url: str, max_len: int = 800_000) -> str:
    r = requests.get(url, timeout=20, headers=USER_AGENT)
    r.raise_for_status()
    h = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", r.text)
    t = re.sub(r"(?is)<[^>]+>", " ", h)
    t = re.sub(r"\s+", " ", t)
    return t[:max_len]

def _collect_reports_text(hf_id: str, gh_repo: Optional[str]) -> List[Dict[str, str]]:
    """Return list of {'url','text','source'} harvested from HF/GH READMEs."""
    corpus = []

    # HF README links
    hf_md = _hf_card_and_readme(hf_id)
    if hf_md:
        urls = list(dict.fromkeys(_LINK_PAT.findall(hf_md)))
        for u in urls:
            if _looks_report(u):
                try:
                    text = _fetch_pdf(u) if u.lower().endswith(".pdf") else _fetch_html(u)
                    if text.strip():
                        corpus.append({"url": u, "text": text, "source": "hf_readme_link"})
                except Exception:
                    continue

    # GH README links
    if gh_repo:
        gh_md = _gh_readme(gh_repo)
        if gh_md:
            urls = list(dict.fromkeys(_LINK_PAT.findall(gh_md)))
            for u in urls:
                if _looks_report(u):
                    try:
                        text = _fetch_pdf(u) if u.lower().endswith(".pdf") else _fetch_html(u)
                        if text.strip():
                            corpus.append({"url": u, "text": text, "source": "gh_readme_link"})
                    except Exception:
                        continue
    return corpus

# ---------- prompts ----------
_SYS_RECALL = """
You extract EVIDENCE about *pre-training* only (methodology and data) for an AI model.
Return a JSON object with two arrays of quotes:

{
  "3-1 (Pre-training)": [{"source":"<short id>","quote":"<verbatim sentence>"}],
  "4-1 (Pre-training Data)": [{"source":"<short id>","quote":"<verbatim sentence>"}]
}

Rules:
- Use ONLY the provided payload articles (each article has url, source and text).
- Quotes must be verbatim spans copied from the text (no paraphrase).
- If no evidence for a key, return [] for that key.
- Do NOT include anything outside 3-1 and 4-1.
""".strip()

_SYS_SUMMARY = """
Write detailed summaries for:
- pretrain_method  (3-1)
- pretrain_data    (4-1)

Rules:
- Use ONLY the provided quotes.
- Long, specific, faithful; no speculation.
- Output JSON with exactly:
  { "pretrain_method": "...", "pretrain_data": "..." }
""".strip()

def _chat_json(sys: str, user: str) -> Dict[str, Any]:
    r = _client.chat.completions.create(
        model=MODEL_NAME,
        reasoning_effort="medium",
        response_format={"type": "json_object"},
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
    )
    try:
        return json.loads(r.choices[0].message.content.strip())
    except Exception:
        return {}

# ---------- public ----------
def filter_pretrain_reports(base_hf_id: str, gh_repo: Optional[str] = None, output_dir: str | Path = ".") -> Dict[str, Any]:
    """
    Make one file: pretrain_reports_{base}.json with:
      - pretrain_method
      - pretrain_data
      - __evidence: {"3-1 (Pre-training)": [...], "4-1 (Pre-training Data)": [...]}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = base_hf_id.replace("/", "_").lower()
    out_path = output_dir / f"pretrain_reports_{base}.json"

    # 1) harvest articles
    articles = _collect_reports_text(base_hf_id, gh_repo)
    # compact payload for the model
    payload = {"articles": [{"id": f"art{i+1}", "url": a["url"], "source": a["source"], "text": a["text"][:400_000]} for i, a in enumerate(articles)]}
    user_txt = json.dumps(payload, ensure_ascii=False)

    # 2) evidence (quotes only)
    ev = _chat_json(_SYS_RECALL, user_txt)
    if "3-1 (Pre-training)" not in ev: ev["3-1 (Pre-training)"] = []
    if "4-1 (Pre-training Data)" not in ev: ev["4-1 (Pre-training Data)"] = []

    # 3) summary (quotes → method/data)
    quotes_only = {"3-1 (Pre-training)": [q.get("quote","") for q in ev.get("3-1 (Pre-training)", [])],
                   "4-1 (Pre-training Data)": [q.get("quote","") for q in ev.get("4-1 (Pre-training Data)", [])]}
    summ = _chat_json(_SYS_SUMMARY, json.dumps(quotes_only, ensure_ascii=False))

    out = {
        "model_id": base_hf_id,
        "pretrain_method": summ.get("pretrain_method",""),
        "pretrain_data":   summ.get("pretrain_data",""),
        "__evidence": {
            "3-1 (Pre-training)": ev.get("3-1 (Pre-training)", []),
            "4-1 (Pre-training Data)": ev.get("4-1 (Pre-training Data)", []),
        }
    }
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("✅ Saved pretrain reports:", out_path)
    return out

if __name__ == "__main__":
    # quick test
    # filter_pretrain_reports("bigscience/bloom", "bigscience/bloom")
    pass
