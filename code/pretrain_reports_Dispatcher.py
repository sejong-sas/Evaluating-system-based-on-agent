# pretrain_reports_Dispatcher.py
# Collect only pre-training related reports/blog links from base HF/GH README,
# summarize 3-1 (Pre-training) & 4-1 (Pre-training Data) with quotes-only,
# and output a single json: pretrain_reports_{base}.json
#
# Target-model guard (RELAXED):
#   1) 우선순위 A — 문장 수준: 문장 안에 타깃 토큰(예: deepseek, deepseek-v3, llama3.1 등)이 직접 언급된 인용만 채택
#   2) 우선순위 B — 섹션/문서(on-topic) 수준: 만약 해당 기사(문서)가 타깃 모델에 관한 글로 판정되면,
#      같은 문장에 토큰이 없어도 인용을 허용 (단, 가급적 프리트레이닝 관련 내용이어야 함)
#
# NOTE:
#   - o3/o1 reasoning 계열은 temperature/top_p 같은 샘플링 파라미터를 지원하지 않음.
#   - 아래 _chat_json_smart() 는 샘플링 파라미터를 절대 넣지 않으며,
#     reasoning 모델(o1/o3*)에만 reasoning_effort를 추가해 호출함.

import os, re, json, hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import requests
from dotenv import load_dotenv
from openai import OpenAI

# ───────────────── Env ─────────────────
load_dotenv()
_API = os.getenv("OPENAI_API_KEY")
if not _API:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
_client = OpenAI(api_key=_API)

MODEL_NAME = os.getenv("OPENAI_MODEL_PRETRAIN_REPORTS", "o3-mini")
USER_AGENT = {"User-Agent": "Mozilla/5.0 (pretrain-reports-dispatcher)"}

# Payload & filter knobs
PER_ARTICLE_CHAR_CAP   = int(os.getenv("PR_PER_ARTICLE_CHAR_CAP", "200000"))    # per-article text cap
TOTAL_PAYLOAD_CHAR_CAP = int(os.getenv("PR_TOTAL_PAYLOAD_CHAR_CAP", "900000"))  # per-chunk total cap
MIN_TOKEN_HITS_IN_BODY = int(os.getenv("PR_MIN_TOKEN_HITS_IN_BODY", "2"))       # body ≥ N token hits to keep article
MAX_ARTICLES_PER_CHUNK = int(os.getenv("PR_MAX_ARTICLES_PER_CHUNK", "8"))       # soft limiter

# Require article body to look pretraining-related (helps precision)
PR_REQUIRE_PRETRAIN_HINT = os.getenv("PR_REQUIRE_PRETRAIN_HINT", "1") == "1"

# Allow section/doc on-topic fallback for quotes that don't mention target tokens
PR_ALLOW_ON_TOPIC_FALLBACK = os.getenv("PR_ALLOW_ON_TOPIC_FALLBACK", "1") == "1"

# ───────────────── Helpers ─────────────────
def _js(o) -> str:
    return json.dumps(o, ensure_ascii=False, indent=2)

def _tok(s: str) -> List[str]:
    s = re.sub(r"[^a-z0-9.]+", " ", (s or "").lower())
    return [t for t in s.split() if t]

def _canonical_model_tokens(model_id: str) -> List[str]:
    """
    Stable tokens derived from model id:
      - split on non-alnum, drop short/generic
      - add collapsed (remove non-alnum) and no-digit variants
      - add (name, name+digits) pattern like llama3 / llama3.1 → llama, llama3
    """
    name = (model_id or "").split("/", 1)[-1].lower()
    raw = re.split(r"[^a-z0-9.]+", name)
    alts = set()
    stop = {"base","it","instruct","chat","model"}
    for t in raw:
        t = t.strip()
        if len(t) >= 3 and t not in stop:
            alts.add(t)
    collapsed = re.sub(r"[^a-z0-9]", "", name)
    nodigit   = re.sub(r"\d+", "", collapsed)
    if len(collapsed) >= 3: alts.add(collapsed)
    if len(nodigit)   >= 3: alts.add(nodigit)
    # llama3 / llama3.1
    for t in list(alts):
        m = re.match(r"([a-z]+)(\d+(?:\.\d+)*)$", t)
        if m:
            alts.add(m.group(1))
            alts.add(m.group(1)+m.group(2).replace(".",""))
    return sorted(alts)

def _contains_any_token(text: str, toks: List[str]) -> bool:
    tl = (text or "").lower().replace("–","-").replace("—","-")
    return any((t and (t in tl)) for t in toks)

def _token_hits(text: str, toks: List[str]) -> int:
    tl = (text or "").lower()
    hits = 0
    for t in toks:
        if not t: continue
        hits += tl.count(t)
        if re.search(rf"\b{re.escape(t)}\b", tl):
            hits += 1
    return hits

def _dedup_evidence(evs: List[Dict[str,str]], limit:int=300) -> List[Dict[str,str]]:
    seen, out = set(), []
    for ev in evs:
        if not (isinstance(ev, dict) and isinstance(ev.get("source"), str) and isinstance(ev.get("quote"), str)):
            continue
        src = ev["source"].strip(); qt = ev["quote"].strip()
        if not src or not qt: continue
        key = (src, qt)
        if key in seen: continue
        seen.add(key); out.append({"source": src, "quote": qt})
        if len(out) >= limit: break
    return out

def _hash_norm(s: str) -> str:
    return hashlib.sha1(re.sub(r"\s+"," ", (s or "")).strip().lower().encode("utf-8")).hexdigest()

def _dedup_articles_by_url(arts: List[Dict[str,str]]) -> List[Dict[str,str]]:
    seen, out = set(), []
    for a in arts:
        u = (a.get("url","") or "").strip().lower()
        k = re.sub(r"^https?://", "", u)
        k = re.sub(r"[?#].*$", "", k).rstrip("/")
        if k in seen: continue
        seen.add(k); out.append(a)
    return out

# ───────────────── Fetchers ─────────────────
def _hf_card_and_readme(hf_id: str, max_len: int = 120_000) -> str:
    txt = ""
    # HF model card API
    try:
        r = requests.get(f"https://huggingface.co/api/models/{hf_id}?full=true", timeout=15, headers=USER_AGENT)
        if r.ok:
            card = (r.json() or {}).get("cardData", {}) or {}
            txt += (card.get("content") or "")[:max_len]
    except Exception:
        pass
    # README raw
    for br in ("main", "master"):
        try:
            rr = requests.get(f"https://huggingface.co/{hf_id}/raw/{br}/README.md", timeout=12, headers=USER_AGENT)
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

def _fetch_pdf(url: str, cap: int) -> str:
    # Try PyMuPDF (fitz). If not available or fails → return "".
    try:
        import fitz  # PyMuPDF
    except Exception:
        return ""
    try:
        r = requests.get(url, timeout=25, headers=USER_AGENT)
        r.raise_for_status()
        with fitz.open(stream=r.content, filetype="pdf") as doc:
            t = "\n".join(p.get_text() for p in doc)
        t = re.sub(r"\s+", " ", t)
        return t[:cap]
    except Exception:
        return ""

def _fetch_html(url: str, cap: int) -> str:
    try:
        r = requests.get(url, timeout=20, headers=USER_AGENT)
        r.raise_for_status()
        h = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", r.text)
        t = re.sub(r"(?is)<[^>]+>", " ", h)
        t = re.sub(r"\s+", " ", t)
        return t[:cap]
    except Exception:
        return ""

def _collect_reports_text(hf_id: str, gh_repo: Optional[str]) -> List[Dict[str, str]]:
    """Return list of {'id','url','text','source'} harvested from HF/GH READMEs (report-ish only)."""
    corpus: List[Dict[str,str]] = []

    # HF README links
    hf_md = _hf_card_and_readme(hf_id)
    if hf_md:
        urls = list(dict.fromkeys(_LINK_PAT.findall(hf_md)))
        for u in urls:
            if not _looks_report(u): continue
            text = _fetch_pdf(u, PER_ARTICLE_CHAR_CAP) if u.lower().endswith(".pdf") else _fetch_html(u, PER_ARTICLE_CHAR_CAP)
            if text.strip():
                corpus.append({"url": u, "text": text, "source": "hf_readme_link"})

    # GH README links
    if gh_repo:
        gh_md = _gh_readme(gh_repo)
        if gh_md:
            urls = list(dict.fromkeys(_LINK_PAT.findall(gh_md)))
            for u in urls:
                if not _looks_report(u): continue
                text = _fetch_pdf(u, PER_ARTICLE_CHAR_CAP) if u.lower().endswith(".pdf") else _fetch_html(u, PER_ARTICLE_CHAR_CAP)
                if text.strip():
                    corpus.append({"url": u, "text": text, "source": "gh_readme_link"})

    # de-dup by URL
    corpus = _dedup_articles_by_url(corpus)

    # assign ids
    for i, a in enumerate(corpus):
        a["id"] = f"art{i+1}"
    return corpus

# ───────────────── Model relevance filter ─────────────────
_PRETRAIN_HINTS = (
    "pretrain", "pre-training", "pretraining",
    "training corpus", "training data", "dataset", "datasets", "corpus",
    "tokens", "billion tokens", "trillion tokens",
    "compute", "flops", "gpu hours", "h100", "tpu", "v100",
    "mixture", "data mixture", "crawl", "common crawl", "c4", "pile", "roots"
)

def _article_related_to_model(art: Dict[str,str], model_tokens: List[str]) -> bool:
    """
    Keep article if:
      • URL contains any model token, OR
      • Body contains ≥ MIN_TOKEN_HITS_IN_BODY of those tokens.
      • AND (optionally) body contains pretraining hints to reduce false positives.
    """
    url = (art.get("url") or "").lower()
    body = (art.get("text") or "").lower()
    if any(t in url for t in model_tokens):
        model_ok = True
    else:
        model_ok = _token_hits(body, model_tokens) >= MIN_TOKEN_HITS_IN_BODY

    if not model_ok:
        return False

    if not PR_REQUIRE_PRETRAIN_HINT:
        return True
    return any(h in body for h in _PRETRAIN_HINTS)

def _filter_articles_for_target(arts: List[Dict[str,str]], model_id: str) -> List[Dict[str,str]]:
    toks = _canonical_model_tokens(model_id)
    if not toks:
        return []
    out = [a for a in arts if _article_related_to_model(a, toks)]
    return out  # allow empty → later "No information"

# ───────────────── Chunking articles ─────────────────
def _chunk_articles(arts: List[Dict[str,str]]) -> List[List[Dict[str,str]]]:
    if not arts:
        return []
    chunks: List[List[Dict[str,str]]] = []
    cur: List[Dict[str,str]] = []
    total = 0
    for a in arts:
        txt_len = len(a.get("text",""))
        if (cur and (total + txt_len > TOTAL_PAYLOAD_CHAR_CAP)) or (len(cur) >= MAX_ARTICLES_PER_CHUNK):
            chunks.append(cur); cur=[]; total=0
        cur.append(a); total += txt_len
    if cur:
        chunks.append(cur)
    return chunks

# ───────────────── Prompts ─────────────────
# NEW: on-topic fallback 규칙을 시스템 프롬프트에 명시
_SYS_RECALL = """
You extract **EVIDENCE about pre-training only** (methodology and data) for the **TARGET model**.

MODEL FILTER (two-tier):
1) Prefer quotes where the sentence itself explicitly mentions one of the TARGET tokens.
2) If an article/section is marked as `on_topic = true` for the TARGET model,
   you may also select quotes that do not contain the tokens, as long as they still
   describe the TARGET model's pretraining method or pretraining data.

Return a JSON object with two arrays of evidence objects:
{
  "3-1 (Pre-training)": [{"source":"<short id>","quote":"<verbatim sentence>"}],
  "4-1 (Pre-training Data)": [{"source":"<short id>","quote":"<verbatim sentence>"}]
}

Rules:
- Use ONLY the provided payload articles (each item has: id, url, source, on_topic, text).
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

def _recall_user_payload(model_id: str, model_tokens: List[str], arts: List[Dict[str,str]]) -> str:
    # compact payload for the model, include on_topic flag per article
    payload = {
        "target_model": model_id,
        "target_tokens": model_tokens,
        "articles": [
            {
                "id": a.get("id",""),
                "url": a.get("url",""),
                "source": a.get("source",""),
                "on_topic": bool(a.get("_on_topic", False)),
                "text": (a.get("text","")[:PER_ARTICLE_CHAR_CAP])
            }
            for a in arts
        ]
    }
    return json.dumps(payload, ensure_ascii=False)

# ───────────────── Chat helper (o3 규칙 반영) ─────────────────
def _is_reasoning_model(name: str) -> bool:
    n = (name or "").lower()
    return n.startswith(("o1", "o3"))

def _chat_json_smart(model_name: str, system: str, user: str, json_only: bool = True) -> Dict[str, Any]:
    """
    • 샘플링 파라미터(temperature, top_p 등)는 절대 추가하지 않음.
    • o1/o3 계열에만 reasoning_effort 추가.
    """
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if json_only:
        payload["response_format"] = {"type": "json_object"}
    if _is_reasoning_model(model_name):
        payload["reasoning_effort"] = "medium"

    r = _client.chat.completions.create(**payload)
    try:
        return json.loads(r.choices[0].message.content.strip())
    except Exception:
        return {}

# ────────────────── Local GH repo inference ──────────────────
def _infer_gh_repo_from_outdir(base_hf_id: str, output_dir: Path) -> Optional[str]:
    """
    If gh_repo is not provided, try to infer from output_dir/github_{base}.json
    (expects {"repo": "owner/name"} or {"full_name": "owner/name"}).
    """
    base = base_hf_id.replace("/", "_").lower()
    p = output_dir / f"github_{base}.json"
    if not p.exists():
        # fallback to project root
        p = Path(f"github_{base}.json")
        if not p.exists():
            return None
    try:
        j = json.load(open(p, encoding="utf-8"))
        rep = (j.get("repo") or j.get("full_name") or "").strip()
        return rep or None
    except Exception:
        return None

# ───────────────── Public ─────────────────
def filter_pretrain_reports(base_hf_id: str, gh_repo: Optional[str] = None, output_dir: str | Path = ".") -> Dict[str, Any]:
    """
    Make one file: pretrain_reports_{base}.json with:
      - pretrain_method
      - pretrain_data
      - __evidence: {"3-1 (Pre-training)": [...], "4-1 (Pre-training Data)": [...]}
    Evidence selection:
      • sentence-level token match (preferred)
      • OR (if enabled) article on-topic fallback
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = base_hf_id.replace("/", "_").lower()
    out_path = output_dir / f"pretrain_reports_{base}.json"

    # 0) infer GH repo if not provided
    if not gh_repo:
        gh_repo = _infer_gh_repo_from_outdir(base_hf_id, output_dir)

    # 1) harvest articles → filter by target-model relevance (+ pretrain hints if enabled)
    all_articles = _collect_reports_text(base_hf_id, gh_repo)
    articles = _filter_articles_for_target(all_articles, base_hf_id)

    # mark on_topic for accepted articles
    for a in articles:
        a["_on_topic"] = True

    toks = _canonical_model_tokens(base_hf_id)
    ev_all_31: List[Dict[str,str]] = []
    ev_all_41: List[Dict[str,str]] = []

    # Build id → on_topic map for post-filter
    id_on_topic = {a.get("id",""): bool(a.get("_on_topic", False)) for a in articles}

    # 2) chunked evidence recall
    for chunk in _chunk_articles(articles):
        if not chunk: continue
        user_txt = _recall_user_payload(base_hf_id, toks, chunk)
        ev = _chat_json_smart(MODEL_NAME, _SYS_RECALL, user_txt, json_only=True)
        arr31 = ev.get("3-1 (Pre-training)", []) or []
        arr41 = ev.get("4-1 (Pre-training Data)", []) or []

        # keep quotes if:
        #   (A) quote mentions target tokens, OR
        #   (B) PR_ALLOW_ON_TOPIC_FALLBACK = True AND source id is on_topic
        def _keep(e):
            if not isinstance(e, dict): return False
            q = e.get("quote","") or ""
            sid = (e.get("source","") or "").strip()
            if _contains_any_token(q, toks):
                return True
            return PR_ALLOW_ON_TOPIC_FALLBACK and id_on_topic.get(sid, False)

        arr31 = [e for e in arr31 if _keep(e)]
        arr41 = [e for e in arr41 if _keep(e)]
        ev_all_31.extend(arr31); ev_all_41.extend(arr41)

    # 3) dedup & summarize
    ev_all_31 = _dedup_evidence(ev_all_31, 300)
    ev_all_41 = _dedup_evidence(ev_all_41, 300)

    quotes_only = {
        "3-1 (Pre-training)": [q.get("quote","") for q in ev_all_31],
        "4-1 (Pre-training Data)": [q.get("quote","") for q in ev_all_41],
    }
    summ = _chat_json_smart(MODEL_NAME, _SYS_SUMMARY, json.dumps(quotes_only, ensure_ascii=False), json_only=True)

    pretrain_method = (summ.get("pretrain_method") or "").strip()
    pretrain_data   = (summ.get("pretrain_data") or "").strip()

    # If no evidence remains, downgrade to "No information"
    if not (ev_all_31 or ev_all_41):
        pretrain_method = "No information"
        pretrain_data   = "No information"

    out = {
        "model_id": base_hf_id,
        "pretrain_method": pretrain_method,
        "pretrain_data":   pretrain_data,
        "__evidence": {
            "3-1 (Pre-training)": ev_all_31,
            "4-1 (Pre-training Data)": ev_all_41,
        }
    }
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("✅ Saved pretrain reports:", out_path)
    return out

# ───────────────────────── CLI ─────────────────────────
if __name__ == "__main__":
    # Example:
    #   filter_pretrain_reports("bigscience/bloom", "bigscience/bloom")
    import sys
    if len(sys.argv) >= 2:
        hf = sys.argv[1]
        gh = sys.argv[2] if len(sys.argv) >= 3 else None
        filter_pretrain_reports(hf, gh)
