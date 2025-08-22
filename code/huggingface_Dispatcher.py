# pip install --upgrade pip
# pip install requests python-dotenv openai huggingface_hub PyMuPDF
# pip install -U accelerate
import json
import re
import requests
import os
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
from openai import OpenAI

from huggingface_Fetcher import huggingface_fetcher
from github_Fetcher import github_fetcher
from arxiv_Fetcher import arxiv_fetcher_from_model
from github_Dispatcher import filter_github_features
from arxiv_Dispatcher import filter_arxiv_features
from reports_Dispatcher import filter_reports_features
from huggingface_Dispatcher import filter_hf_features
from openness_Evaluator import evaluate_openness_from_files
from inference import run_inference
from pretrain_reports_Dispatcher import filter_pretrain_reports
from difflib import SequenceMatcher

import html
import difflib

# Load environment variables and initialize OpenAI client
dotenv_path = os.path.join(os.getcwd(), '.env')
load_dotenv(dotenv_path)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Warn/keys helpers (NEW) ----------
def _warn(msg: str):
    print(msg, flush=True)

def _has_openai_key() -> bool:
    if not os.getenv("OPENAI_API_KEY"):
        _warn("🔑 OPENAI_API_KEY is not set — GPT steps will be skipped.")
        return False
    return True

# ---------- Heuristics & utils ----------
MONOREPO_DENYLIST = {
    "google-research/google-research",
    "google-research/google-research-private",
}
BAD_REPO_KEYWORDS = {
    "api","client","sdk","demo","website","docs","doc","notebook","colab",
    "examples","sample","bench","leaderboard","eval","evaluation","convert",
    "export","deploy","inference","space","slim","angelslim","awesome","papers"
}

def _tokens(s: str) -> set[str]:
    """
    Tokenize for loose overlap. We allow >=2 length to keep short names like 'ax'.
    """
    import re as _re
    return set(t for t in _re.sub(r"[^a-z0-9]+"," ", (s or "").lower()).split() if len(t) >= 2)

def _norm_name(s: str) -> str:
    """Remove non-alnum to compare names like 'A.X-3.1' vs 'AX-3' robustly."""
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def _name_similarity_ratio(hf_id: str, repo: str) -> float:
    """Fuzzy ratio between HF model name part and repo name part."""
    hf_name = (hf_id.split("/",1)[1] if "/" in hf_id else hf_id)
    repo_name = (repo.split("/",1)[1] if "/" in repo else repo)
    a = _norm_name(hf_name)
    b = _norm_name(repo_name)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()

def _org_match_or_similar_name(a: str, b: str, thresh: float = 0.75) -> bool:
    """True if org is exactly same or very similar (ratio >= thresh)."""
    a = (a or "").lower(); b = (b or "").lower()
    if not a or not b:
        return False
    if a == b:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= thresh

def _looks_relevant_repo(repo: str, hf_id: str) -> bool:
    """
    Family/series repos OK; block obvious monorepos & tool/demo/docs repos.
    Uses BOTH token overlap and fuzzy name/org similarity.
    """
    if not repo or not hf_id:
        return False
    rl = repo.lower()
    if rl in MONOREPO_DENYLIST:
        return False

    org = rl.split("/",1)[0]
    name = rl.split("/",1)[1]

    hf_org = hf_id.split("/",1)[0].lower() if "/" in hf_id else ""
    org_ok = _org_match_or_similar_name(org, hf_org)

    toks_hf = _tokens(hf_id)
    toks_repo = _tokens(name)
    overlap_ok = bool(toks_hf & toks_repo)

    sim = _name_similarity_ratio(hf_id, repo)
    sim_ok = sim >= 0.62  # tolerate short model names like A.X-3.1

    if not (overlap_ok or sim_ok):
        return False

    # Block noisy repos unless strong family match
    if any(k in name for k in BAD_REPO_KEYWORDS):
        fam_hit = overlap_ok or (sim >= 0.7)
        if not fam_hit:
            return False

    # If org is totally different and name looks like demo/instruct, be strict
    if not org_ok and ("instruct" in name or "demo" in name or "sagemaker" in name):
        return False

    return True

def _score_repo(repo: str, hf_id: str) -> int:
    """
    Score: org match/similarity + token overlap + name similarity - noise keywords.
    """
    rl = repo.lower()
    org, name = rl.split("/",1)
    org_hf = hf_id.split("/",1)[0].lower() if "/" in hf_id else ""
    score = 0

    if org == org_hf or _org_match_or_similar_name(org, org_hf):
        score += 8
    else:
        score -= 6

    toks_hf   = _tokens(hf_id)
    toks_repo = _tokens(name)
    score += 2 * len(toks_hf & toks_repo)

    # name similarity
    sim = _name_similarity_ratio(hf_id, repo)
    score += int(round(10 * sim))  # up to +10

    if any(k in name for k in ("model","models","llm")): score += 2
    if any(k in name for k in BAD_REPO_KEYWORDS): score -= 2

    # org 불일치 + instruct 네이밍은 추가 패널티
    if "instruct" in name and org != org_hf:
        score -= 4

    return score

# ──────────────────────────────────────────────────────────────
# GPT-based rescue: does this repo really contain THIS model?
def _fetch_repo_snapshot(repo: str) -> dict:
    """
    Grab short README (main/master) and a sample of tree paths.
    """
    headers = {"Accept": "application/vnd.github+json"}
    tok = os.getenv("GITHUB_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"

    # README
    readme = ""
    for br in ["main", "master"]:
        try:
            r = requests.get(
                f"https://raw.githubusercontent.com/{repo}/{br}/README.md",
                headers=headers, timeout=15
            )
            if r.status_code == 200:
                readme = r.text
                break
        except Exception:
            pass

    # TREE
    paths = []
    for br in ["main", "master"]:
        try:
            tr = requests.get(
                f"https://api.github.com/repos/{repo}/git/trees/{br}?recursive=1",
                headers=headers, timeout=15
            )
            if tr.status_code == 200:
                j = tr.json()
                for it in j.get("tree", []):
                    p = str(it.get("path", ""))
                    if p:
                        paths.append(p)
                break
        except Exception:
            pass

    # truncate
    readme = (readme or "")[:8000]
    paths = paths[:400]
    return {"readme": readme, "paths": paths}

def _gpt_repo_model_signal(repo: str, hf_id: str) -> tuple[bool, float, str]:
    """
    Ask GPT to judge if the repo is substantially about the target HF model.
    Returns (ok, confidence, reason)
    """
    if not _has_openai_key():
        return (False, 0.0, "No OPENAI_API_KEY")

    snap = _fetch_repo_snapshot(repo)
    if not snap["readme"] and not snap["paths"]:
        return (False, 0.0, "Empty snapshot")

    model_name = hf_id.split("/",1)[1]
    focus_tokens = list(dict.fromkeys(re.split(r"[-_. ]+", model_name.lower())))

    sys = "You decide if this GitHub repo is substantially about the target model."
    prompt = f"""
Target model HF id: {hf_id}
Focus tokens (any is fine): {focus_tokens}

Return strict JSON:
{{
  "ok": true|false,
  "confidence": 0.0-1.0,
  "reason": "short",
  "signals": ["e.g., exact model name in README", "files mentioning it", "HF link match"]
}}

Consider the following snapshot (truncated):
=== README ===
{snap["readme"]}

=== PATHS (sample) ===
{json.dumps(snap["paths"], ensure_ascii=False)}
"""
    try:
        rsp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL_SIGNAL", "gpt-4o-mini"),
            response_format={"type": "json_object"},
            temperature=0,
            messages=[
                {"role":"system", "content": sys},
                {"role":"user", "content": prompt}
            ]
        )
        obj = json.loads(rsp.choices[0].message.content)
        ok = bool(obj.get("ok"))
        conf = float(obj.get("confidence", 0))
        reason = obj.get("reason","")
        return (ok and conf >= 0.6, conf, reason)
    except Exception as e:
        _warn(f"🔑 OpenAI error during repo-signal check: {e}")
        return (False, 0.0, "OpenAI error")

def _rescue_accept_with_gpt(repo: str, hf_id: str) -> bool:
    ok, conf, reason = _gpt_repo_model_signal(repo, hf_id)
    if ok:
        print(f"🤖 accept via GPT-signal ({conf:.2f}): {repo} — {reason}")
        return True
    print(f"🤖 reject via GPT-signal ({conf:.2f}): {repo}")
    return False

# ──────────────────────────────────────────────────────────────
# Estimate the pretrained (base) model with GPT-4o
def gpt_detect_base_model(hf_id: str) -> str | None:
    """
    • If the input model is a finetuned model → return the pretrained model ID
    • If the input is already a pretrained (original) model → return None
    • If unsure → return None
    """
    import textwrap, re as _re, requests as _req, json as _json

    def _hf_card_readme(mid: str, max_len: int = 12000) -> str:
        try:
            r = _req.get(f"https://huggingface.co/api/models/{mid}?full=true", timeout=15)
            if r.status_code in (401,403):
                _warn("🔑 HF card 401/403 — set HF_TOKEN if you have access.")
            elif r.status_code == 429:
                _warn("⏳ HF card rate-limited (429) — try later or set HF_TOKEN.")
            card = (r.json().get("cardData") or {})
            txt = (card.get("content") or "")[:max_len]
            for br in ["main","master"]:
                rr = _req.get(f"https://huggingface.co/{mid}/raw/{br}/README.md", timeout=10)
                if rr.status_code == 200:
                    txt += "\n\n" + rr.text[:max_len]
                    break
            return txt
        except Exception:
            return ""

    # 모델명만 추출 (org 무시)
    def _name_only(mid: str) -> str:
        nm = (mid.split("/",1)[1] if "/" in mid else mid).lower()
        return _re.sub(r"[^a-z0-9]+","", nm)

    prompt_sys = textwrap.dedent(f"""
        You are an expert at analyzing AI model information to identify the 'pretrained (base) model'.

        • The input model **{hf_id}** might be a finetuned model.
        • Read the Hugging Face card / README below and
          ➡️ infer the most likely pretrained model ID this model derives from.
        • If the input is already a pretrained model, return null.

        ➤ Output format → a single line of JSON only! Examples:
            {{ "pretrain_model": "bigscience/bloom-560m" }}
          or
            {{ "pretrain_model": null }}
    """).strip()

    ctx = _hf_card_readme(hf_id)
    if not ctx or not _has_openai_key():
        return None

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role":"system","content":prompt_sys},
                      {"role":"user","content":ctx}],
            temperature=0
        )
        pred = _json.loads(resp.choices[0].message.content)
        pre_id = (pred.get("pretrain_model") or "").strip()
        if not pre_id:
            return None

        # ✅ org 무시하고 모델명이 같으면 자기 자신 → None
        if _name_only(pre_id) == _name_only(hf_id):
            return None

        # 존재 확인
        if test_hf_model_exists(pre_id):
            return pre_id
    except Exception as e:
        _warn(f"🔑 OpenAI error during base-model detection — {e}")
    return None

# ──────────────────────────────────────────────────────────────


# 1. Input parsing: URL or org/model
def extract_model_info(input_str: str) -> dict:
    platform = None
    organization = model = None
    if input_str.startswith("http"):
        parsed = urlparse(input_str)
        domain = parsed.netloc.lower()
        segments = parsed.path.strip("/").split("/")
        if len(segments) >= 2:
            organization = segments[0]
            model = segments[1].split("?")[0].split("#")[0].replace(".git", "")
            if "huggingface" in domain:
                platform = "huggingface"
            elif "github" in domain:
                platform = "github"
    else:
        parts = input_str.strip().split("/")
        if len(parts) == 2:
            organization, model = parts
            platform = "unknown"
    if not organization or not model:
        raise ValueError("Invalid input format. Enter 'org/model' or a URL.")
    full_id = f"{organization}/{model}"
    hf_id = full_id.lower()
    return {"platform": platform, "organization": organization,
            "model": model, "full_id": full_id, "hf_id": hf_id}

# 2. Existence tests (with warnings)
def test_hf_model_exists(model_id: str) -> bool:
    resp = requests.get(f"https://huggingface.co/api/models/{model_id}")
    if resp.status_code in (401, 403):
        _warn("🔑 Hugging Face API says 401/403 — model may be private. "
              "Set HF_TOKEN in .env if you have access.")
    elif resp.status_code == 429:
        _warn("⏳ Hugging Face API rate limited (429) — try later or set HF_TOKEN.")
    return resp.status_code == 200

def test_github_repo_exists(repo: str) -> bool:
    headers = {"Accept": "application/vnd.github+json"}
    tok = os.getenv("GITHUB_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    resp = requests.get(f"https://api.github.com/repos/{repo}", headers=headers)
    if resp.status_code == 403 and not tok:
        _warn("🔑 GitHub API rate limited (403) — set GITHUB_TOKEN in .env "
              "(Personal Access Token) to raise limits.")
    return resp.status_code == 200

# 3. Link parsing helper (★ preserve original case)
def extract_repo_from_url(url: str) -> str:
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) >= 2:
        repo = parts[1].split("?")[0].split("#")[0].replace(".git", "")
        return f"{parts[0]}/{repo}"  # preserve original case
    return ""

# 4. HF page → GitHub (collector; used also for pretrain base GH lookup)
def find_github_in_huggingface(model_id: str) -> str | None:
    """
    Collect GitHub link candidates from the HF model card/README and return the most plausible repository.
    """
    def _extract_repo_from_url_preserve(url: str) -> str | None:
        try:
            p = urlparse(url)
            if "github.com" not in p.netloc.lower():
                return None
            seg = p.path.strip("/").split("/")
            if len(seg) >= 2:
                repo = seg[1].split("?")[0].split("#")[0].replace(".git", "")
                return f"{seg[0]}/{repo}"
        except Exception:
            pass
        return None

    def _tokenize(s: str) -> list[str]:
        s = s.lower().replace("/", " ")
        s = re.sub(r"[^a-z0-9]+", " ", s)
        return [t for t in s.split() if t]

    def _score(repo_lower: str, hf_org: str, toks: list[str]) -> int:
        score = 0
        org, name = repo_lower.split("/", 1)
        if hf_org and org == hf_org.lower():
            score += 5
        for t in toks:
            if t and t in name:
                score += 2
        if any(k in name for k in ["model", "models", "llm"]):
            score += 2
        for k in ["api","client","sdk","demo","website","docs","doc","notebook","colab",
                  "examples","sample","bench","leaderboard","eval","evaluation","convert",
                  "export","deploy","inference","space","slim","angelslim"]:
            if k in name:
                score -= 2
        return score

    try:
        card_resp = requests.get(
            f"https://huggingface.co/api/models/{model_id}?full=true"
        )
        if card_resp.status_code in (401,403):
            _warn("🔑 HF card 401/403 — set HF_TOKEN if you have access.")
        elif card_resp.status_code == 429:
            _warn("⏳ HF card rate-limited (429) — try later or set HF_TOKEN.")
        card = card_resp.json().get("cardData", {}) or {}

        hf_org = model_id.split("/")[0] if "/" in model_id else ""
        toks = _tokenize(model_id)

        cand_map: dict[str, str] = {}

        def _add_candidate(rep: str | None):
            if not rep:
                return
            cand_map.setdefault(rep.lower(), rep)

        for field in ["repository", "homepage"]:
            links = (card.get("links", {}) or {}).get(field)
            if isinstance(links, str):
                _add_candidate(_extract_repo_from_url_preserve(links))
            elif isinstance(links, list):
                for u in links:
                    _add_candidate(_extract_repo_from_url_preserve(str(u)))

        content = card.get("content", "") or ""
        for url in re.findall(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", content):
            _add_candidate(_extract_repo_from_url_preserve(url))

        for br in ["main", "master"]:
            try:
                r = requests.get(f"https://huggingface.co/{model_id}/raw/{br}/README.md")
                if r.status_code == 200:
                    for url in re.findall(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", r.text):
                        _add_candidate(_extract_repo_from_url_preserve(url))
                    break
            except Exception:
                pass

        if not cand_map:
            return None

        best_lower, best_score = None, -10**9
        for rep_lower, rep_orig in cand_map.items():
            if not test_github_repo_exists(rep_orig):
                continue
            s = _score(rep_lower, hf_org, toks)
            if s > best_score:
                best_lower, best_score = rep_lower, s

        return cand_map[best_lower] if best_lower else None
    except Exception:
        return None

# 5. GitHub page → HF (kept for optional flows)
def find_huggingface_in_github(repo: str) -> str:
    for fname in ["README.md"]:
        for branch in ["main", "master"]:
            raw_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{fname}"
            try:
                r = requests.get(raw_url)
                if r.status_code == 200:
                    m = re.search(r"https?://huggingface\.co/([\w\-]+/[\w\-\.]+)", r.text, re.IGNORECASE)
                    if m:
                        candidate = m.group(1).lower()
                        if not candidate.startswith('collections/'):
                            return candidate
                    m2 = re.search(r"huggingface\.co/([\w\-]+/[\w\-\.]+)", r.text, re.IGNORECASE)
                    if m2:
                        candidate = m2.group(1).lower()
                        if not candidate.startswith('collections/'):
                            return candidate
                    m_md = re.search(r"\[.*?\]\((https?://huggingface\.co/[\w\-/\.]+)\)", r.text, re.IGNORECASE)
                    if m_md:
                        candidate = extract_model_info(m_md.group(1))["hf_id"]
                        if not candidate.startswith('collections/'):
                            return candidate
                    m_html = re.search(r'<a\s+href="https?://huggingface\.co/([\w\-/\.]+)"', r.text, re.IGNORECASE)
                    if m_html:
                        candidate = m_html.group(1).lower()
                        if not candidate.startswith('collections/'):
                            return candidate
            except:
                pass
    try:
        html = requests.get(f"https://github.com/{repo}").text
        m3 = re.findall(r"https://huggingface\.co/[\w\-]+/[\w\-\.]+", html, re.IGNORECASE)
        for link in m3:
            if 'href' in html[html.find(link)-20:html.find(link)]:
                candidate = extract_model_info(link)["hf_id"]
                if not candidate.startswith('collections/'):
                    return candidate
    except:
        pass
    return None

def gpt_guess_github_from_huggingface(hf_id: str) -> str:
    if not _has_openai_key():
        return None
    prompt = f"""
For the model '{hf_id}' registered on Hugging Face, infer the GitHub repository that hosts the original source code.

🟢 Rules to follow:
1. Return **only** the exact GitHub path in 'organization/repo' format (no links, no description).
2. Avoid overly broad monorepos like 'google-research/google-research'; prefer a model-specific repository if available.
3. If it's a distilled model, identify the parent model's repository.
4. Refer to the model name, architecture, paper, tokenizer, and libraries used (PyTorch, JAX, T5, etc.) to make an accurate guess.
5. Output must be **exactly one line**, e.g., `facebookresearch/llama`

🔴 Do not include any explanations—only the GitHub repository path.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        guess = response.choices[0].message.content.strip()
        if "/" in guess:
            return guess
    except Exception as e:
        _warn(f"🔑 OpenAI error during GPT HF→GH guess — check OPENAI_API_KEY/quota. Detail: {e}")
    return None

def gpt_guess_huggingface_from_github(gh_id: str) -> str:
    if not _has_openai_key():
        return None
    prompt = f"""
For the model '{gh_id}' on GitHub, infer the corresponding Hugging Face model ID.
- Output only the exact organization/repository path.
- Base your inference on the GitHub repository associated with the model's name or paper.
- Example output: facebookresearch/llama
- Avoid broad monorepos like 'google-research/google-research'; prefer a repository dedicated to the model if available.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        guess = response.choices[0].message.content.strip().lower()
        if "/" in guess:
            return guess
    except Exception as e:
        _warn(f"🔑 OpenAI error during GPT GH→HF guess — check OPENAI_API_KEY/quota. Detail: {e}")
    return None

# --- Priority 2 (HF README context → GPT) ---
def gpt_guess_github_from_hf_with_context(hf_id: str) -> list[str]:
    """
    Ask GPT to extract up to 3 likely GitHub repos from the HF card/README text.
    Returns: ["org/repo", ...] (sorted by confidence/order)
    """
    import textwrap, requests, json as _json

    def _hf_card_readme(mid: str, max_len: int = 20000) -> str:
        try:
            card = requests.get(
                f"https://huggingface.co/api/models/{mid}?full=true", timeout=15
            )
            if card.status_code in (401,403):
                _warn("🔑 HF card 401/403 — set HF_TOKEN if you have access.")
            elif card.status_code == 429:
                _warn("⏳ HF card rate-limited (429) — try later or set HF_TOKEN.")
            card = card.json().get("cardData", {}) or {}
            txt = (card.get("content") or "")[:max_len]
            for br in ["main", "master"]:
                r = requests.get(f"https://huggingface.co/{mid}/raw/{br}/README.md", timeout=10)
                if r.status_code == 200:
                    txt += "\n\n" + r.text[:max_len]
                    break
            return txt
        except Exception:
            return ""

    ctx = _hf_card_readme(hf_id)
    if not ctx.strip():
        return []

    if not _has_openai_key():
        return []

    sys = "You extract GitHub repositories for the given model from the provided text."
    prompt = textwrap.dedent(f"""
    Model: {hf_id}

    From the Hugging Face card/README content below, list up to 3 GitHub repositories
    that most likely host the ORIGINAL source code or the FAMILY repository for this model
    (series repos are allowed, e.g., a family repo containing multiple model variants).

    Return JSON only:
    {{
      "candidates": [
        {{"repo":"org/name","confidence":0.0,"why":"short reason"}},
        ...
      ]
    }}

    Content:
    ---
    {ctx}
    ---
    """).strip()

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type":"json_object"},
            messages=[{"role":"system","content":sys},{"role":"user","content":prompt}],
            temperature=0
        )
        data = _json.loads(resp.choices[0].message.content)
        cands = []
        for c in (data.get("candidates") or []):
            repo = (c.get("repo") or "").strip()
            if "/" in repo:
                cands.append(repo)
        return cands[:3]
    except Exception as e:
        _warn(f"🔑 OpenAI error during HF-context guess — check OPENAI_API_KEY/quota. Detail: {e}")
        return []

# --- Priority 3 (Web search → candidates) ---
def web_search_github_candidates(hf_id: str) -> list[str]:
    """
    Find likely GitHub repos via web search.
    Prefers Tavily; falls back to GitHub Search API if no Tavily key is set.
    """
    import os as _os, requests as _req, re as _re

    name = hf_id.split("/",1)[1]
    queries = [
        f"{name} github repository",
        f"{name} model github",
        f"{name} official github",
    ]

    # 1) Tavily (if available)
    tavily = _os.getenv("TAVILY_API_KEY")
    if tavily:
        urls = []
        for q in queries:
            try:
                r = _req.post("https://api.tavily.com/search",
                              json={"api_key":tavily,"query":q,"max_results":5},
                              timeout=15)
                if r.status_code != 200:
                    _warn(f"🔑 Tavily error (HTTP {r.status_code}) — check key/quota; will fall back to GitHub Search.")
                    continue
                try:
                    j = r.json()
                except Exception:
                    _warn("🔑 Tavily returned non-JSON — possible service issue/quota; using fallback.")
                    continue
                if not j.get("results"):
                    err = j.get("error") or j.get("message") or "no results"
                    _warn(f"🔑 Tavily returned no results ({err}) — check key/quota; using fallback.")
                    continue
                for it in j.get("results", []):
                    u = it.get("url","")
                    if "github.com" in u:
                        m = _re.search(r"github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", u)
                        if m: urls.append(m.group(1))
            except Exception as e:
                _warn(f"🔑 Tavily request failed — {e}. Using GitHub Search fallback.")
        if urls:
            return list(dict.fromkeys(urls))[:5]
    else:
        _warn("ℹ️ TAVILY_API_KEY not set — using GitHub Search API fallback.")

    # 2) GitHub Search API (fallback)
    gh_tok = _os.getenv("GITHUB_TOKEN")
    headers = {"Accept":"application/vnd.github+json"}
    if gh_tok:
        headers["Authorization"] = f"Bearer {gh_tok}"
    else:
        _warn("ℹ️ No GITHUB_TOKEN — unauthenticated GitHub Search has low rate limits.")

    repos = []
    for q in queries:
        try:
            r = _req.get("https://api.github.com/search/repositories",
                         params={"q":q, "sort":"stars", "order":"desc", "per_page":5},
                         headers=headers, timeout=15)
            if r.status_code == 403 and not gh_tok:
                _warn("🔑 GitHub Search rate limited (403) — set GITHUB_TOKEN in .env to raise limits.")
                continue
            j = r.json()
            for item in j.get("items", [])[:5]:
                full = item.get("full_name")
                if full: repos.append(full)
        except Exception as e:
            _warn(f"⚠️ GitHub Search failed: {e}")
    return list(dict.fromkeys(repos))[:5]

def _robust_fetch_report(url: str) -> tuple[str, str, str]:
    """
    주어진 URL에서 텍스트를 최대한 확보해 (text, used_url, method)를 반환.
    method: 'direct-pdf' | 'direct-html' | 'normalized-url' | 'jina-reader' | 'wayback' | 'failed'
    """
    import re, requests
    try:
        import fitz  # PyMuPDF
    except Exception:
        fitz = None

    UA = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/123.0 Safari/537.36",
        "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.8",
        "Referer": "https://huggingface.co/"
    }

    def _norm_candidates(u: str) -> list[str]:
        # 원본 + 교정 후보들(CloudFront→CDN, 오타, OpenAI blog→index)
        out = [u]
        if "d4mucfpksywv.cloudfront.net" in u:
            out.append(u.replace("d4mucfpksywv.cloudfront.net", "cdn.openai.com"))
        if "language_moodels" in u:
            out.append(u.replace("language_moodels", "language_models"))
        if "language-moodels" in u:
            out.append(u.replace("language-moodels", "language-models"))
        if "openai.com/blog/" in u:
            out.append(u.replace("/blog/", "/index/"))
        # 중복 제거
        seen, uniq = set(), []
        for x in out:
            if x not in seen:
                seen.add(x); uniq.append(x)
        return uniq

    def _is_pdf(u: str, resp=None) -> bool:
        if u.lower().endswith(".pdf"):
            return True
        if resp is not None:
            ct = (resp.headers.get("Content-Type") or "").lower()
            if "pdf" in ct:
                return True
        return False

    def _strip_html(html: str) -> str:
        html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html or "")
        html = re.sub(r"(?is)<[^>]+>", " ", html)
        html = re.sub(r"\s+", " ", html)
        return html[:800_000]

    # 1) 직접 요청(+정상화 후보들 순회)
    for cand in _norm_candidates(url):
        try:
            r = requests.get(cand, headers=UA, timeout=25)
            if r.status_code == 200:
                if _is_pdf(cand, r):
                    if not fitz:
                        return "", cand, "pdf-but-pymupdf-missing"
                    with fitz.open(stream=r.content, filetype="pdf") as doc:
                        text = "\n".join(p.get_text() for p in doc)
                    if text.strip():
                        return text, cand, "direct-pdf"
                else:
                    text = _strip_html(r.text)
                    if text.strip():
                        return text, cand, "direct-html"
        except Exception:
            pass  # 다음 후보 시도

    # 2) Jina Reader 폴백 (403/DNS 회피, HTML 텍스트화)
    try:
        # 원본 스킴 유지
        scheme = "http" if url.startswith("http://") else "https"
        jurl = f"https://r.jina.ai/{scheme}://{url.split('://', 1)[1]}"
        r = requests.get(jurl, headers=UA, timeout=25)
        if r.status_code == 200 and r.text.strip():
            return r.text[:800_000], jurl, "jina-reader"
    except Exception:
        pass

    # 3) Wayback 폴백(베스트에포트)
    try:
        wb = f"https://web.archive.org/web/{url}"
        r = requests.get(wb, headers=UA, timeout=25)
        if r.status_code == 200:
            t = _strip_html(r.text)
            if t.strip():
                return t[:800_000], wb, "wayback"
    except Exception:
        pass

    return "", url, "failed"


# --- Utility: harvest technical reports from GitHub JSON into reports_fulltext_{hf}.json
def harvest_reports_from_github_json(gh: dict, hf_id: str, output_dir: str | Path = "."):
    """
    GitHub fetcher JSON(README + file list)에서 논문/블로그/리포트 링크를 추출해
    텍스트를 확보(강화 폴백 사용)하고 reports_fulltext_{hf}.json에 병합 저장.
    """
    import re
    from pathlib import Path

    def _extract_urls(text: str) -> list[str]:
        urls = re.findall(r'https?://[^\s)>"\']+', text or "")
        # dedupe
        seen, out = set(), []
        for u in urls:
            if u not in seen:
                seen.add(u); out.append(u)
        return out

    def _looks_report(u: str) -> bool:
        ul = u.lower()
        return (
            ul.endswith(".pdf")
            or "arxiv.org" in ul
            or "openai.com/" in ul
            or "cdn.openai.com/" in ul
            or "cloudfront.net/" in ul
            or any(k in ul for k in [
                "technical-report", "tech-report", "whitepaper", "white-paper", "paper",
                "/docs", "docs.", "/blog", "blog.", "/research", "research.",
                "/index/", "/posts/",
            ])
        )

    full_texts = []

    def _append_link_only(u: str, how: str = "link-only"):
        full_texts.append({"arxiv_id": u, "full_text": "", "fetch_method": how})

    # 1) README 안의 링크들
    for u in _extract_urls(gh.get("readme", "")):
        if not _looks_report(u):
            continue
        try:
            text, used_url, how = _robust_fetch_report(u)
            full_texts.append({"arxiv_id": used_url, "full_text": text, "fetch_method": how})
        except Exception:
            _append_link_only(u)

    # 2) 리포지토리 내의 PDF 파일들
    repo = gh.get("repo", "")
    branch = gh.get("branch", "main")
    for p in (gh.get("files") or []):
        if str(p).lower().endswith(".pdf"):
            raw = f"https://raw.githubusercontent.com/{repo}/{branch}/{p}"
            try:
                text, used_url, how = _robust_fetch_report(raw)
                full_texts.append({"arxiv_id": used_url, "full_text": text, "fetch_method": how})
            except Exception:
                _append_link_only(raw)

    # 3) 저장/병합
    if full_texts:
        base = hf_id.replace("/", "_").lower()
        out = Path(output_dir) / f"reports_fulltext_{base}.json"
        merged = []
        if out.exists():
            try:
                merged = (json.load(open(out, encoding="utf-8")).get("full_texts") or [])
            except Exception:
                merged = []
        json.dump({"model_id": hf_id, "full_texts": merged + full_texts},
                  open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"📄 Reports merged to: {out}")


# === [ADD] Hugging Face JSON에서 리포트/블로그 링크를 수확해 reports_fulltext_{hf}.json에 병합 ===
def harvest_reports_from_hf_json(hf: dict, hf_id: str, output_dir: str | Path = "."):
    """
    Hugging Face fetcher 산출물(hf json)에서 논문/블로그/리포트 링크를 추출해
    텍스트를 확보(강화 폴백 사용)하고 reports_fulltext_{hf}.json에 병합 저장한다.
    - cardData.content (README 마크다운)
    - cardData.links.* (paper/homepage/blog/documentation 등)
    - readme (fetcher가 넣어준 원문이 있으면)
    """
    import re
    from pathlib import Path

    def _uniq(seq):
        return list(dict.fromkeys([s for s in seq if isinstance(s, str) and s.strip()]))

    def _extract_urls_from_text(txt: str) -> list[str]:
        if not txt: return []
        # 마크다운/HTML 앵커 안의 URL까지 포함해 폭넓게 포착
        urls = re.findall(r'https?://[^\s)>"\'\]]+', txt)
        return _uniq(urls)

    def _looks_report_like(u: str) -> bool:
        ul = u.lower()
        return (
            ul.endswith(".pdf")
            or "arxiv.org" in ul
            or any(k in ul for k in [
                "/paper", "paper/", "whitepaper", "technical-report", "tech-report",
                "/docs", "docs.", "/blog", "blog.", "/research", "research.",
                "/index/", "/posts/", # 새 블로그 구조 대응
            ])
        )

    if not isinstance(hf, dict):
        return

    # 1) 텍스트 소스 모으기
    content = ""
    content += (hf.get("readme") or "")
    cd = hf.get("cardData") or {}
    if isinstance(cd, dict):
        content += "\n" + (cd.get("content") or "")

    urls_text = _extract_urls_from_text(content)

    # 2) links 필드에서 URL 추가
    links_obj = (cd.get("links") or {}) if isinstance(cd, dict) else {}
    urls_links = []
    for fld in ("paper","papers","homepage","repository","documentation","blog","arxiv"):
        val = links_obj.get(fld)
        if isinstance(val, str):
            urls_links.append(val)
        elif isinstance(val, list):
            urls_links.extend([str(x) for x in val])

    # 3) 통합 후보
    candidates = _uniq([u for u in (urls_text + urls_links) if _looks_report_like(u)])
    if not candidates:
        return

    # 4) 강화 폴백으로 텍스트 수집
    full_texts = []
    for u in candidates:
        try:
            text, used_url, how = _robust_fetch_report(u)  # ← 이전에 추가한 강화 페처 재사용
        except NameError:
            # 만약 _robust_fetch_report가 아직 없다면 최소 폴백
            text, used_url, how = "", u, "missing-robust-fetcher"
        full_texts.append({
            "arxiv_id": used_url,
            "full_text": text,
            "fetch_method": how
        })

    # 5) 병합 저장
    base = hf_id.replace("/", "_").lower()
    out = Path(output_dir) / f"reports_fulltext_{base}.json"
    merged = []
    if out.exists():
        try:
            old = json.load(open(out, encoding="utf-8"))
            merged = (old.get("full_texts") or [])
        except Exception:
            merged = []
    json.dump({"model_id": hf_id, "full_texts": merged + full_texts},
              open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"📄 Reports merged to: {out} (HF sources)")



def run_all_fetchers(user_input: str):
    import os, json, requests
    from pathlib import Path
    from inference import run_inference

    outdir = make_model_dir(user_input)
    print(f"📁 Output path: {outdir}")

    info = extract_model_info(user_input)
    hf_id = gh_id = None
    found_rank_hf = found_rank_gh = None
    full = info['full_id']
    hf_cand = info['hf_id']

    hf_ok = test_hf_model_exists(hf_cand)
    gh_ok = test_github_repo_exists(full)
    print(f"1️⃣ HF: {hf_ok}, GH: {gh_ok}")

    if hf_ok:
        hf_id = hf_cand
        found_rank_hf = 1
    if gh_ok:
        gh_id = full
        found_rank_gh = 1

    # --- 2nd priority: HF card/README direct GitHub links (RESTORED) ---
    if hf_ok and not gh_id:
        gh_link = find_github_in_huggingface(hf_cand)
        print(f"🔍 2nd-priority HF→GH direct link: {gh_link}")
        if gh_link and test_github_repo_exists(gh_link):
            if _looks_relevant_repo(gh_link, hf_cand) or _rescue_accept_with_gpt(gh_link, hf_cand):
                gh_id = gh_link
                found_rank_gh = 2

    # --- 3rd priority: GPT with HF README context ---
    if hf_ok and not gh_id:
        cands = gpt_guess_github_from_hf_with_context(hf_cand)
        print(f"⏳ 3rd-priority GPT(HF README) candidates: {cands}")
        best = None; best_score = -10**9
        for rep in cands:
            if not test_github_repo_exists(rep):
                continue
            relevant = _looks_relevant_repo(rep, hf_cand) or _rescue_accept_with_gpt(rep, hf_cand)
            if not relevant:
                print(f"🚫 skip (failed relevance/org-sim): {rep}")
                continue
            s = _score_repo(rep, hf_cand)
            if s > best_score:
                best, best_score = rep, s
        if best:
            gh_id = best
            found_rank_gh = 3
            print(f"✅ Adopted 3rd-priority GH: {best} (score={best_score})")

    # --- 4th priority: Web search → filter/score ---
    if hf_ok and not gh_id:
        cands = web_search_github_candidates(hf_cand)
        print(f"⏳ 4th-priority Web-search candidates: {cands}")
        best = None; best_score = -10**9
        for rep in cands:
            if not test_github_repo_exists(rep):
                continue
            relevant = _looks_relevant_repo(rep, hf_cand) or _rescue_accept_with_gpt(rep, hf_cand)
            if not relevant:
                print(f"🚫 skip (failed relevance/org-sim): {rep}")
                continue
            s = _score_repo(rep, hf_cand)
            if s > best_score:
                best, best_score = rep, s
        if best:
            gh_id = best
            found_rank_gh = 4
            print(f"✅ Adopted 4th-priority GH: {best} (score={best_score})")

    # --- 2nd priority (GH→HF): GPT guess (as-is; with warnings) ---
    if gh_ok and not hf_id:
        guess_hf = gpt_guess_huggingface_from_github(full)
        print(f"⏳ 2nd-priority GPT GH→HF guess: {guess_hf}")
        if guess_hf and test_hf_model_exists(guess_hf):
            hf_id = guess_hf
            found_rank_hf = 2
            print("⚠️ GPT-derived guess. Please verify the model ID is correct.")

    # ───────────────── HF processing ─────────────────
    data = {}
    if hf_id:
        rank_hf = found_rank_hf or 'none'
        print(f"✅ HF model: {hf_id} (found at priority: {rank_hf})")
        data = huggingface_fetcher(hf_id, save_to_file=True, output_dir=outdir)
        # [ADD] HF README/card에서 리포트/블로그 텍스트 수확 (403/DNS 폴백 포함)
        try:
            harvest_reports_from_hf_json(data, hf_id, output_dir=outdir)
        except Exception as e:
            print("⚠️ HF report harvesting failed:", e)

        try:
            hf_filtered = filter_hf_features(hf_id, output_dir=outdir)
        except FileNotFoundError:
            hf_filtered = {}
            print("⚠️ Hugging Face JSON file not found; skipping filtering")

    # ───────────────── GitHub processing ─────────────────
    gh_data = {}
    if gh_id:
        rank_gh = found_rank_gh or 'none'
        print(f"✅ GH repo: {gh_id} (found at priority: {rank_gh})")
        try:
            gh_data = github_fetcher(gh_id, branch="main", save_to_file=True, output_dir=outdir) or {}
        except requests.exceptions.HTTPError:
            print("⚠️ Failed to access 'main' branch; retrying with 'master'...")
            try:
                gh_data = github_fetcher(gh_id, branch="master", save_to_file=True, output_dir=outdir) or {}
            except Exception as e:
                print("❌ 'master' branch also failed:", e)

        try:
            gh_filtered = filter_github_features(gh_id, output_dir=outdir)
        except FileNotFoundError:
            gh_filtered = {}
            print("⚠️ GitHub JSON file not found; skipping filtering")
    else:
        print("⚠️ No GitHub info")

    # ───────────────── Paper/Report aggregation (arXiv + external reports) ─────────────────
    if hf_id:
        try:
            arxiv_fetcher_from_model(hf_id, save_to_file=True, output_dir=outdir)
        except Exception as e:
            print("⚠️ arXiv fetch failed:", e)

    try:
        if gh_data and hf_id:
            harvest_reports_from_github_json(gh_data, hf_id, output_dir=outdir)
    except Exception as e:
        print("⚠️ GH report harvesting failed:", e)

    try:
        ax_filtered = filter_arxiv_features(hf_id, output_dir=outdir)
    except FileNotFoundError:
        ax_filtered = {}
        print("⚠️ No arXiv/report inputs found for dispatcher; skipping")

    try:
        rpt_filtered = filter_reports_features(hf_id, output_dir=outdir)
    except FileNotFoundError:
        rpt_filtered = {}
        print("⚠️ No report inputs found for reports dispatcher; skipping")

    # ─── GPT-based pretrained model detection + pipeline ───────────────
    base_model_id = gpt_detect_base_model(hf_id) if hf_id else None
    if base_model_id:
        print(f"🧱 Pretrained (base) model found by GPT: {base_model_id}")

        # 1) Hugging Face fetch/dispatch
        huggingface_fetcher(base_model_id, save_to_file=True, output_dir=outdir)
        from pretrain_hf_Dispatcher import filter_pretrain_hf
        filter_pretrain_hf(base_model_id, output_dir=outdir)

        # 2) GitHub for base model — same priorities (HF card → GPT → Web)
        base_gh = None

        # 2-1) Priority 1: direct from HF card/README parsing
        cand_gh = find_github_in_huggingface(base_model_id)
        if cand_gh and test_github_repo_exists(cand_gh):
            if _looks_relevant_repo(cand_gh, base_model_id) or _rescue_accept_with_gpt(cand_gh, base_model_id):
                base_gh = cand_gh
                print(f"✅ Base GH (P1 HF page): {base_gh}")
        else:
            print("ℹ️ Base GH not found via HF page; trying GPT/Web fallbacks...")

        # 2-2) Priority 2: GPT with HF README context
        if not base_gh:
            cands = gpt_guess_github_from_hf_with_context(base_model_id)
            print(f"⏳ Base GH P2 GPT(HF README) candidates: {cands}")
            best = None; best_score = -10**9
            for rep in cands or []:
                if not test_github_repo_exists(rep):
                    continue
                relevant = _looks_relevant_repo(rep, base_model_id) or _rescue_accept_with_gpt(rep, base_model_id)
                if not relevant:
                    print(f"🚫 skip (failed relevance/org-sim): {rep}")
                    continue
                s = _score_repo(rep, base_model_id)
                if s > best_score:
                    best, best_score = rep, s
            if best:
                base_gh = best
                print(f"✅ Base GH (P2 GPT): {base_gh} (score={best_score})")

        # 2-3) Priority 3: Web search
        if not base_gh:
            cands = web_search_github_candidates(base_model_id)
            print(f"⏳ Base GH P3 Web-search candidates: {cands}")
            best = None; best_score = -10**9
            for rep in cands or []:
                if not test_github_repo_exists(rep):
                    continue
                relevant = _looks_relevant_repo(rep, base_model_id) or _rescue_accept_with_gpt(rep, base_model_id)
                if not relevant:
                    print(f"🚫 skip (failed relevance/org-sim): {rep}")
                    continue
                s = _score_repo(rep, base_model_id)
                if s > best_score:
                    best, best_score = rep, s
            if best:
                base_gh = best
                print(f"✅ Base GH (P3 Web): {base_gh} (score={best_score})")

        if base_gh:
            try:
                github_fetcher(base_gh, save_to_file=True, output_dir=outdir)
                from pretrain_github_Dispatcher import filter_pretrain_gh
                filter_pretrain_gh(base_gh, output_dir=outdir)
            except Exception as e:
                print("⚠️ GH fetch/dispatch failed:", e)
        else:
            print("⚠️ Could not find the base model's GitHub repo after P1/P2/P3; skipping GH fetcher")

        # 3) arXiv (only if available)
        try:
            ax_ok = arxiv_fetcher_from_model(base_model_id,
                                             save_to_file=True,
                                             output_dir=outdir)
            if ax_ok:
                from pretrain_arxiv_Dispatcher import filter_pretrain_arxiv
                filter_pretrain_arxiv(base_model_id, output_dir=outdir)
            else:
                print("⚠️ Could not find a paper link; skipping arXiv fetcher")
        except Exception as e:
            print("⚠️ arXiv fetch/dispatch failed:", e)

        # 4) Pretrain reports dispatcher
        try:
            from pretrain_reports_Dispatcher import filter_pretrain_reports
            filter_pretrain_reports(base_model_id, output_dir=outdir)
        except FileNotFoundError:
            print("⚠️ No pretrain reports/arXiv inputs found for pretrain_reports; skipping")

    # ───────────────── Openness evaluation ─────────────────
    try:
        print("📝 Starting openness evaluation...")
        eval_res = evaluate_openness_from_files(
            full,
            base_dir=str(outdir),
            base_model_id=base_model_id
        )
        base = full.replace("/", "_")
        outfile = Path(outdir) / f"openness_score_{base}.json"
        print(f"✅ Openness evaluation complete. Result file: {outfile}")
    except Exception as e:
        print("⚠️ Error during openness evaluation:", e)

    # ───────────────── README-based inference ─────────────────
    readme_text = ""
    try:
        if isinstance(data, dict):
            readme_text = (
                data.get("readme") or
                ((data.get("cardData") or {}).get("content") if isinstance(data.get("cardData"), dict) else "")
            ) or ""
        if not readme_text.strip() and hf_id:
            base_for_file = hf_id.replace("/", "_").lower()
            hf_json_path = Path(outdir) / f"huggingface_{base_for_file}.json"
            if hf_json_path.exists():
                with open(hf_json_path, "r", encoding="utf-8") as f:
                    hf_json = json.load(f)
                cd = hf_json.get("cardData") or {}
                readme_text = (cd.get("content") or "")
    except Exception as e:
        print(f"⚠️ README extraction failed: {e}")

    if readme_text and readme_text.strip():
        try:
            os.environ["MODEL_OUTPUT_DIR"] = str(outdir)
            run_inference(readme_text, output_dir=outdir, keep_code=True)
        except Exception as e:
            print("⚠️ Failed to run inference:", e)
    else:
        print("⚠️ README is empty; skipping inference stage")


def make_model_dir(user_input: str) -> Path:
    info = extract_model_info(user_input)
    base = info["hf_id"]
    safe = re.sub(r'[<>:"/\\|?*\s]+', "_", base)
    path = Path(safe)
    path.mkdir(parents=True, exist_ok=True)
    return path

###################################################################
if __name__ == "__main__":
    try:
        n = int(input("🔢 Number of models to process: ").strip())
    except ValueError:
        print("Please enter a number."); exit(1)

    models: list[str] = []
    for i in range(1, n + 1):
        m = input(f"[{i}/{n}] 🌐 HF/GH URL or org/model: ").strip()
        if m:
            models.append(m)

    print("\n🚀 Processing", len(models), "models sequentially.\n")

    for idx, user_input in enumerate(models, 1):
        print(f"\n======== {idx}/{len(models)} ▶ {user_input} ========")
        try:
            model_dir = make_model_dir(user_input)
            print(f"📁 Directory to create/use: {model_dir}")
            run_all_fetchers(user_input)

            info  = extract_model_info(user_input)
            hf_id = info["hf_id"]
            if test_hf_model_exists(hf_id):
                with open(model_dir / "identified_model.txt", "w", encoding="utf-8") as f:
                    f.write(hf_id)
                print(f"✅ Saved model ID: {model_dir / 'identified_model.txt'}")

        except Exception as e:
            print("❌ Error encountered while processing:", e)
            continue

    print("\n🎉 All tasks completed.")
