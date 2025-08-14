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
# from openness_Evaluator import evaluate_openness
from github_Dispatcher import filter_github_features
from arxiv_Dispatcher import filter_arxiv_features
from huggingface_Dispatcher import filter_hf_features
from openness_Evaluator import evaluate_openness_from_files
from inference import run_inference

import html

# Load environment variables and initialize OpenAI client
dotenv_path = os.path.join(os.getcwd(), '.env')
load_dotenv(dotenv_path)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    import re as _re
    return set(t for t in _re.sub(r"[^a-z0-9]+"," ", s.lower()).split() if len(t) >= 3)

def _looks_relevant_repo(repo: str, hf_id: str) -> bool:
    """Allow family/series repos; block monorepos and tool/demo/docs repos."""
    rl = repo.lower()
    if rl in MONOREPO_DENYLIST:
        return False
    name = rl.split("/",1)[1]
    toks_hf = _tokens(hf_id)
    toks_repo = _tokens(name)
    if not (toks_hf & toks_repo):
        return False
    if any(k in name for k in BAD_REPO_KEYWORDS):
        # If family token matches strongly, allow it (series repo).
        fam_hit = any(len(t)>=5 and t in toks_repo for t in toks_hf)
        if not fam_hit:
            return False
    return True

def _score_repo(repo: str, hf_id: str) -> int:
    """Simple score: org match + token overlap - noise keywords."""
    rl = repo.lower()
    org, name = rl.split("/",1)
    org_hf = hf_id.split("/")[0].lower() if "/" in hf_id else ""
    score = 0
    if org == org_hf: score += 5
    toks_hf   = _tokens(hf_id)
    toks_repo = _tokens(name)
    score += 2 * len(toks_hf & toks_repo)
    if any(k in name for k in ("model","models","llm")): score += 2
    if any(k in name for k in BAD_REPO_KEYWORDS): score -= 2
    return score

# ──────────────────────────────────────────────────────────────
# Estimate the pretrained (base) model with GPT-4o
def gpt_detect_base_model(hf_id: str) -> str | None:
    """
    • If the input model is a finetuned model → return the pretrained model ID
    • If the input is already a pretrained (original) model → return None
    • If unsure and GPT returns null → return None
    """
    import textwrap

    def _hf_card_readme(mid: str, max_len: int = 12000) -> str:
        try:
            card = requests.get(
                f"https://huggingface.co/api/models/{mid}?full=true"
            ).json().get("cardData", {}) or {}
            txt = (card.get("content") or "")[:max_len]
            for br in ["main", "master"]:
                r = requests.get(f"https://huggingface.co/{mid}/raw/{br}/README.md")
                if r.status_code == 200:
                    txt += "\n\n" + r.text[:max_len]
                    break
            return txt
        except Exception:
            return ""

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
    if not ctx:
        return None

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt_sys},
                {"role": "user",   "content": ctx}
            ],
            temperature=0
        )
        pred = json.loads(resp.choices[0].message.content)
        pre_id = pred.get("pretrain_model")
        # Validation: use only if it exists and differs from the input
        if pre_id and isinstance(pre_id, str):
            pre_id = pre_id.strip()
            if pre_id.lower() != hf_id.lower() and test_hf_model_exists(pre_id):
                return pre_id
    except Exception as e:
        print("⚠️ Failed to detect pretrained model via GPT:", e)
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

# 2. Existence tests
def test_hf_model_exists(model_id: str) -> bool:
    resp = requests.get(f"https://huggingface.co/api/models/{model_id}")
    return resp.status_code == 200

def test_github_repo_exists(repo: str) -> bool:
    resp = requests.get(f"https://api.github.com/repos/{repo}")
    return resp.status_code == 200

# 3. Link parsing helper (★ preserve original case)
def extract_repo_from_url(url: str) -> str:
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) >= 2:
        repo = parts[1].split("?")[0].split("#")[0].replace(".git", "")
        return f"{parts[0]}/{repo}"  # preserve original case
    return ""

# 4. HF page → GitHub (collect candidates → lowercase compare/score → return original casing)
def find_github_in_huggingface(model_id: str) -> str | None:
    """
    Collect all possible GitHub link candidates from the HF model card and score them,
    then return the most plausible repository.
    Comparison/scoring is done in lowercase; return value preserves original case.
    """
    def _extract_repo_from_url_preserve(url: str) -> str | None:
        try:
            p = urlparse(url)
            if "github.com" not in p.netloc.lower():
                return None
            seg = p.path.strip("/").split("/")
            if len(seg) >= 2:
                repo = seg[1].split("?")[0].split("#")[0].replace(".git", "")
                return f"{seg[0]}/{repo}"  # preserve original case
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
        card = requests.get(
            f"https://huggingface.co/api/models/{model_id}?full=true"
        ).json().get("cardData", {}) or {}

        hf_org = model_id.split("/")[0] if "/" in model_id else ""
        toks = _tokenize(model_id)

        # lower → original mapping
        cand_map: dict[str, str] = {}

        def _add_candidate(rep: str | None):
            if not rep:
                return
            cand_map.setdefault(rep.lower(), rep)  # dedup by lower key; keep original value

        # 1) links.repository / links.homepage
        for field in ["repository", "homepage"]:
            links = (card.get("links", {}) or {}).get(field)
            if isinstance(links, str):
                _add_candidate(_extract_repo_from_url_preserve(links))
            elif isinstance(links, list):
                for u in links:
                    _add_candidate(_extract_repo_from_url_preserve(str(u)))

        # 2) model card content
        content = card.get("content", "") or ""
        for url in re.findall(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", content):
            _add_candidate(_extract_repo_from_url_preserve(url))

        # 3) raw README
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

        # 4) score only the candidates that exist (comparison in lower); return original case
        best_lower, best_score = None, -10**9
        for rep_lower, rep_orig in cand_map.items():
            if not test_github_repo_exists(rep_orig):  # existence check with original case
                continue
            s = _score(rep_lower, hf_org, toks)
            if s > best_score:
                best_lower, best_score = rep_lower, s

        return cand_map[best_lower] if best_lower else None
    except Exception:
        return None

# 5. GitHub page → HF (fallback: raw README and HTML)
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
                    m_md = re.search(r"\\[.*?\\]\\((https?://huggingface\.co/[\w\-/\.]+)\\)", r.text, re.IGNORECASE)
                    if m_md:
                        candidate = extract_model_info(m_md.group(1))["hf_id"]
                        if not candidate.startswith('collections/'):
                            return candidate
                    m_html = re.search(r'<a\\s+href="https?://huggingface\.co/([\w\-/\.]+)"', r.text, re.IGNORECASE)
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
        print("⚠️ Error during GPT HF→GH guess:", e)
    return None

def gpt_guess_huggingface_from_github(gh_id: str) -> str:
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
        print("⚠️ Error during GPT GH→HF guess:", e)
    return None

# --- NEW: Priority 3 (HF README context → GPT) ---
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
            ).json().get("cardData", {}) or {}
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
        print("⚠️ GPT HF-context guess failed:", e)
        return []

# --- NEW: Priority 4 (Web search → candidates) ---
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
                j = r.json()
                for it in j.get("results", []):
                    u = it.get("url","")
                    if "github.com" in u:
                        m = _re.search(r"github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", u)
                        if m: urls.append(m.group(1))
            except Exception:
                pass
        return list(dict.fromkeys(urls))[:5]

    # 2) GitHub Search API (fallback)
    gh_tok = _os.getenv("GITHUB_TOKEN")
    headers = {"Accept":"application/vnd.github+json"}
    if gh_tok: headers["Authorization"] = f"Bearer {gh_tok}"

    repos = []
    for q in queries:
        try:
            r = _req.get("https://api.github.com/search/repositories",
                         params={"q":q, "sort":"stars", "order":"desc", "per_page":5},
                         headers=headers, timeout=15)
            j = r.json()
            for item in j.get("items", [])[:5]:
                full = item.get("full_name")
                if full: repos.append(full)
        except Exception:
            pass
    return list(dict.fromkeys(repos))[:5]

# --- Utility: harvest technical reports from GitHub JSON into reports_fulltext_{hf}.json
def harvest_reports_from_github_json(gh: dict, hf_id: str, output_dir: str | Path = "."):
    """
    From a GitHub fetcher's JSON (README + file list), harvest technical reports and
    merge them into reports_fulltext_{hf}.json. Supports:
      - Links in README pointing to PDFs or report-like pages (company blog/docs/research)
      - Repository-embedded PDFs (download and OCR text via PyMuPDF)
    """
    import re, requests, fitz
    from pathlib import Path

    def _extract_urls(text: str) -> list[str]:
        urls = re.findall(r'https?://[^\s)>\"]+', text or "")
        return list(dict.fromkeys(urls))

    def _looks_report(u: str) -> bool:
        ul = u.lower()
        return ul.endswith(".pdf") or any(k in ul for k in [
            "technical-report", "tech-report", "whitepaper", "white-paper", "paper",
            "/docs", "docs.", "/blog", "blog.", "/research", "research."
        ])

    def _fetch_pdf(url: str) -> str:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        with fitz.open(stream=r.content, filetype="pdf") as doc:
            return "\n".join(p.get_text() for p in doc)

    def _fetch_html(url: str) -> str:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        h = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", r.text)
        t = re.sub(r"(?is)<[^>]+>", " ", h)
        t = re.sub(r"\s+", " ", t)
        return t[:800_000]

    full_texts = []

    # README links
    for u in _extract_urls(gh.get("readme", "")):
        if not _looks_report(u):
            continue
        try:
            txt = _fetch_pdf(u) if u.lower().endswith(".pdf") else _fetch_html(u)
            if txt.strip():
                full_texts.append({"arxiv_id": u, "full_text": txt})
        except Exception as e:
            print("⚠️ report fetch failed:", u, e)

    # Repo PDFs (raw.githubusercontent.com/{repo}/{branch}/{path})
    repo = gh.get("repo", "")
    branch = gh.get("branch", "main")
    for p in (gh.get("files") or []):
        if str(p).lower().endswith(".pdf"):
            url = f"https://raw.githubusercontent.com/{repo}/{branch}/{p}"
            try:
                txt = _fetch_pdf(url)
                if txt.strip():
                    full_texts.append({"arxiv_id": url, "full_text": txt})
            except Exception as e:
                print("⚠️ report fetch failed:", url, e)

    # Save/merge as reports_fulltext_{hf}.json
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

def run_all_fetchers(user_input: str):
    # Place imports required for inference within the function to avoid dependency tangles
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

    if hf_ok and not gh_id:
        gh_link = find_github_in_huggingface(hf_cand)
        print(f"🔍 2nd-priority HF→GH link: {gh_link}")
        if gh_link and test_github_repo_exists(gh_link):
            gh_id = gh_link
            found_rank_gh = 2

    if gh_ok and not hf_id:
        hf_link = find_huggingface_in_github(full)
        print(f"🔍 2nd-priority GH→HF link: {hf_link}")
        if hf_link and test_hf_model_exists(hf_link):
            hf_id = hf_link
            found_rank_hf = 2

    # --- 3rd priority: GPT with HF README context (NEW) ---
    if hf_ok and not gh_id:
        cands = gpt_guess_github_from_hf_with_context(hf_cand)
        print(f"⏳ 3rd-priority GPT(HF README) candidates: {cands}")
        best = None; best_score = -10**9
        for rep in cands:
            if not test_github_repo_exists(rep):
                continue
            if not _looks_relevant_repo(rep, hf_cand):
                print(f"🚫 skip (irrelevant/monorepo): {rep}")
                continue
            s = _score_repo(rep, hf_cand)
            if s > best_score: best, best_score = rep, s
        if best:
            gh_id = best
            found_rank_gh = 3
            print(f"✅ Adopted 3rd-priority GH: {best} (score={best_score})")

    # --- 4th priority: Web search → filter/score (NEW) ---
    if hf_ok and not gh_id:
        cands = web_search_github_candidates(hf_cand)
        print(f"⏳ 4th-priority Web-search candidates: {cands}")
        best = None; best_score = -10**9
        for rep in cands:
            if not test_github_repo_exists(rep):
                continue
            if not _looks_relevant_repo(rep, hf_cand):
                print(f"🚫 skip (irrelevant/monorepo): {rep}")
                continue
            s = _score_repo(rep, hf_cand)
            if s > best_score: best, best_score = rep, s
        if best:
            gh_id = best
            found_rank_gh = 4
            print(f"✅ Adopted 4th-priority GH: {best} (score={best_score})")

    if gh_ok and not hf_id:
        guess_hf = gpt_guess_huggingface_from_github(full)
        print(f"⏳ 3rd-priority GPT GH→HF guess: {guess_hf}")
        if guess_hf and test_hf_model_exists(guess_hf):
            hf_id = guess_hf
            found_rank_hf = 3
            print("⚠️ GPT-derived guess. Please verify the model ID is correct.")

    # ───────────────── HF processing ─────────────────
    data = {}
    if hf_id:
        rank_hf = found_rank_hf or 'none'
        print(f"✅ HF model: {hf_id} (found at priority: {rank_hf})")
        data = huggingface_fetcher(hf_id, save_to_file=True, output_dir=outdir)
        try:
            hf_filtered = filter_hf_features(hf_id, output_dir=outdir)
        except FileNotFoundError:
            hf_filtered = {}
            print("⚠️ Hugging Face JSON file not found; skipping filtering")
    else:
        print("⚠️ No Hugging Face info")

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
    # 1) arXiv PDFs (from HF tags)
    if hf_id:
        try:
            arxiv_fetcher_from_model(hf_id, save_to_file=True, output_dir=outdir)
        except Exception as e:
            print("⚠️ arXiv fetch failed:", e)

    # 2) Technical reports from GitHub (README links + repo PDFs) → merge into reports_fulltext_{hf}.json
    try:
        if gh_data and hf_id:
            harvest_reports_from_github_json(gh_data, hf_id, output_dir=outdir)
    except Exception as e:
        print("⚠️ GH report harvesting failed:", e)

    # 3) Dispatcher (will run even if no arXiv paper, as long as any reports exist)
    try:
        ax_filtered = filter_arxiv_features(hf_id, output_dir=outdir)
    except FileNotFoundError:
        ax_filtered = {}
        print("⚠️ No arXiv/report inputs found for dispatcher; skipping")

    # ─── GPT-based pretrained model detection + pipeline ───────────────
    base_model_id = gpt_detect_base_model(hf_id) if hf_id else None
    if base_model_id:
        print(f"🧱 Pretrained (base) model found by GPT: {base_model_id}")

        # 1) Hugging Face fetch/dispatch
        huggingface_fetcher(base_model_id, save_to_file=True, output_dir=outdir)
        from pretrain_hf_Dispatcher import filter_pretrain_hf
        filter_pretrain_hf(base_model_id, output_dir=outdir)

        # 2) GitHub (only if available)
        base_gh = find_github_in_huggingface(base_model_id)
        if base_gh:
            try:
                github_fetcher(base_gh, save_to_file=True, output_dir=outdir)
                from pretrain_github_Dispatcher import filter_pretrain_gh
                filter_pretrain_gh(base_gh, output_dir=outdir)
            except Exception as e:
                print("⚠️ GH fetch/dispatch failed:", e)
        else:
            print("⚠️ Could not find the base model's GitHub repo; skipping GH fetcher")

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
    else:
        base_model_id = None  # GPT returned null → no pretrained model

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

    # ───────────────── README-based inference (save inside the folder) ─────────────────
    # 1) First try from the in-memory fetcher result
    readme_text = ""
    try:
        if isinstance(data, dict):
            # Safely extract according to the huggingface_fetcher return schema
            readme_text = (
                data.get("readme") or
                ((data.get("cardData") or {}).get("content") if isinstance(data.get("cardData"), dict) else "")
            ) or ""
        # 2) If empty, fallback to file
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
            # Ensure result JSONs are created inside the model folder
            os.environ["MODEL_OUTPUT_DIR"] = str(outdir)  # optional safety guard
            run_inference(readme_text, output_dir=outdir, keep_code=True)
        except Exception as e:
            print("⚠️ Failed to run inference:", e)
    else:
        print("⚠️ README is empty; skipping inference stage")


def make_model_dir(user_input: str) -> Path:
    info = extract_model_info(user_input)
    base = info["hf_id"]                         # e.g.) 'bigscience/bloomz-560m'
    safe = re.sub(r'[<>:"/\\|?*\s]+', "_", base) # e.g.) 'bigscience_bloomz-560m'
    path = Path(safe)
    path.mkdir(parents=True, exist_ok=True)
    return path

###################################################################
# if __name__ == "__main__":
#     user_input = input("🌐 HF/GH URL 또는 org/model: ").strip()
#     model_dir = make_model_dir(user_input)
#     print(f"📁 생성/사용할 폴더: {model_dir}")
#     run_all_fetchers(user_input)
#
#     info = extract_model_info(user_input)
#     hf_id = info['hf_id']
#
#     if test_hf_model_exists(hf_id):
#         with open(model_dir / "identified_model.txt", "w", encoding="utf-8") as f:
#             f.write(hf_id)
#         print(f"✅ 모델 ID 저장 완료: {model_dir / 'identified_model.txt'}")
#######################################################################
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
            # Log the error and continue to the next model
            continue

    print("\n🎉 All tasks completed.")


    # # ✅ Run inference immediately
    # # prompt = input("📝 Enter a prompt: ")
    # # run_inference(hf_id, prompt)
