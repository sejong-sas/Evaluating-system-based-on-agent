# -*- coding: utf-8 -*-
# model_Identifier.py (case-robust, model-agnostic)
# 변경점(요약)
#  - GitHub 레포지토리 리졸버 전면 개편(정적 org alias 제거, 강토큰/버전 가드, 3rd-party 억제, 패밀리 허용)
#  - HF 카드/README에서 찾은 org를 자동 동맹셋에 추가
#  - P0~P3 파이프라인 통합 + 점수화 + 임계치
#  - 프리트레인(베이스) 경로도 동일 함수 재사용
#  - OpenAI 호출에서 temperature 제거 (o3-mini 호환)
#
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
import shutil
import glob
import sys, datetime, contextlib

# ──────────────────────────────────────────────────────────────
# Env & client
dotenv_path = os.path.join(os.getcwd(), '.env')
load_dotenv(dotenv_path)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Warn/keys helpers ----------
def _warn(msg: str):
    print(msg, flush=True)

def _has_openai_key() -> bool:
    if not os.getenv("OPENAI_API_KEY"):
        _warn("🔑 OPENAI_API_KEY is not set — GPT steps will be skipped.")
        return False
    return True

# ---------- Canonical naming helpers ----------
def _norm_base(mid: str) -> str:
    return (mid or "").replace("/", "_").lower()

def _raw_base(mid: str) -> str:
    return (mid or "").replace("/", "_")

def _ensure_lowercase_alias(kind: str, key: str, outdir: str | Path):
    outdir = Path(outdir)
    raw = outdir / f"{kind}_{_raw_base(key)}.json"
    low = outdir / f"{kind}_{_norm_base(key)}.json"
    if low.exists():
        return
    if raw.exists():
        try:
            raw.rename(low)
        except OSError:
            shutil.copyfile(raw, low)

def _open_json_anycase_by_key(kind: str, key: str, outdir: str | Path) -> dict:
    outdir = Path(outdir)
    low = outdir / f"{kind}_{_norm_base(key)}.json"
    raw = outdir / f"{kind}_{_raw_base(key)}.json"
    if low.exists():
        return json.load(open(low, encoding="utf-8"))
    if raw.exists():
        return json.load(open(raw, encoding="utf-8"))
    name = (key.split("/", 1)[1] if "/" in key else key)
    pats = [
        str(outdir / f"{kind}_*{name.lower()}*.json"),
        str(outdir / f"{kind}_*{name}*.json"),
    ]
    for pat in pats:
        cands = sorted(glob.glob(pat))
        if cands:
            return json.load(open(cands[0], encoding="utf-8"))
    raise FileNotFoundError(str(low))

def _ensure_common_aliases(hf_id: str | None, gh_id: str | None, outdir: str | Path):
    if hf_id:
        for kind in ("huggingface", "arxiv_fulltext", "reports_fulltext",
                     "huggingface_filtered_final", "reports_filtered_final",
                     "arxiv_filtered_final"):
            _ensure_lowercase_alias(kind, hf_id, outdir)
    if gh_id:
        for kind in ("github", "github_filtered_final"):
            _ensure_lowercase_alias(kind, gh_id, outdir)

# ---------- Heuristics & utils ----------
BAD_REPO_KEYWORDS = {
    # 도구/데모/클라이언트/바인딩/포트/양자화/런타임/컴파일러/엔진/배포/샘플/문서/리스트 등
    "api","client","sdk","demo","website","docs","doc","notebook","colab",
    "examples","sample","bench","leaderboard","eval","evaluation","convert",
    "export","deploy","space","slim","angelslim","awesome","papers","paper-list",
    "binding","bindings","plugin","bot","server","service","gateway","proxy",

    # 포팅/런타임/백엔드/가속/온디바이스
    "cpp","rust","go","java","swift","tfjs","wasm","webgpu","ncnn","ggml","gguf",
    "coreml","tflite","android","ios","openvino","tensorrt","onnx","web",
    "kobold","oobabooga","text-generation-webui","exllama","exllamav2","llamacpp",
    "lmstudio","ollama","mlc","mnn","ktransformers",

    # 양자화/포맷/변환 계열
    "quant","quantization","gptq","awq","bnb","exl2","marlin",

    # 추론/서빙 전용 냄새 (과도한 경우 감점)
    "inference","serving","runtime","engine","accelerate","adapter","adapter-transformers"
}

# 후보 레포 이름에서 무시할 흔한 토큰(거의 예외 없이 등장)
IGNORE_NAME_TOKENS = {
    "ai","llm","model","models","base","chat","instruct","instruction","sft","rl",
    "eval","evaluation","dev","it","hf","hub","repo","official","release",
    "torch","pytorch","jax","tensorflow","tf","flax","jaxlib","cuda","rocm",
    "train","training","pretrain","finetune","fine-tune","fine-tuning","post-train",
    "open","open-source","opensource","research","lab","team","project","examples",
    "v","v2","v3","v4","v5","beta","alpha","rc","preview","early","nightly","latest",
    "generator","pipeline","scripts","configs","tools"
}

def _tokens(s: str) -> set[str]:
    import re as _re
    return set(t for t in _re.sub(r"[^a-z0-9]+"," ", (s or "").lower()).split() if len(t) >= 2)

def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def _name_similarity_ratio(hf_id: str, repo: str) -> float:
    hf_name = (hf_id.split("/",1)[1] if "/" in hf_id else hf_id)
    repo_name = (repo.split("/",1)[1] if "/" in repo else repo)
    a = _norm_name(hf_name)
    b = _norm_name(repo_name)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()

def _org_match_or_similar_name(a: str, b: str, thresh: float = 0.78) -> bool:
    a = (a or "").lower(); b = (b or "").lower()
    if not a or not b:
        return False
    if a == b:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= thresh

def _extract_version_tokens(model_id: str) -> dict:
    """
    모델명에서 버전/사이즈/접미사 토큰을 추출.
    ex) 'falcon3-7b-instruct' → core: ['falcon3','falcon'], sizes: ['7b'], suffix: ['instruct'], majors: ['3']
    """
    name = (model_id.split("/",1)[1] if "/" in model_id else model_id).lower()
    nm = re.sub(r"[^a-z0-9\.\-]+","-", name)
    pieces = [p for p in re.split(r"[-_]+", nm) if p]

    sizes = set([p for p in pieces if re.fullmatch(r"(\d+)([bkmt])", p)])
    suffix = set([p for p in pieces if p in {"instruct","chat","base","it","sft"}])
    majors = set()

    # 'qwen2.5', 'llama3.1' → majors {'2.5','3.1'} + {'2','3'}
    for p in pieces:
        m = re.match(r".*?(\d+(?:\.\d+)*)", p)
        if m:
            majors.add(m.group(1))
            majors.add(m.group(1).split(".")[0])

    # core tokens: 접두사+주요버전 결합 / 접두사만
    prefix = re.sub(r"[^a-z]+","", pieces[0]) if pieces else ""
    core = set()
    for mj in list(majors):
        if prefix:
            core.add(f"{prefix}{mj.replace('.','')}")
    if prefix:
        core.add(prefix)

    strong = set()
    strong |= sizes
    strong |= suffix
    strong |= core
    # '3-7b' 같이 결합
    for mj in majors:
        for sz in sizes:
            strong.add(f"{mj}-{sz}")
    # collapse 전체 문자열도 강토큰에 포함
    strong.add(_norm_name(name))

    return {
        "sizes": sorted(sizes),
        "suffix": sorted(suffix),
        "majors": sorted(majors),
        "core": sorted(core),
        "strong": sorted(strong)
    }

def _has_conflicting_version(repo_name: str, target_majors: list[str]) -> bool:
    """
    레포 이름에 다른 주버전(또는 소수점 버전)이 강하게 드러나면 충돌로 간주.
    ex) target: ['3','3.1']인데 이름에 '2.5'나 'v2' 등이 들어있으면 True
    """
    rn = repo_name.lower()
    # vX / X.Y / X 형태 탐지
    nums = set(re.findall(r"v(\d+(?:\.\d+)*)", rn))
    nums |= set(re.findall(r"(\d+\.\d+)", rn))
    nums |= set(re.findall(r"(^|[^a-z])(\d+)([^a-z]|$)", rn))
    found = set()
    for t in nums:
        if isinstance(t, tuple):
            t = t[1]
        found.add(str(t))
        found.add(str(t).split(".")[0])
    # 타겟과 교집합 없고, 뭔가 숫자가 있으면 충돌로 봄
    return bool(found) and not (found & set([m for m in target_majors]))

def _count_token_hits(text: str, tokens: list[str]) -> int:
    tl = (text or "").lower()
    return sum(1 for t in tokens if t and t in tl)

# ──────────────────────────────────────────────────────────────
# GitHub API helpers
def test_hf_model_exists(model_id: str) -> bool:
    resp = requests.get(f"https://huggingface.co/api/models/{model_id}")
    if resp.status_code in (401, 403):
        _warn("🔑 Hugging Face API says 401/403 — model may be private. Set HF_TOKEN in .env if you have access.")
    elif resp.status_code == 429:
        _warn("⏳ HF API rate limited (429) — try later or set HF_TOKEN.")
    return resp.status_code == 200

def test_github_repo_exists(repo: str) -> bool:
    headers = {"Accept": "application/vnd.github+json"}
    tok = os.getenv("GITHUB_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    resp = requests.get(f"https://api.github.com/repos/{repo}", headers=headers)
    if resp.status_code == 403 and not tok:
        _warn("🔑 GitHub API rate limited (403) — set GITHUB_TOKEN in .env.")
    return resp.status_code == 200

def _fetch_repo_snapshot(repo: str) -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    tok = os.getenv("GITHUB_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"

    readme = ""
    for br in ["main", "master"]:
        try:
            r = requests.get(f"https://raw.githubusercontent.com/{repo}/{br}/README.md",
                             headers=headers, timeout=15)
            if r.status_code == 200:
                readme = r.text
                break
        except Exception:
            pass

    paths = []
    for br in ["main", "master"]:
        try:
            tr = requests.get(f"https://api.github.com/repos/{repo}/git/trees/{br}?recursive=1",
                              headers=headers, timeout=15)
            if tr.status_code == 200:
                j = tr.json()
                for it in j.get("tree", []):
                    p = str(it.get("path", ""))
                    if p:
                        paths.append(p)
                break
        except Exception:
            pass

    return {"readme": (readme or "")[:12000], "paths": paths[:600]}

# ──────────────────────────────────────────────────────────────
# Robust report fetcher (unchanged interface, minor refactor)
def _robust_fetch_report(url: str) -> tuple[str, str, str]:
    """
    URL에서 텍스트 최대 확보 → (text, used_url, method).
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
        out = [u]
        if "d4mucfpksywv.cloudfront.net" in u:
            out.append(u.replace("d4mucfpksywv.cloudfront.net", "cdn.openai.com"))
        if "language_moodels" in u:
            out.append(u.replace("language_moodels", "language_models"))
        if "language-moodels" in u:
            out.append(u.replace("language-moodels", "language-models"))
        if "openai.com/blog/" in u:
            out.append(u.replace("/blog/", "/index/"))
        uniq = []
        seen = set()
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
            pass

    try:
        scheme = "http" if url.startswith("http://") else "https"
        jurl = f"https://r.jina.ai/{scheme}://{url.split('://', 1)[1]}"
        r = requests.get(jurl, headers=UA, timeout=25)
        if r.status_code == 200 and r.text.strip():
            return r.text[:800_000], jurl, "jina-reader"
    except Exception:
        pass

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

# ──────────────────────────────────────────────────────────────
# HF JSON → 리포트 링크 수확 (유지)
def harvest_reports_from_github_json(gh: dict, hf_id: str, output_dir: str | Path = "."):
    import re
    from pathlib import Path

    def _extract_urls(text: str) -> list[str]:
        urls = re.findall(r'https?://[^\s)>"\']+', text or "")
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

    for u in _extract_urls(gh.get("readme", "")):
        if not _looks_report(u):
            continue
        try:
            text, used_url, how = _robust_fetch_report(u)
            full_texts.append({"arxiv_id": used_url, "full_text": text, "fetch_method": how})
        except Exception:
            _append_link_only(u)

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

    if full_texts:
        base = _norm_base(hf_id)
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
        _ensure_lowercase_alias("reports_fulltext", hf_id, output_dir)

def harvest_reports_from_hf_json(hf: dict, hf_id: str, output_dir: str | Path = "."):
    import re
    from pathlib import Path

    def _uniq(seq):
        return list(dict.fromkeys([s for s in seq if isinstance(s, str) and s.strip()]))

    def _extract_urls_from_text(txt: str) -> list[str]:
        if not txt: return []
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
                "/index/", "/posts/",
            ])
        )

    if not isinstance(hf, dict):
        return

    content = (hf.get("readme") or "")
    cd = hf.get("cardData") or {}
    if isinstance(cd, dict):
        content += "\n" + (cd.get("content") or "")

    urls_text = _extract_urls_from_text(content)

    links_obj = (cd.get("links") or {}) if isinstance(cd, dict) else {}
    urls_links = []
    for fld in ("paper","papers","homepage","repository","documentation","blog","arxiv"):
        val = links_obj.get(fld)
        if isinstance(val, str):
            urls_links.append(val)
        elif isinstance(val, list):
            urls_links.extend([str(x) for x in val])

    candidates = _uniq([u for u in (urls_text + urls_links) if _looks_report_like(u)])
    if not candidates:
        return

    full_texts = []
    for u in candidates:
        try:
            text, used_url, how = _robust_fetch_report(u)
        except NameError:
            text, used_url, how = "", u, "missing-robust-fetcher"
        full_texts.append({"arxiv_id": used_url, "full_text": text, "fetch_method": how})

    base = _norm_base(hf_id)
    out = Path(output_dir) / f"reports_fulltext_{base}.json"
    merged = []
    if out.exists():
        try:
            old = json.load(open(out, encoding="utf-8"))
            merged = (old.get("full_texts") or [])
        except Exception:
            merged = []
    json.dump({"model_id": hf_id, "full_texts": merged + full_texts},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"📄 Reports merged to: {out} (HF sources)")
    _ensure_lowercase_alias("reports_fulltext", hf_id, output_dir)

# ──────────────────────────────────────────────────────────────
# GitHub 후보 생성 (HF 카드/README에서)
def _hf_card_and_readme(mid: str, max_len: int = 30000) -> dict:
    out = {"card_content":"", "readme_md":""}
    try:
        r = requests.get(f"https://huggingface.co/api/models/{mid}?full=true", timeout=15)
        if r.status_code in (401,403):
            _warn("🔑 HF card 401/403 — set HF_TOKEN if you have access.")
        elif r.status_code == 429:
            _warn("⏳ HF rate-limited (429) — try later or set HF_TOKEN.")
        card = (r.json().get("cardData") or {})
        out["card_content"] = (card.get("content") or "")[:max_len]
        for br in ["main","master"]:
            rr = requests.get(f"https://huggingface.co/{mid}/raw/{br}/README.md", timeout=10)
            if rr.status_code == 200:
                out["readme_md"] = rr.text[:max_len]
                break
    except Exception:
        pass
    return out

def _extract_github_repos_from_text(text: str) -> list[str]:
    if not text: return []
    repos = re.findall(r"https?://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", text)
    # 중복 제거, .git 제거
    out = []
    seen = set()
    for rep in repos:
        rep = rep.split("?")[0].split("#")[0].replace(".git","")
        low = rep.lower()
        if low not in seen:
            seen.add(low); out.append(rep)
    return out

def _collect_hf_linked_repos(hf_id: str) -> tuple[list[str], set[str]]:
    """
    HF 카드의 링크 필드(repository/documentation/homepage/paper 등) + 카드/README 본문 내 GH 링크 전수 수집.
    returns: (repo_list, allowed_orgs)
    """
    try:
        r = requests.get(f"https://huggingface.co/api/models/{hf_id}?full=true", timeout=15)
        card = (r.json().get("cardData") or {})
    except Exception:
        card = {}

    linked = []
    links_obj = (card.get("links") or {}) if isinstance(card, dict) else {}
    for fld in ("repository","documentation","homepage","paper","papers","blog","arxiv"):
        val = links_obj.get(fld)
        if isinstance(val, str):
            m = re.findall(r"https?://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", val)
            for rep in m:
                linked.append(rep.split("?")[0].split("#")[0].replace(".git",""))
        elif isinstance(val, list):
            for v in val:
                m = re.findall(r"https?://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", str(v))
                for rep in m:
                    linked.append(rep.split("?")[0].split("#")[0].replace(".git",""))

    ctx = _hf_card_and_readme(hf_id, max_len=50000)
    linked += _extract_github_repos_from_text(ctx.get("card_content",""))
    linked += _extract_github_repos_from_text(ctx.get("readme_md",""))

    # 허용 org: HF org + 링크에서 발견된 org
    allowed_orgs = set()
    hf_org = hf_id.split("/",1)[0].lower() if "/" in hf_id else ""
    if hf_org: allowed_orgs.add(hf_org)
    for rep in linked:
        og = rep.split("/",1)[0].lower()
        allowed_orgs.add(og)

    # 유니크
    uniq = []
    seen = set()
    for rep in linked:
        low = rep.lower()
        if low not in seen:
            seen.add(low); uniq.append(rep)
    return uniq, allowed_orgs

# ──────────────────────────────────────────────────────────────
# Web search fallback (기존 함수 유지)
def web_search_github_candidates(hf_id: str) -> list[str]:
    import os as _os, requests as _req, re as _re

    name = hf_id.split("/",1)[1]
    queries = [
        f"{name} github repository",
        f"{name} model github",
        f"{name} official github",
    ]

    tavily = _os.getenv("TAVILY_API_KEY")
    if tavily:
        urls = []
        for q in queries:
            try:
                r = _req.post("https://api.tavily.com/search",
                              json={"api_key":tavily,"query":q,"max_results":5},
                              timeout=15)
                if r.status_code != 200:
                    _warn(f"🔑 Tavily error (HTTP {r.status_code}) — fallback to GitHub Search.")
                    continue
                try:
                    j = r.json()
                except Exception:
                    _warn("🔑 Tavily returned non-JSON — fallback.")
                    continue
                if not j.get("results"):
                    continue
                for it in j.get("results", []):
                    u = it.get("url","")
                    if "github.com" in u:
                        m = _re.search(r"github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", u)
                        if m: urls.append(m.group(1))
            except Exception as e:
                _warn(f"🔑 Tavily request failed — {e}. Using GitHub Search fallback.")
        if urls:
            # unique
            out = []
            seen=set()
            for rep in urls:
                low = rep.lower()
                if low not in seen:
                    seen.add(low); out.append(rep)
            return out[:8]
    else:
        _warn("ℹ️ TAVILY_API_KEY not set — using GitHub Search API fallback.")

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
                         params={"q":q, "sort":"stars", "order":"desc", "per_page":8},
                         headers=headers, timeout=15)
            if r.status_code == 403 and not gh_tok:
                _warn("🔑 GitHub Search rate limited (403) — set GITHUB_TOKEN.")
                continue
            j = r.json()
            for item in j.get("items", [])[:8]:
                full = item.get("full_name")
                if full: repos.append(full)
        except Exception as e:
            _warn(f"⚠️ GitHub Search failed: {e}")
    # unique
    out=[]; seen=set()
    for rep in repos:
        low=rep.lower()
        if low not in seen:
            seen.add(low); out.append(rep)
    return out[:8]

# ──────────────────────────────────────────────────────────────
# Repo 신호 스코어링
def _score_repo_signals(repo: str, hf_id: str, strong_tokens: list[str],
                        allowed_orgs: set[str], from_hf_link: bool) -> tuple[int, dict]:
    """
    다중 신호 기반 점수화:
      + org affinity / 유사도
      + 이름/README/경로의 강토큰 일치
      + HF 링크 출처 가중
      - 제3자 포트/툴/클라/바인딩/양자화 냄새
      - 버전 충돌 (다른 major 숫자)
    """
    rl = repo.lower()
    org, name = rl.split("/",1)
    hf_org = hf_id.split("/",1)[0].lower() if "/" in hf_id else ""

    major_info = _extract_version_tokens(hf_id)
    majors = major_info["majors"]

    snap = _fetch_repo_snapshot(repo)
    readme = snap["readme"]; paths = snap["paths"]

    score = 0
    detail = {"org_affinity":0, "name_hits":0, "readme_hits":0, "path_hits":0,
              "bad_keywords":0, "from_hf_link": int(bool(from_hf_link)),
              "version_conflict":0}

    # org affinity
    if org in allowed_orgs:
        score += 9; detail["org_affinity"] = 9
    elif _org_match_or_similar_name(org, hf_org):
        score += 6; detail["org_affinity"] = 6
    else:
        score -= 6; detail["org_affinity"] = -6

    # 강토큰 매칭
    name_hits = _count_token_hits(name, strong_tokens)
    readme_hits = _count_token_hits(readme, strong_tokens)
    path_hits = _count_token_hits(" ".join(paths[:300]), strong_tokens)

    detail["name_hits"] = name_hits
    detail["readme_hits"] = readme_hits
    detail["path_hits"] = path_hits

    score += 6 * min(1, name_hits)           # 이름에 한 번이라도 뜨면 +6
    score += 3 * min(2, readme_hits)         # README 매칭 최대 +6
    score += 2 * min(2, path_hits)           # 경로 매칭 최대 +4

    # HF 링크 출처 보너스
    if from_hf_link:
        score += 5

    # 포트/도구 냄새 감점
    bad = sum(1 for k in BAD_REPO_KEYWORDS if k in name)
    bad += sum(1 for k in BAD_REPO_KEYWORDS if k in readme.lower())
    score -= 2 * min(4, bad)   # 최대 -8
    detail["bad_keywords"] = bad

    # 버전 충돌
    if _has_conflicting_version(name, majors):
        score -= 8
        detail["version_conflict"] = -8

    return score, detail

def _is_strongly_valid(repo: str, strong_tokens: list[str], allowed_orgs: set[str],
                       score: int, detail: dict, strict_for_third_party: bool) -> bool:
    """수락 임계치. 3rd-party는 더 엄격."""
    org = repo.split("/",1)[0].lower()
    in_family = (org in allowed_orgs)

    # 강토큰 최소 요구: 이름 또는 README/경로 중 1개 이상
    has_any_strong = (detail["name_hits"] > 0 or detail["readme_hits"] > 0 or detail["path_hits"] > 0)

    # 3rd-party면 멀티 시그널 필요: 이름 + (README 또는 경로)
    if strict_for_third_party and not in_family:
        if not (detail["name_hits"] > 0 and (detail["readme_hits"] > 0 or detail["path_hits"] > 0)):
            return False
        return score >= 8  # 약간 높은 임계치

    # 패밀리/동맹 org면 토큰만 충족해도 허용 (버전충돌 없을 것)
    return has_any_strong and score >= 4

# ──────────────────────────────────────────────────────────────
# 메인 리졸버 (모든 경로에서 재사용)
def resolve_github_repo_for_hf_model(hf_id: str) -> str | None:
    """
    HF 모델 ID로부터 단계적 후보 수집(P0~P3) → 다중신호 점수화 → 임계치 수락.
    - Falcon3→Falcon-H1 같은 *버전 오선택* 방지(강토큰/버전 가드)
    - bloomz→bloomz.cpp 같은 *써드파티 포트* 방지(멀티시그널+점수 임계)
    - bigscience/bloomz → bigscience-workshop/xmtf 같은 *패밀리 레포* 허용
    """
    hf_org = hf_id.split("/",1)[0].lower() if "/" in hf_id else ""
    name_only = (hf_id.split("/",1)[1] if "/" in hf_id else hf_id)
    vinfo = _extract_version_tokens(hf_id)
    strong = vinfo["strong"]

    # P0/P1: HF 카드·README에서 후보 + allowed_orgs
    p01_list, allowed_orgs = _collect_hf_linked_repos(hf_id)

    # 후보를 (레포, 출처가 HF인지 여부)로 표준화
    candidates = []
    hf_card_content = _hf_card_and_readme(hf_id, 30000)
    linked_from_hf = set([r.lower() for r in p01_list])

    for rep in p01_list:
        candidates.append((rep, True))

    # P2: HF 카드/README 본문 내 명시 링크는 이미 포함됨. (p01_list가 포함)
    # P3: 웹/GH 검색
    p3 = web_search_github_candidates(hf_id)
    for rep in p3:
        if rep.lower() not in linked_from_hf:
            candidates.append((rep, False))

    if not candidates:
        return None

    # 평가/스코어
    best = None
    best_score = -10**9
    best_detail = {}
    for rep, from_hf in candidates:
        if not test_github_repo_exists(rep):
            continue
        score, detail = _score_repo_signals(rep, hf_id, strong_tokens=list(strong),
                                            allowed_orgs=allowed_orgs,
                                            from_hf_link=from_hf)
        # 수락 기준 (3rd-party는 더 엄격)
        strict_third = True
        if not _is_strongly_valid(rep, list(strong), allowed_orgs, score, detail, strict_third):
            print(f"🔎 Candidate rejected: {rep} (score={score}, detail={detail})")
            continue

        # 베스트 갱신
        if score > best_score:
            best, best_score, best_detail = rep, score, detail

    if best:
        print(f"✅ Resolved GH repo: {best} (score={best_score}, detail={best_detail})")
    else:
        print("⚠️ No GitHub repo passed the thresholds.")
    return best

# ──────────────────────────────────────────────────────────────
# GPT checks (optional rescue). temperature 제거
def _gpt_repo_model_signal(repo: str, hf_id: str) -> tuple[bool, float, str]:
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

=== README (trunc) ===
{snap["readme"]}

=== PATHS (sample) ===
{json.dumps(snap["paths"][:120], ensure_ascii=False)}
"""
    try:
        rsp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL_SIGNAL", "gpt-4o-mini"),
            response_format={"type": "json_object"},
            messages=[{"role":"system", "content": sys},
                      {"role":"user", "content": prompt}]
        )
        obj = json.loads(rsp.choices[0].message.content)
        ok = bool(obj.get("ok"))
        conf = float(obj.get("confidence", 0))
        reason = obj.get("reason","")
        return (ok and conf >= 0.6, conf, reason)
    except Exception as e:
        _warn(f"🔑 OpenAI error during repo-signal check: {e}")
        return (False, 0.0, "OpenAI error")

# ──────────────────────────────────────────────────────────────
# Base model detector (temperature 제거)
def gpt_detect_base_model(hf_id: str) -> str | None:
    import requests as _req

    def _hf_card_readme(mid: str, max_len: int = 12000) -> str:
        try:
            r = _req.get(f"https://huggingface.co/api/models/{mid}?full=true", timeout=15)
            if r.status_code in (401,403):
                _warn("🔑 HF card 401/403 — set HF_TOKEN.")
            elif r.status_code == 429:
                _warn("⏳ HF card rate-limited (429).")
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

    def _name_only(mid: str) -> str:
        nm = (mid.split("/",1)[1] if "/" in mid else mid).lower()
        return re.sub(r"[^a-z0-9]+","", nm)

    ctx = _hf_card_readme(hf_id)
    if not ctx or not _has_openai_key():
        return None

    prompt_sys = f"""
You analyze a Hugging Face model card/README and identify its PRETRAINED (base) model.
If the input is already a base model, return null.

Output JSON only:
{{ "pretrain_model": "org/name" | null }}
""".strip()

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role":"system","content":prompt_sys},
                      {"role":"user","content":ctx}]
        )
        pred = json.loads(resp.choices[0].message.content)
        pre_id = (pred.get("pretrain_model") or "").strip()
        if not pre_id:
            return None
        if _name_only(pre_id) == _name_only(hf_id):
            return None
        if test_hf_model_exists(pre_id):
            return pre_id
    except Exception as e:
        _warn(f"🔑 OpenAI error during base-model detection — {e}")
    return None

# ──────────────────────────────────────────────────────────────
# 입력 파싱/검증
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

# ──────────────────────────────────────────────────────────────
# 메인 파이프라인
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

    # --- GitHub 리졸버 (통합) ---
    if hf_ok and not gh_id:
        gh_id = resolve_github_repo_for_hf_model(hf_cand)

    # ───────────────── HF processing ─────────────────
    data = {}
    if hf_id:
        rank_hf = found_rank_hf or 'none'
        print(f"✅ HF model: {hf_id} (found at priority: {rank_hf})")
        data = huggingface_fetcher(hf_id, save_to_file=True, output_dir=outdir)
        _ensure_lowercase_alias("huggingface", hf_id, outdir)

        try:
            harvest_reports_from_hf_json(data, hf_id, output_dir=outdir)
        except Exception as e:
            print("⚠️ HF report harvesting failed:", e)

        try:
            _ensure_lowercase_alias("huggingface", hf_id, outdir)
            hf_filtered = filter_hf_features(hf_id, output_dir=outdir)
        except FileNotFoundError:
            hf_filtered = {}
            print("⚠️ Hugging Face JSON file not found; skipping filtering")

    # ───────────────── GitHub processing ─────────────────
    gh_data = {}
    if gh_id:
        print(f"✅ GH repo: {gh_id}")
        try:
            gh_data = github_fetcher(gh_id, branch="main", save_to_file=True, output_dir=outdir) or {}
            _ensure_lowercase_alias("github", gh_id, outdir)
        except requests.exceptions.HTTPError:
            print("⚠️ Failed to access 'main' branch; retrying with 'master'...")
            try:
                gh_data = github_fetcher(gh_id, branch="master", save_to_file=True, output_dir=outdir) or {}
                _ensure_lowercase_alias("github", gh_id, outdir)
            except Exception as e:
                print("❌ 'master' branch also failed:", e)

        try:
            gh_filtered = filter_github_features(gh_id, output_dir=outdir)
        except FileNotFoundError:
            gh_filtered = {}
            print("⚠️ GitHub JSON file not found; skipping filtering")
    else:
        print("⚠️ No GitHub info")

    # ───────────────── Paper/Report aggregation ─────────────────
    if hf_id:
        try:
            ax_ok = arxiv_fetcher_from_model(hf_id, save_to_file=True, output_dir=outdir)
            if ax_ok:
                _ensure_lowercase_alias("arxiv_fulltext", hf_id, outdir)
        except Exception as e:
            print("⚠️ arXiv fetch failed:", e)

    try:
        if gh_data and hf_id:
            harvest_reports_from_github_json(gh_data, hf_id, output_dir=outdir)
    except Exception as e:
        print("⚠️ GH report harvesting failed:", e)

    try:
        _ensure_lowercase_alias("arxiv_fulltext", hf_id, outdir)
        ax_filtered = filter_arxiv_features(hf_id, output_dir=outdir)
    except FileNotFoundError:
        ax_filtered = {}
        print("⚠️ No arXiv/report inputs found for dispatcher; skipping")

    try:
        rpt_filtered = filter_reports_features(hf_id, output_dir=outdir)
    except FileNotFoundError:
        rpt_filtered = {}
        print("⚠️ No report inputs found for reports dispatcher; skipping")

    # ─── 베이스 모델 탐지 + 동일 리졸버 재사용 ───────────────
    base_model_id = gpt_detect_base_model(hf_id) if hf_id else None
    if base_model_id:
        print(f"🧱 Pretrained (base) model found by GPT: {base_model_id}")

        huggingface_fetcher(base_model_id, save_to_file=True, output_dir=outdir)
        _ensure_lowercase_alias("huggingface", base_model_id, outdir)

        try:
            from pretrain_hf_Dispatcher import filter_pretrain_hf
            filter_pretrain_hf(base_model_id, output_dir=outdir)
        except Exception as e:
            print("⚠️ pretrain_hf dispatcher failed:", e)

        base_gh = resolve_github_repo_for_hf_model(base_model_id)
        if base_gh:
            try:
                github_fetcher(base_gh, save_to_file=True, output_dir=outdir)
                _ensure_lowercase_alias("github", base_gh, outdir)
                from pretrain_github_Dispatcher import filter_pretrain_gh
                filter_pretrain_gh(base_gh, output_dir=outdir)
            except Exception as e:
                print("⚠️ GH fetch/dispatch failed:", e)
        else:
            print("⚠️ Could not find the base model's GitHub repo; skipping GH fetcher")

        try:
            ax_ok = arxiv_fetcher_from_model(base_model_id, save_to_file=True, output_dir=outdir)
            if ax_ok:
                _ensure_lowercase_alias("arxiv_fulltext", base_model_id, outdir)
                from pretrain_arxiv_Dispatcher import filter_pretrain_arxiv
                filter_pretrain_arxiv(base_model_id, output_dir=outdir)
            else:
                print("⚠️ Could not find a paper link; skipping arXiv fetcher")
        except Exception as e:
            print("⚠️ arXiv fetch/dispatch failed:", e)

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
        base = _norm_base(full)
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
            hf_json = _open_json_anycase_by_key("huggingface", hf_id, outdir)
            cd = hf_json.get("cardData") or {}
            readme_text = (cd.get("content") or "")
    except Exception as e:
        print(f"⚠️ README extraction failed: {e}")

    if readme_text and readme_text.strip():
        try:
            os.environ["MODEL_OUTPUT_DIR"] = str(outdir)
            # run_inference(readme_text, output_dir=outdir, keep_code=True)
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

# ── Tee logger (console + file) ─────────────────────────────────────────
class _TeeIO:
    def __init__(self, *streams): self._streams = streams
    def write(self, data):
        for s in self._streams:
            try: s.write(data)
            except Exception: pass
    def flush(self):
        for s in self._streams:
            try: s.flush()
            except Exception: pass

@contextlib.contextmanager
def tee_logs(log_path: Path, mode: str = "a"):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, mode, encoding="utf-8") as f:
        orig_out, orig_err = sys.stdout, sys.stderr
        sys.stdout = _TeeIO(orig_out, f)
        sys.stderr = _TeeIO(orig_err, f)
        try:
            yield
        finally:
            try:
                sys.stdout.flush(); sys.stderr.flush()
            finally:
                sys.stdout, sys.stderr = orig_out, orig_err

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
        try:
            model_dir = make_model_dir(user_input)
        except Exception:
            model_dir = Path(".")
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        try:
            info = extract_model_info(user_input)
            base_name = info["hf_id"].replace("/", "_").lower()
        except Exception:
            base_name = re.sub(r"[^a-z0-9_]+", "_", user_input.lower())
        log_file = model_dir / f"run_{ts}_{base_name}.log"

        with tee_logs(log_file):
            print(f"\n======== {idx}/{len(models)} ▶ {user_input} ========")
            try:
                print(f"📁 Directory to create/use: {model_dir}")
                run_all_fetchers(user_input)

                info  = extract_model_info(user_input)
                hf_id = info["hf_id"]
                if test_hf_model_exists(hf_id):
                    with open(model_dir / "identified_model.txt", "w", encoding="utf-8") as f:
                        f.write(hf_id)
                    print(f"✅ Saved model ID: {model_dir / 'identified_model.txt'}")

            except Exception as e:
                import traceback
                print("❌ Error encountered while processing:", e)
                traceback.print_exc()
                continue

            print(f"🧾 Log saved to: {log_file}")

    print("\n🎉 All tasks completed.")
