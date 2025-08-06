# model_Identifier.py

import os
import re
import json
import requests
from pathlib import Path
from contextlib import contextmanager
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI

from huggingface_Fetcher import huggingface_fetcher
from github_Fetcher import github_fetcher
from arxiv_Fetcher import arxiv_fetcher_from_model

from github_Dispatcher import filter_github_features
from arxiv_Dispatcher import filter_arxiv_features
from huggingface_Dispatcher import filter_hf_features

from openness_Evaluator import evaluate_openness_from_files
from inference import run_inference


# ========= 환경설정 =========
dotenv_path = os.path.join(os.getcwd(), ".env")
load_dotenv(dotenv_path)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ========= 유틸 =========
@contextmanager
def _pushd(path: Path):
    """작업 디렉토리를 일시적으로 변경."""
    old = Path.cwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def extract_model_info(input_str: str) -> dict:
    """HF/GH URL 또는 'org/model'을 받아 파싱."""
    platform = None
    organization = model = None

    if input_str.startswith("http"):
        parsed = urlparse(input_str)
        domain = parsed.netloc.lower()
        segments = parsed.path.strip("/").split("/")
        if len(segments) >= 2:
            organization = segments[0]
            model = (
                segments[1].split("?")[0]
                .split("#")[0]
                .replace(".git", "")
            )
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
        raise ValueError("올바른 입력 형식이 아닙니다. 'org/model' 또는 URL을 입력하세요.")

    full_id = f"{organization}/{model}"
    hf_id = full_id.lower()
    return {
        "platform": platform,
        "organization": organization,
        "model": model,
        "full_id": full_id,
        "hf_id": hf_id,
    }


def test_hf_model_exists(model_id: str) -> bool:
    resp = requests.get(f"https://huggingface.co/api/models/{model_id}")
    return resp.status_code == 200


def test_github_repo_exists(repo: str) -> bool:
    resp = requests.get(f"https://api.github.com/repos/{repo}")
    return resp.status_code == 200


def extract_repo_from_url(url: str) -> str:
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) >= 2:
        repo = parts[1].split("?")[0].split("#")[0].replace(".git", "")
        return f"{parts[0]}/{repo}"
    return ""


def find_github_in_huggingface(model_id: str) -> str | None:
    """HF 모델 카드/README에서 GH repo를 찾아냄."""
    try:
        card = requests.get(
            f"https://huggingface.co/api/models/{model_id}?full=true"
        ).json().get("cardData", {})
        # 1) 링크 필드 우선
        for field in ["repository", "homepage"]:
            url = card.get("links", {}).get(field, "")
            if "github.com" in url:
                return extract_repo_from_url(url)
        # 2) 모델 카드 본문에서 찾기
        content = card.get("content", "")
        all_links = re.findall(r"https://github\.com/[\w\-]+/[\w\-\.]+", content)
        for link in all_links:
            if "github.com" in link:
                return extract_repo_from_url(link)
    except Exception:
        pass

    # 3) raw README에서 찾기
    for branch in ["main", "master"]:
        raw_url = f"https://huggingface.co/{model_id}/raw/{branch}/README.md"
        try:
            r = requests.get(raw_url)
            if r.status_code == 200:
                m2 = re.search(
                    r"github\.com/([\w\-]+/[\w\-\.]+)", r.text, re.IGNORECASE
                )
                if m2:
                    return m2.group(1)
        except Exception:
            pass
    return None


def find_huggingface_in_github(repo: str) -> str | None:
    """GH README/페이지에서 HF 모델 ID를 찾아냄."""
    for fname in ["README.md"]:
        for branch in ["main", "master"]:
            raw_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{fname}"
            try:
                r = requests.get(raw_url)
                if r.status_code == 200:
                    m = re.search(
                        r"https?://huggingface\.co/([\w\-]+/[\w\-\.]+)",
                        r.text,
                        re.IGNORECASE,
                    )
                    if m:
                        candidate = m.group(1).lower()
                        if not candidate.startswith("collections/"):
                            return candidate

                    m2 = re.search(
                        r"huggingface\.co/([\w\-]+/[\w\-\.]+)",
                        r.text,
                        re.IGNORECASE,
                    )
                    if m2:
                        candidate = m2.group(1).lower()
                        if not candidate.startswith("collections/"):
                            return candidate

                    m_md = re.search(
                        r"\[.*?\]\((https?://huggingface\.co/[\w\-/\.]+)\)",
                        r.text,
                        re.IGNORECASE,
                    )
                    if m_md:
                        candidate = extract_model_info(m_md.group(1))["hf_id"]
                        if not candidate.startswith("collections/"):
                            return candidate

                    m_html = re.search(
                        r'<a\s+href="https?://huggingface\.co/([\w\-/\.]+)"',
                        r.text,
                        re.IGNORECASE,
                    )
                    if m_html:
                        candidate = m_html.group(1).lower()
                        if not candidate.startswith("collections/"):
                            return candidate
            except Exception:
                pass

    try:
        html = requests.get(f"https://github.com/{repo}").text
        m3 = re.findall(
            r"https://huggingface\.co/[\w\-]+/[\w\-\.]+", html, re.IGNORECASE
        )
        for link in m3:
            if "href" in html[html.find(link) - 20 : html.find(link)]:
                candidate = extract_model_info(link)["hf_id"]
                if not candidate.startswith("collections/"):
                    return candidate
    except Exception:
        pass
    return None


def gpt_guess_github_from_huggingface(hf_id: str) -> str | None:
    prompt = f"""
Hugging Face에 등록된 모델 '{hf_id}'에 대해, 이 모델의 원본 코드가 저장된 GitHub 저장소를 추정하세요.

🟢 규칙:
1) 'organization/repo' 형식으로 **정확한 경로만** 반환(링크/설명 X)
2) 모노리포(google-research/google-research)보단 모델 전용 저장소 우선
3) distill 모델이면 부모 모델 저장소를 고려
4) 이름/논문/토크나이저/프레임워크 단서를 활용
5) 결과는 한 줄만. 예: facebookresearch/llama
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        guess = resp.choices[0].message.content.strip()
        if "/" in guess:
            return guess
    except Exception as e:
        print("⚠️ GPT HF→GH 추정 중 오류 발생:", e)
    return None


def gpt_guess_huggingface_from_github(gh_id: str) -> str | None:
    prompt = f"""
GitHub 저장소 '{gh_id}'와 연결된 Hugging Face 모델 ID를 추정하세요.
- 정확한 organization/model만 출력(설명 X)
- 모노리포보단 모델 전용 리포 우선
- 예시 출력: facebookresearch/llama
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        guess = resp.choices[0].message.content.strip().lower()
        if "/" in guess:
            return guess
    except Exception as e:
        print("⚠️ GPT GH→HF 추정 중 오류 발생:", e)
    return None


def make_model_dir(user_input: str) -> Path:
    """모델별 서브디렉토리 생성(안전한 폴더명)."""
    info = extract_model_info(user_input)
    base = info["hf_id"]  # 예: bigscience/bloomz-560m

    # 안전한 디렉토리명: 슬래시/공백/특수문자 -> '_', 소문자
    safe = re.sub(r"[^\w.-]+", "_", base).replace("/", "_").lower()
    path = Path(safe)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ========= 메인 파이프라인 =========
def run_all_fetchers(user_input: str):
    info = extract_model_info(user_input)
    full = info["full_id"]
    hf_cand = info["hf_id"]

    hf_id = None
    gh_id = None
    found_rank_hf = None
    found_rank_gh = None

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
        print(f"🔍 2순위 HF→GH link: {gh_link}")
        if gh_link and test_github_repo_exists(gh_link):
            gh_id = gh_link
            found_rank_gh = 2

    if gh_ok and not hf_id:
        hf_link = find_huggingface_in_github(full)
        print(f"🔍 2순위 GH→HF link: {hf_link}")
        if hf_link and test_hf_model_exists(hf_link):
            hf_id = hf_link
            found_rank_hf = 2

    if hf_ok and not gh_id:
        guess_gh = gpt_guess_github_from_huggingface(hf_cand)
        print(f"⏳ 3순위 GPT HF→GH guess: {guess_gh}")
        if guess_gh and test_github_repo_exists(guess_gh):
            gh_id = guess_gh
            found_rank_gh = 3
            print("⚠️ GPT 추정 결과입니다. 실제 포함 모델 확인 필요")

    if gh_ok and not hf_id:
        guess_hf = gpt_guess_huggingface_from_github(full)
        print(f"⏳ 3순위 GPT GH→HF guess: {guess_hf}")
        if guess_hf and test_hf_model_exists(guess_hf):
            hf_id = guess_hf
            found_rank_hf = 3
            print("⚠️ GPT 추정 결과입니다. 모델 ID 정확성 확인 필요")

    if not hf_id and not gh_id:
        print("❌ HF/GH 모두 식별 실패")
        return

    # === 모델별 서브디렉토리에서 모든 산출물 저장 ===
    outdir = make_model_dir(user_input)
    data = None  # HF 데이터(README용)

    with _pushd(outdir):
        # ---- Hugging Face 측 수집/필터 ----
        if hf_id:
            rank_hf = found_rank_hf or "없음"
            print(f"✅ HF model: {hf_id} (발견: {rank_hf}순위)")
            data = huggingface_fetcher(hf_id, save_to_file=True)
            arxiv_fetcher_from_model(hf_id, save_to_file=True)

            try:
                _ = filter_hf_features(hf_id)
            except FileNotFoundError:
                print("⚠️ HuggingFace JSON 파일이 없어 필터링 생략")

            try:
                _ = filter_arxiv_features(hf_id)
            except FileNotFoundError:
                print("⚠️ arXiv JSON 파일이 없어 필터링 생략")
        else:
            print("⚠️ HuggingFace 정보 없음")

        # ---- GitHub 측 수집/필터 ----
        if gh_id:
            rank_gh = found_rank_gh or "없음"
            print(f"✅ GH repo: {gh_id} (발견: {rank_gh}순위)")
            try:
                github_fetcher(gh_id, branch="main", save_to_file=True)
            except requests.exceptions.HTTPError:
                print("⚠️ main 브랜치 실패, master로 재시도...")
                try:
                    github_fetcher(gh_id, branch="master", save_to_file=True)
                except Exception as e:
                    print("❌ master 브랜치도 실패:", e)

            try:
                _ = filter_github_features(gh_id)
            except FileNotFoundError:
                print("⚠️ GitHub JSON 파일이 없어 필터링 생략")
        else:
            print("⚠️ GitHub 정보 없음")

        # ---- README 기반 간단 추론(선택) ----
        if isinstance(data, dict) and data.get("readme"):
            run_inference(data["readme"])

        # ---- 개방성 평가 ----
        try:
            print("📝 개방성 평가 시작...")
            try:
                # 새 시그니처(베이스 디렉토리 전달 지원)
                _ = evaluate_openness_from_files(full, base_dir=str(outdir))
            except TypeError:
                # 구 시그니처(인자 하나짜리) 호환
                _ = evaluate_openness_from_files(full)
            print(f"✅ 개방성 평가 완료. 결과 파일: openness_score_{full.replace('/', '_')}.json")
        except Exception as e:
            print("⚠️ 개방성 평가 중 오류 발생:", e)


# ========= 엔트리포인트 =========
if __name__ == "__main__":
    user_input = input("🌐 HF/GH URL 또는 org/model: ").strip()
    run_all_fetchers(user_input)
