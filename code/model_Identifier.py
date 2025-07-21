import json
import re
import requests
import os
from urllib.parse import urlparse
from dotenv import load_dotenv
from openai import OpenAI
from Huggingface_Fatcher import huggingface_fetcher
from github_Fatcher import github_fetcher
from arxiv_Fetcher import arxiv_fetcher_from_model
# from openness_Evaluator import evaluate_openness  # 평가 모듈 임포트
from github_Dispatcher import filter_github_features
from arxiv_Dispatcher import filter_arxiv_features
from huggingface_Dispatcher import filter_hf_features
from openness_Evaluator import evaluate_openness_from_files
from inference import run_inference

# 환경 변수 로드 및 OpenAI 클라이언트 초기화
dotenv_path = os.path.join(os.getcwd(), '.env')
load_dotenv(dotenv_path)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. 입력 파싱: URL 또는 org/model
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
        raise ValueError("올바른 입력 형식이 아닙니다. 'org/model' 또는 URL을 입력하세요.")
    full_id = f"{organization}/{model}"
    hf_id = full_id.lower()
    return {"platform": platform, "organization": organization,
            "model": model, "full_id": full_id, "hf_id": hf_id}

# 2. 존재 여부 테스트
def test_hf_model_exists(model_id: str) -> bool:
    resp = requests.get(f"https://huggingface.co/api/models/{model_id}")
    return resp.status_code == 200

def test_github_repo_exists(repo: str) -> bool:
    resp = requests.get(f"https://api.github.com/repos/{repo}")
    return resp.status_code == 200

# 3. 링크 파싱 헬퍼
def extract_repo_from_url(url: str) -> str:
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) >= 2:
        repo = parts[1].split("?")[0].split("#")[0].replace(".git", "")
        return f"{parts[0]}/{repo}"
    return ""

# 4. HF 페이지 → GitHub (fallback: raw README)
def find_github_in_huggingface(model_id: str) -> str:
    try:
        card = requests.get(f"https://huggingface.co/api/models/{model_id}?full=true").json().get("cardData", {})
        for field in ["repository", "homepage"]:
            url = card.get("links", {}).get(field, "")
            if "github.com" in url:
                return extract_repo_from_url(url)
        content = card.get("content", "")
        all_links = re.findall(r"https://github\.com/[\w\-]+/[\w\-\.]+", content)
        for link in all_links:
            if "github.com" in link:
                return extract_repo_from_url(link)
    except:
        pass
    for branch in ["main", "master"]:
        raw_url = f"https://huggingface.co/{model_id}/raw/{branch}/README.md"
        try:
            r = requests.get(raw_url)
            if r.status_code == 200:
                m2 = re.search(r"github\.com/([\w\-]+/[\w\-\.]+)", r.text, re.IGNORECASE)
                if m2:
                    return m2.group(1)
        except:
            pass
    return None

# 5. GitHub 페이지 → HF (fallback: raw README and HTML)
def find_huggingface_in_github(repo: str) -> str:
    for fname in ["README.md", "README.en.md", "model_card.md"]:
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

# 7. 메인 로직
def run_all_fetchers(user_input: str):
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
    if gh_ok and not hf_id:
        guess_hf = gpt_guess_huggingface_from_github(full)
        print(f"⏳ 3순위 GPT GH→HF guess: {guess_hf}")
        if guess_hf and test_hf_model_exists(guess_hf):
            hf_id = guess_hf
            found_rank_hf = 3

    if hf_id:
        rank_hf = found_rank_hf or '없음'
        print(f"✅ HF model: {hf_id} (발견: {rank_hf}순위)")
        huggingface_fetcher(hf_id, save_to_file=True)
        arxiv_fetcher_from_model(hf_id, save_to_file=True)
        try:
            hf_filtered = filter_hf_features(hf_id)
        except FileNotFoundError:
            gh_filtered = {}
            print("⚠️ Huggingfacw JSON 파일이 존재하지 않아 필터링 생략")
        try:
            ax_filtered = filter_arxiv_features(hf_id)
        except FileNotFoundError:
            ax_filtered = {}
            print("⚠️ arxiv JSON 파일이 존재하지 않아 필터링 생략")
    else:
        print("⚠️ HF 정보 없음")

    if gh_id:
        rank_gh = found_rank_gh or '없음'
        print(f"✅ GH repo: {gh_id} (발견: {rank_gh}순위)")
        github_fetcher(gh_id, branch="main", save_to_file=True)
        try:
            gh_filtered = filter_github_features(gh_id)
        except FileNotFoundError:
            gh_filtered = {}
            print("⚠️ GitHub JSON 파일이 존재하지 않아 필터링 생략")
    else:
        print("⚠️ GH 정보 없음")

      # 9. Openness 평가 수행
    
    try:
        print("📝 개방성 평가 시작...")
        eval_res = evaluate_openness_from_files(full)
        print(f"✅ 개방성 평가 완료. 결과 파일: openness_score_{full.replace('/', '_')}.json")
    except Exception as e:
        print("⚠️ 개방성 평가 중 오류 발생:", e)



if __name__ == "__main__":
    user_input = input("🌐 HF/GH URL 또는 org/model: ").strip()
    run_all_fetchers(user_input)

    info = extract_model_info(user_input)
    hf_id = info['hf_id']

    if test_hf_model_exists(hf_id):
        with open("identified_model.txt", "w", encoding="utf-8") as f:
            f.write(hf_id)
        print(f"✅ 모델 ID 저장 완료: {hf_id}")

        # ✅ 바로 추론까지 실행
        prompt = input("📝 프롬프트를 입력하세요: ")
        run_inference(hf_id, prompt)