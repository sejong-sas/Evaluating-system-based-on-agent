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

# 5. GitHub 페이지 → HF (fallback: raw README and HTML) ,,,, "README.en.md", "model_card.md"###################
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
Hugging Face에 등록된 모델 '{hf_id}'에 대해, 이 모델의 원본 코드가 저장된 GitHub 저장소를 추정하세요.

🟢 지켜야 할 규칙:
1. 'organization/repo' 형식으로 **정확한 GitHub 경로만** 반환하세요 (링크 X, 설명 X).
2. 'google-research/google-research'처럼 너무 일반적인 모노리포지터리는 피하고, 모델 단위 저장소가 있다면 그것을 우선 추정하세요.
3. distill 모델인 경우 부모 모델을 찾아주세요.
4. 해당 모델의 이름, 구조, 논문, tokenizer, 사용 라이브러리(PyTorch, JAX, T5 등)를 참고해서 정확한 repo를 추정하세요.
5. 결과는 **딱 한 줄**, 예: `facebookresearch/llama`

🔴 출력에는 부가 설명 없이 GitHub 저장소 경로만 포함해야 합니다.
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
        print("⚠️ GPT HF→GH 추정 중 오류 발생:", e)
    return None

def gpt_guess_huggingface_from_github(gh_id: str) -> str:
    prompt = f"""
Hugging Face에 등록된 모델 '{gh_id}'의 원본 코드가 저장된 Hugging Face 모델 ID를 추정해주세요.
- 정확한 organization/repository 경로만 출력해주세요.
- 모델 이름이나 관련 논문에서 유래된 GitHub 저장소를 기준으로 추정하세요.
- 예시 출력: facebookresearch/llama
- 'google-research/google-research'처럼 광범위한 모노리포지터리는 피하고, 해당 모델을 위한 별도 저장소가 있다면 그쪽을 우선 고려하세요.
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
        print("⚠️ GPT GH→HF 추정 중 오류 발생:", e)
    return None

def run_all_fetchers(user_input: str):
    outdir = make_model_dir(user_input)
    print(f"📁 출력 경로: {outdir}")
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
            print("⚠️ GPT 추정 결과입니다. 저장소가 실제로 해당 모델을 포함하는지 검토 필요")

    if gh_ok and not hf_id:
        guess_hf = gpt_guess_huggingface_from_github(full)
        print(f"⏳ 3순위 GPT GH→HF guess: {guess_hf}")
        if guess_hf and test_hf_model_exists(guess_hf):
            hf_id = guess_hf
            found_rank_hf = 3
            print("⚠️ GPT 추정 결과입니다. 모델 ID가 정확한지 검토 필요")
    
    
    if hf_id:
        rank_hf = found_rank_hf or '없음'
        print(f"✅ HF model: {hf_id} (발견: {rank_hf}순위)")
        data = huggingface_fetcher(hf_id, save_to_file=True, output_dir=outdir)
        arxiv_fetcher_from_model(hf_id, save_to_file=True, output_dir=outdir)
        try:
            hf_filtered = filter_hf_features(hf_id, output_dir=outdir)
        except FileNotFoundError:
            hf_filtered = {}
            print("⚠️ HuggingFace JSON 파일이 존재하지 않아 필터링 생략")
        try:
            ax_filtered = filter_arxiv_features(hf_id, output_dir=outdir)
        except FileNotFoundError:
            ax_filtered = {}
            print("⚠️ arXiv JSON 파일이 존재하지 않아 필터링 생략")
    else:
        print("⚠️ HuggingFace 정보 없음")
    if gh_id:
        rank_gh = found_rank_gh or '없음'
        print(f"✅ GH repo: {gh_id} (발견: {rank_gh}순위)")

        try:
            github_fetcher(gh_id, branch="main", save_to_file=True, output_dir=outdir)
        except requests.exceptions.HTTPError:
            print("⚠️ main 브랜치 접근 실패, master 브랜치로 재시도...")
            try:
                github_fetcher(gh_id, branch="master", save_to_file=True, output_dir=outdir)
            except Exception as e:
                print("❌ master 브랜치도 실패:", e)

        try:
            gh_filtered = filter_github_features(gh_id, output_dir=outdir)
        except FileNotFoundError:
            gh_filtered = {}
            print("⚠️ GitHub JSON 파일이 존재하지 않아 필터링 생략")
    else:
        print("⚠️ GitHub 정보 없음")

    run_inference(data.get("readme"))
    
# 8. Openness 평가 수행
    try:
        print("📝 개방성 평가 시작...")
        eval_res = evaluate_openness_from_files(full, base_dir=str(outdir))
        base = full.replace("/", "_")
        outfile = Path(outdir) / f"openness_score_{base}.json"
        print(f"✅ 개방성 평가 완료. 결과 파일: {outfile}")
    except Exception as e:
        print("⚠️ 개방성 평가 중 오류 발생:", e)

    # README 기반 추론은 data가 있을 때만
    if 'data' in locals() and isinstance(data, dict) and data.get("readme"):
        run_inference(data.get("readme"))


def make_model_dir(user_input: str) -> Path:
    info = extract_model_info(user_input)        # 위에 이미 있는 함수
    base = info["hf_id"]                         # ex) 'bigscience/bloomz-560m'
    # 폴더명으로 안전하게 변경: 슬래시/공백/금지문자 -> _
    safe = re.sub(r'[<>:"/\\|?*\s]+', "_", base) # ex) 'bigscience_bloomz-560m'
    path = Path(safe)
    path.mkdir(parents=True, exist_ok=True)      # 상위 폴더까지 생성, 있으면 그냥 통과
    return path

if __name__ == "__main__":
    user_input = input("🌐 HF/GH URL 또는 org/model: ").strip()
    model_dir = make_model_dir(user_input)
    print(f"📁 생성/사용할 폴더: {model_dir}") 
    run_all_fetchers(user_input)

    info = extract_model_info(user_input)
    hf_id = info['hf_id']

    if test_hf_model_exists(hf_id):
        with open("identified_model.txt", "w", encoding="utf-8") as f:
            f.write(hf_id)
        print(f"✅ 모델 ID 저장 완료: {hf_id}")

    #     # ✅ 바로 추론까지 실행
    #     prompt = input("📝 프롬프트를 입력하세요: ")
    #     run_inference(hf_id, prompt)