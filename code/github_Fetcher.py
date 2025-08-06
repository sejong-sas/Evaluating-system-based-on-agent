from typing import Dict, Any
import requests
import json
from pathlib import Path

def github_fetcher(repo_full_name: str,
                   branch: str = "main",
                   token: str = None,
                   save_to_file: bool = True,
                   output_dir: str | Path = ".") -> Dict[str, Any]:
    """
    Fetches from a GitHub repo:
      - files: 모든 파일 경로 리스트
      - license_files: 'LICENSE'로 시작하는 모든 파일과 그 내용
      - readme: README.md 내용 (없으면 빈 문자열)
      - py_files: 모든 .py 파일과 그 내용
      저장 옵션 포함 (save_to_file=True 시 JSON 파일로 저장)
    """
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    # 1) 저장소 트리 조회
    tree_url = f"https://api.github.com/repos/{repo_full_name}/git/trees/{branch}?recursive=1"
    resp = requests.get(tree_url, headers=headers)
    resp.raise_for_status()
    tree = resp.json().get("tree", [])

    # 2) 모든 파일 경로 추출
    paths = [item["path"] for item in tree if item["type"] == "blob"]

    # 콘텐츠 fetch helper
    def fetch_raw(path: str) -> str:
        raw_url = f"https://raw.githubusercontent.com/{repo_full_name}/{branch}/{path}"
        r = requests.get(raw_url, headers=headers)
        return r.text if r.status_code == 200 else ""

    # 3) LICENSE* 파일
    license_paths = [p for p in paths if p.upper().startswith("LICENSE")]
    license_files = {p: fetch_raw(p) for p in license_paths}

    # 4) README.md
    readme = fetch_raw("README.md") if "README.md" in paths else ""

    # 5) .py 파일
    py_files = {p: fetch_raw(p) for p in paths if p.endswith(".py")}

    result = {
        "repo": repo_full_name,
        "branch": branch,
        "files": paths,
        "license_files": license_files,
        "readme": readme,
        "py_files": py_files
    }

    # 6) 파일 저장
    if save_to_file:
        filename_safe = repo_full_name.replace("/", "_")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"github_{filename_safe}.json"   # ★
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"✅ GitHub JSON 파일 저장 완료: {output_path}")
    return result


# 테스트 실행
if __name__ == "__main__":
    info = github_fetcher("google-gemini/gemma-cookbook", branch="main")


# 사용 예시
if __name__ == "__main__":
    info = github_fetcher("google-gemini/gemma-cookbook", branch="main")
    import json
    # print(json.dumps(info, indent=2, ensure_ascii=False))


 # import json
    # print(json.dumps(result, indent=2, ensure_ascii=False))

    for key,values in info.items():
        print("*"*30)
        print(key)
        print(values)
