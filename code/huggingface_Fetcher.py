from typing import Dict, Any
import requests
import json
import os
from pathlib import Path

def huggingface_fetcher(model_id: str, save_to_file: bool = True, output_dir: str | Path = ".") -> Dict[str, Any]:
    base_api = f"https://huggingface.co/api/models/{model_id}?full=true"
    resp = requests.get(base_api)
    resp.raise_for_status()
    data = resp.json()

    # 1) siblings 목록 수집
    siblings = [f.get("rfilename", "") for f in data.get("siblings", [])]

    # helper: raw URL 생성
    def fetch_raw(filename: str) -> str:
        branches = ["main", "refs/convert", "refs/pr/1"]
        for branch in branches:
            url = f"https://huggingface.co/{model_id}/resolve/{branch}/{filename}"
            r = requests.get(url)
            if r.status_code == 200:
                return r.text
        return ""

    # 2) 주요 파일 콘텐츠 가져오기
    readme = fetch_raw("README.md") if "README.md" in siblings else ""
    config = fetch_raw("config.json") if "config.json" in siblings else ""
    generation_config = fetch_raw("generation_config.json") if "generation_config.json" in siblings else ""

    # LICENSE 파일 탐색
    license_candidates = [fn for fn in siblings if fn.upper().startswith("LICENSE")]
    license_file = ""
    if license_candidates:
        license_file = fetch_raw(license_candidates[0])

    # 3) .py 파일 모두 가져오기
    py_files = {}
    for fn in siblings:
        if fn.endswith(".py"):
            py_files[fn] = fetch_raw(fn)

    # 4) 결과 구성
    result = {
        "model_id": model_id,
        "files": siblings,
        "readme": readme,
        "config": config,
        "generation_config": generation_config,
        "license_file": license_file,
        "py_files": py_files
    }

    # 5) JSON 저장
    if save_to_file:
        filename_safe = model_id.replace("/", "_")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)         # ★ 폴더 생성
        output_path = output_dir / f"huggingface_{filename_safe}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"✅ JSON 파일 저장 완료: {output_path}")

    return result


# 단독 실행 시 테스트 코드
if __name__ == "__main__":
    test_model_id = "skt/A.X-4.0"
    huggingface_fetcher(test_model_id)


# 사용 
if __name__ == "__main__":
    result = huggingface_fetcher("deepseek-ai/DeepSeek-R1")
    # import json
    # print(json.dumps(result, indent=2, ensure_ascii=False))

    for key,values in result.items():
        print("*"*30)
        print(key)
        print(values)










    """
    Fetches file list and contents of key files for a given Hugging Face model.
    
    Returns a dict with:
    - model_id
    - files: list of sibling filenames
    - readme: content of README.md (or empty if 없음)
    - config: content of config.json (or empty if 없음)
    - license_file: content of LICENSE* files (or empty if 없음)
    - py_files: dict mapping each *.py filename to its content
    """