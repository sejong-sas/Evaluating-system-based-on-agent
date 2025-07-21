# github_disfeter.py

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# 1. 환경 변수에서 OPENAI_API_KEY 로드
load_dotenv()
_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise RuntimeError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
_client = OpenAI(api_key=_api_key)

# 2. 평가용 Prompt (16개 항목 정의 포함)
_EVAL_PROMPT = """
 당신은 GitHub 저장소에서 AI 모델 개방성 평가에 필요한 정보를 추출하는 전문가입니다.
    아래 16개 개방성 평가 항목과 관련된 정보모두!!!(하나도 빠짐 없이, 변형 없이) 골라 JSON으로 반환해주세요.
    오직 GitHub 페처가 만든 원본 JSON 데이터만 참고하고, 다른 자료는 절대 참조하지 마세요.

  {
  "1-1 (가중치 Weights)": "모델 가중치의 공개 여부, 위치, 접근 방식, 누구나 다운로드 가능한지에 관련된 모든 내용",
  "1-2 (코드 Code)": "모델 훈련 및 실행을 위한 코드가 공개되었는지, 어떤 부분이 공개되었는지에 관련된 모든 내용",
  "1-3 (라이선스 License)": "라이선스의 존재 여부, 종류, 허용된 권리(사용, 수정, 배포, 상업적 이용)에 관련된 모든 내용",
  "1-4 (논문 Paper)": "모델과 관련된 공식 논문, 기술 보고서, 블로그 등 문서의 존재와 링크에 관련된 모든 내용",
  "1-5 (아키텍처 Architecture)": "모델 아키텍처(레이어 수, 하이퍼파라미터 등)와 구조 설계의 세부 정보에 관련된 모든 내용",
  "1-6 (토크나이저 Tokenizer)": "어떤 토크나이저를 사용하는지, 이름과 구조, 다운로드 가능 여부에 관련된 모든 내용",

  "2-1 (하드웨어 Hardware)": "모델 훈련에 사용된 하드웨어 종류(H100, TPU 등), 수량, 계산 자원 규모에 관련된 모든 내용",
  "2-2 (소프트웨어 Software)": "훈련에 사용된 소프트웨어(프레임워크, 라이브러리 등)의 종류, 버전, 설정에 관련된 모든 내용",
  "2-3 (API)": "모델이 접근 가능한 API의 존재 여부, 문서 링크, 사용 예제, 공개 여부에 관련된 모든 내용",

  "3-1 (사전학습 Pre-training)": "사전학습(pre-training) 시 사용된 방법론, 절차, 데이터 흐름, 하이퍼파라미터 설정 등에 관련된 모든 내용",
  "3-2 (파인튜닝 Fine-tuning)": "파인튜닝 방식, 목적, 데이터 사용 여부, 재현 가능한 파이프라인 존재 여부에 관련된 모든 내용",
  "3-3 (강화학습 Reinforcement Learning)": "RLHF, DPO 등 강화학습 알고리즘 사용 여부, 구체적인 방식과 절차, 설정값 등에 관련된 모든 내용",

  "4-1 (사전학습 데이터 Pre-training Data)": "사전학습에 사용된 데이터의 종류, 수량, 출처, 사용 범위 및 구성 방식에 관련된 모든 내용",
  "4-2 (파인튜닝 데이터 Fine-tuning Data)": "파인튜닝에 사용된 데이터셋의 출처, 구성, 데이터 예시, 공개 여부 등에 관련된 모든 내용",
  "4-3 (강화학습 데이터 Reinforcement Learning Data)": "강화학습에 사용된 데이터셋의 구성, 접근 가능 여부, 출처, 생성 방식에 관련된 모든 내용",
  "4-4 (데이터 필터링 Data Filtering)": "데이터 필터링 또는 정제 방법, 사용된 기준, 필터링 과정과 그 영향에 관련된 모든 내용"
}
  
출력은 아래와 같은 형식의 JSON이어야 합니다:

{
  "repo": "org/repo",
  "1-2 (코드 Code)": {
    "files": ["train.py", "inference.py"],
    "readme": "README에서 '모델 구조 구현 코드 제공' 문장 발견"
  },
  "1-3 (라이선스 License)": {
    "license_files": ["LICENSE"],
    "license_type": "Apache-2.0"
  },
  ...
}

※ 점수(score)는 포함하지 말고, 설명이나 인용된 내용만 요약하세요.
※ 절대 항목 설명 텍스트를 복사하지 마세요.
※ 오직 GitHub JSON 내용 기반으로만 답변하세요.
"""

def filter_github_features(model_name: str, save: bool = True) -> dict:
    """
    GitHub 페처가 만든 github_<model_name>.json로부터
    GitHub 관련 정보만 추출하여 dict로 반환합니다.
    (필요 시 파일로도 저장)

    Args:
      model_name: "org/model" 형태
      save: 추출 결과를 github_filtered_<model>.json 으로 저장할지 여부

    Returns:
      평가된 GitHub 정보가 담긴 dict
    """
    base = model_name.replace("/", "_")
    in_path = f"github_{base}.json"
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"'{in_path}' 파일을 찾을 수 없습니다.")

    # 1) 원본 GitHub JSON 로드
    with open(in_path, "r", encoding="utf-8") as f:
        gh_data = json.load(f)

    # 2) GPT 요청
    gh_payload = json.dumps(gh_data, ensure_ascii=False)
    resp = _client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": _EVAL_PROMPT},
            {"role": "user",   "content": gh_payload}
        ],
        temperature=0
    )
    content = resp.choices[0].message.content.strip()

    # 3) 순수 JSON 파싱
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end   = content.rfind("}") + 1
        result = json.loads(content[start:end])

    # 4) 파일 저장
    if save:
        out_path = f"github_filtered_{base}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ GitHub 정보 필터링 결과 저장: {out_path}")

    return result
