# huggingface_disfeter.py

import os, json, re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise RuntimeError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
_client = OpenAI(api_key=_api_key)

_EVAL_PROMPT = ("""
 당신은 Hugging Face 저장소에서 AI 모델 개방성 평가에 필요한 정보를 추출하는 전문가입니다.
    아래 16개 개방성 평가 항목과 관련된 정보모두!!!(하나도 빠짐 없이, 변형 없이) 골라 JSON으로 반환해주세요.
    오직 Hugging Face 페처가 만든 원본 JSON 데이터만 참고하고, 다른 자료는 절대 참조하지 마세요.

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
오직 Hugging Face 원본 JSON만 참고하고, 외부 자료는 절대 참조하지 마세요.
""")
_KEYWORDS = re.compile(
    r"(license|weights|tokenizer|architecture|hardware|dataset|api|training|fine[- ]?tuning|reinforcement)",
    re.IGNORECASE,
)
_MAX_CHARS = 20_000  # ≈ 8~10k 토큰: gpt-4o 16k 한도 대비 안전
#################################################################################
def _preprocess(raw: str) -> str:
    """관심 키워드가 포함된 줄만 추린 뒤 길이 제한."""
    lines = [ln for ln in raw.splitlines() if _KEYWORDS.search(ln)]
    text = "\n".join(lines)
    return text[:_MAX_CHARS]




def filter_hf_features(model_name: str, save: bool = True) -> dict:
    base = model_name.replace("/", "_")
    in_path = f"huggingface_{base}.json"
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"'{in_path}' 파일을 찾을 수 없습니다.")

    with open(in_path, "r", encoding="utf-8") as f:
        hf_data = json.load(f)

    payload = json.dumps(hf_data, ensure_ascii=False)
    resp = _client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": _EVAL_PROMPT},
            {"role": "user", "content": payload}
        ],
        temperature=0
    )
    content = resp.choices[0].message.content.strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        result = json.loads(content[start:end])

    if save:
        out_path = f"huggingface_filtered_{base}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ Hugging Face 정보 필터링 결과 저장: {out_path}")

    return result
