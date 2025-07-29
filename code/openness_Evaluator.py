import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# 환경변수에서 API 키 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=api_key)

# 평가 기준 프롬프트
EVALUATION_PROMPT = """
당신은 오픈 소스 AI 모델의 개방성을 체계적으로 평가하는 전문가입니다.
요청받은 모델에 대해 다음 16개 세부 항목을 조사하고 각각 Open(1점), Semi-Open(0.5점), Closed(0점)으로 평가해주세요.

## 1. 모델 기본 개방성 (Model Basic Openness) - 6개 항목
### 1-1. 가중치 (Weights)
- Open(1점): 모델 가중치가 허가 없이 공개적으로 이용 가능
- Semi-Open(0.5점): 허가를 받은 후 모델 가중치 이용 가능
- Closed(0점): 모델 가중치가 공개되지 않아 사용 불가
### 1-2. 코드 (Code)
- Open(1점): 모델 훈련 및 구현에 사용된 전체 코드가 공개
- Semi-Open(0.5점): 모델 훈련 및 구현 코드의 일부만 공개
- Closed(0점): 훈련 및 구현 코드가 공개되지 않음
### 1-3. 라이선스 (License)
- Open(1점): 사용, 수정, 재배포, 상업적 이용에 제한 없음 (MIT, Apache 등)
- Semi-Open(0.5점): 사용, 수정, 재배포, 상업적 이용 중 1개 이상 제한
- Closed(0점): 3개 이상 제한 존재하거나 해당 라이선스 없음
### 1-4. 논문 (Paper)
- Open(1점): 공식 논문 또는 기술 보고서 존재
- Semi-Open(0.5점): 웹사이트 또는 블로그 포스트 존재
- Closed(0점): 관련 문서 없음
### 1-5. 아키텍처 (Architecture)
- Open(1점): 모델 구조와 하이퍼파라미터가 완전히 공개 (레이어 수, 하이퍼파라미터 등)
- Semi-Open(0.5점): 모델 구조만 공개 (예: Transformer 사용 언급)
- Closed(0점): 모델 구조 관련 정보 미공개
### 1-6. 토크나이저 (Tokenizer)
- Open(1점): 사용된 토크나이저 이름이 명시적으로 공개 (SentencePiece 등)
- Semi-Open(0.5점): 다운로드 및 사용 가능한 토크나이저 존재 (Hugging Face 등록)
- Closed(0점): 토크나이저 관련 정보 미공개 및 사용 불가

## 2. 접근성 및 재현성 (Accessibility and Reproducibility) - 3개 항목
### 2-1. 하드웨어 (Hardware)
- Open(1점): 모델 훈련에 필요한 하드웨어 종류와 수량 완전 공개 (예: 1920 x H100)
- Semi-Open(0.5점): 훈련에 필요한 하드웨어 종류만 공개 (H100, TPU 등)
- Closed(0점): 하드웨어 요구사항 미공개
### 2-2. 소프트웨어 (Software)
- Open(1점): 모델 훈련에 필요한 소프트웨어 사양 완전 공개
- Semi-Open(0.5점): 소프트웨어 사양 일부 공개 (프레임워크, 라이브러리 등)
- Closed(0점): 소프트웨어 사양 미공개
### 2-3. API
- Open(1점): 공개 API 존재
- Semi-Open(0.5점): 현재 비공개이나 향후 공개 예정
- Closed(0점): API 존재하지 않음

## 3. 훈련 방법론 개방성 (Training Methodology Openness) - 3개 항목
### 3-1. 사전학습 (Pre-training)
- Open(1점): 재현 가능할 정도로 훈련 과정 및 방법론 상세 공개
- Semi-Open(0.5점): 일부 훈련 방법만 언급 또는 설명
- Closed(0점): 훈련 방법론 미공개
### 3-2. 파인튜닝 (Fine-tuning)
- Open(1점): 파인튜닝 방법론 완전 공개
- Semi-Open(0.5점): 파인튜닝 방법 일부 공개
- Closed(0점): 파인튜닝 방법 미공개 (해당 없는 경우 N/A)
### 3-3. 강화학습 (Reinforcement Learning)
- Open(1점): RLHF, DPO 등 강화학습 방법론 완전 공개
- Semi-Open(0.5점): 강화학습 방법 일부 공개
- Closed(0점): 강화학습 방법 미공개 (해당 없는 경우 N/A)

## 4. 데이터 개방성 (Data Openness) - 4개 항목
### 4-1. 사전학습 데이터 (Pre-training Data)
- Open(1점): 훈련 데이터의 수량 및 출처 완전 공개
- Semi-Open(0.5점): 데이터 종류만 간략히 공개
- Closed(0점): 데이터 미공개
### 4-2. 파인튜닝 데이터 (Fine-tuning Data)
- Open(1점): 파인튜닝 데이터 완전 공개
- Semi-Open(0.5점): 데이터 일부 공개
- Closed(0점): 데이터 미공개 (해당 없는 경우 N/A)
### 4-3. 강화학습 데이터 (Reinforcement Learning Data)
- Open(1점): RL 데이터 완전 공개
- Semi-Open(0.5점): 데이터 일부 공개
- Closed(0점): 데이터 미공개 (해당 없는 경우 N/A)
### 4-4. 데이터 필터링 (Data Filtering)
- Open(1점): 데이터 필터링 방법론 및 내용 완전 공개
- Semi-Open(0.5점): 필터링 정보 일부 공개
- Closed(0점): 필터링 정보 미공개



출력 예시:
{
  "model": "org/model",
  "scores": {
    "1-1 가중치": 1,
    "1-2 코드": 0.5,
    ...
    "4-4 데이터 필터링": 0
  },
  "total_score": 12.5
}
❗️이 JSON **한 블록**만 반환하세요 — 다른 주석·백틱·설명 절대 금지.
"""

def evaluate_openness(model_name: str, hf_json: dict = None, gh_json: dict = None, arxiv_json: dict = None) -> dict:
    hf = hf_json or {}
    gh = gh_json or {}
    arxiv = arxiv_json or {}

    payload = {
        "model": model_name,
        "data": {
            "huggingface": hf,
            "github": gh,
            "arxiv": arxiv
        }
    }

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": EVALUATION_PROMPT},
            {"role": "user",   "content": json.dumps(payload, ensure_ascii=False)}
        ],
        temperature=0,
        response_format={"type": "json_object"}   # 👈 NEW!
    )

    raw = response.choices[0].message.content.strip()
    return json.loads(raw)   # response_format 보장 → try/except 필요 없어도 OK

    text = response.choices[0].message.content.strip()
    if not text:
        raise ValueError("❌ GPT 응답이 비어 있습니다.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == -1:
            raise ValueError("❌ GPT 응답에서 유효한 JSON 블록을 찾을 수 없습니다.")
        return json.loads(text[start:end])

def evaluate_openness_from_files(model_name: str) -> dict:
    base = model_name.replace("/", "_")

    def load(path: str) -> dict:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ JSON 파싱 실패: {path}")
        return {}

    hf = load(f"huggingface_filtered_{base.lower()}.json")
    gh = load(f"github_filtered_{base}.json")
    arxiv = load(f"arxiv_filtered_{base}.json")

    result = evaluate_openness(model_name, hf, gh, arxiv)

    out_path = f"openness_score_{base}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"📝 평가 결과 저장 완료: {out_path}")
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("사용법: python openness_Evaluator.py <org/model>")
        sys.exit(1)

    model_id = sys.argv[1]
    evaluate_openness_from_files(model_id)
