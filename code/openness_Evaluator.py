# openness_Evaluator.py
# ─────────────────────────────────────────────────────────
# • Hugging Face 모델 존재 시 자동 1점: 1-1, 1-5, 1-6
# • 나머지 13개 항목은 GPT 평가
# • 훈련 방법론(3-1~3-3)은 arxiv_Dispatcher JSON을 최우선 참고
# • GPT 응답 스키마 강제: {"scores": {...}, "total_score": ...}
# ─────────────────────────────────────────────────────────

import os, json
from typing import Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

load_dotenv()
_API_KEY = os.getenv("OPENAI_API_KEY")
if not _API_KEY:
    raise RuntimeError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
client = OpenAI(api_key=_API_KEY)

# ─────────── 평가 기준 전문 ───────────
CRITERIA_TEXT = """
## 1. 모델 기본 개방성 (Model Basic Openness) - 6개 항목
### 1-1. 가중치 (Weights) - 만약 허깅페이스에 모델이 올라와 있다면 무조건 Open
- Open(1점): 모델 가중치가 허가 없이 공개적으로 이용 가능
- Semi-Open(0.5점): 허가를 받은 후 모델 가중치 이용 가능
- Closed(0점): 모델 가중치가 공개되지 않아 사용 불가
### 1-2. 코드 (Code) - 만약 허깅페이스에 .py 파일이 있으면 무조건 Open
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
### 1-5. 아키텍처 (Architecture) - 만약 허깅페이스에 모델이 올라와 있다면 무조건 Open
- Open(1점): 모델 구조와 하이퍼파라미터가 완전히 공개
- Semi-Open(0.5점): 모델 구조만 공개
- Closed(0점): 모델 구조 정보 미공개
### 1-6. 토크나이저 (Tokenizer) - 만약 허깅페이스에 모델이 올라와 있다면 무조건 Open
- Open(1점): 사용된 토크나이저가 명시적으로 공개
- Semi-Open(0.5점): 다운로드 가능한 토크나이저 존재
- Closed(0점): 토크나이저 정보 미공개

## 2. 접근성 및 재현성 (Accessibility and Reproducibility) - 3개 항목
### 2-1. 하드웨어 (Hardware)
- Open(1점): 훈련 하드웨어 종류·수량 완전 공개
- Semi-Open(0.5점): 하드웨어 종류만 공개
- Closed(0점): 하드웨어 정보 미공개
### 2-2. 소프트웨어 (Software)
- Open(1점): 훈련에 필요한 소프트웨어 사양 완전 공개
- Semi-Open(0.5점): 일부만 공개
- Closed(0점): 정보 미공개
### 2-3. API
- Open(1점): 공개 API 존재
- Semi-Open(0.5점): 향후 공개 예정
- Closed(0점): API 없음

## 3. 훈련 방법론 개방성 (Training Methodology Openness) - 3개 항목
### 3-1. 사전학습 (Pre-training)
- Open(1점): 재현 가능 수준의 상세 공개
- Semi-Open(0.5점): 일부 방법만 언급
- Closed(0점): 방법 미공개
### 3-2. 파인튜닝 (Fine-tuning)
- Open(1점): 방법론 완전 공개
- Semi-Open(0.5점): 일부 공개
- Closed(0점): 미공개/N/A
### 3-3. 강화학습 (Reinforcement Learning)
- Open(1점): RLHF, DPO 등 상세 공개
- Semi-Open(0.5점): 일부 공개
- Closed(0점): 미공개/N/A

## 4. 데이터 개방성 (Data Openness) - 4개 항목
### 4-1. 사전학습 데이터 (Pre-training Data)
- Open(1점): 수량·출처 완전 공개
- Semi-Open(0.5점): 종류만 공개
- Closed(0점): 미공개
### 4-2. 파인튜닝 데이터 (Fine-tuning Data)
- Open(1점): 데이터 완전 공개
- Semi-Open(0.5점): 일부 공개
- Closed(0점): 미공개/N/A
### 4-3. 강화학습 데이터 (Reinforcement Learning Data)
- Open(1점): 데이터 완전 공개
- Semi-Open(0.5점): 일부 공개
- Closed(0점): 미공개/N/A
### 4-4. 데이터 필터링 (Data Filtering)
- Open(1점): 필터링 방법론·내용 완전 공개
- Semi-Open(0.5점): 일부 공개
- Closed(0점): 미공개
""".strip()

# ─────────── GPT 시스템 프롬프트 ───────────
EVALUATION_PROMPT = f"""
{CRITERIA_TEXT}

❗️훈련 방법론 개방성(3-1 ~ 3-3)은 arxiv_Dispatcher가 만든 JSON(논문 정보)을 **가장 우선** 참고하세요.
HuggingFace·GitHub 정보는 보조 참고만 허용됩니다.

또한 Hugging Face에 모델이 존재하므로 **다음 세 항목은 이미 Open(1점)** 입니다.
  • 1-1 Weights • 1-5 Architecture • 1-6 Tokenizer
→ 이 세 항목은 scores에 넣지 마세요.

반드시 아래 스키마처럼 단일 JSON 블록을 반환하십시오:

{{
  "scores": {{
    "1-2 코드": {{ "score": 1,   "reason": "..." }},
    ...
  }},
  "total_score": 12.5
}}
다른 주석·백틱·불필요 텍스트를 포함하면 안 됩니다.
""".strip()

# ─────────── 자동 1점 항목 ───────────
AUTO_OPEN_LABELS = {
    "1-1 가중치":   "허깅페이스에 모델 가중치 공개",
    "1-5 아키텍처": "허깅페이스 카드에 아키텍처 정보 공개",
    "1-6 토크나이저": "허깅페이스 카드/config에 토크나이저 정보 공개",
}

def _auto_scores(hf_json: Dict[str, Any]) -> Dict[str, Dict]:
    return {lbl: {"score": 1, "reason": reason}
            for lbl, reason in AUTO_OPEN_LABELS.items()} if hf_json else {}

# ─────────── GPT 평가 함수 ───────────
def _gpt_evaluate(model: str,
                  hf: Dict, gh: Dict, ax: Dict) -> Dict[str, Dict]:
    payload = {
        "model": model,
        "data": {"huggingface": hf, "github": gh, "arxiv": ax}
    }
    rsp = client.chat.completions.create(
        model="o3-mini",
        reasoning_effort="medium",
        response_format={"type":"json_object"},
        messages=[
            {"role":"system","content":EVALUATION_PROMPT},
            {"role":"user",  "content":json.dumps(payload, ensure_ascii=False)}
        ]
    )
    raw = json.loads(rsp.choices[0].message.content.strip())
    scores_dict = raw.get("scores", raw)      # 유연 파싱
    out = {}
    for k, v in scores_dict.items():
        if isinstance(v, dict):
            out[k] = {"score": v.get("score", 0), "reason": v.get("reason","")}
        elif isinstance(v, (int, float)):
            out[k] = {"score": v, "reason": ""}
    return out

# ─────────── 메인 평가 함수 ───────────
def evaluate_openness(model_name: str,
                      hf_json=None, gh_json=None, arxiv_json=None) -> Dict:
    hf, gh, ax = hf_json or {}, gh_json or {}, arxiv_json or {}

    scores = _gpt_evaluate(model_name, hf, gh, ax)
    scores.update(_auto_scores(hf))           # 자동 1점 항목 추가/덮어쓰기

    total = sum(v["score"] for v in scores.values())
    return {"model": model_name, "scores": scores, "total_score": total}

# ─────────── 파일 로더 & CLI ───────────
def _load(p):
    if os.path.exists(p) and os.path.getsize(p):
        try:
            return json.load(open(p,encoding="utf-8"))
        except json.JSONDecodeError:
            print("⚠️ JSON 파싱 실패:", p)
    return {}

def evaluate_openness_from_files(model_name: str, base_dir: str | Path = "."):
    base = model_name.replace("/", "_").lower()
    base_dir = Path(base_dir)

    # 폴더 우선, 없으면 루트 폴백
    def _load_from_base(filename: str):
        p = base_dir / filename
        if p.exists() and p.stat().st_size:
            try:
                return json.load(open(p, encoding="utf-8"))
            except json.JSONDecodeError:
                print("⚠️ JSON 파싱 실패:", p)
        # 루트 폴백
        if os.path.exists(filename) and os.path.getsize(filename):
            try:
                return json.load(open(filename, encoding="utf-8"))
            except json.JSONDecodeError:
                print("⚠️ JSON 파싱 실패:", filename)
        return {}

    hf = _load_from_base(f"huggingface_filtered_final_{base}.json")
    gh = _load_from_base(f"github_filtered_final_{base}.json")
    ax = _load_from_base(f"arxiv_filtered_final_{base}.json")

    res = evaluate_openness(model_name, hf, gh, ax)
    out = base_dir / f"openness_score_{base}.json"
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("📝 평가 결과 저장:", out)
    return res


if __name__ == "__main__":
    evaluate_openness_from_files("bigscience/bloomz-560m")
