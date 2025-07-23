# Evaluating-system-based-on-agent
Follow-up project of paper(Evaluating the Openness of Impactful AI Models with a Focus on LLMs)

##  시스템 구조

```mermaid
graph TD
    A[사용자 입력 (URL 또는 org/model)] --> B[모델 식별 (model_Identifier.py)]

    B --> C1[Hugging Face 수집 (huggingface_Fatcher.py)]
    C1 --> C2[arXiv 논문 수집 (arxiv_Fetcher.py)]

    C1 --> D1[HF 정보 필터링 (huggingface_Dispatcher.py)]
    C2 --> D2[arXiv 필터링 (arxiv_Dispatcher.py)]

    B --> E1[GitHub 수집 (github_Fatcher.py)]
    E1 --> E2[GitHub 필터링 (github_Dispatcher.py)]

    D1 --> F[개방성 평가 (openness_Evaluator.py)]
    D2 --> F
    E2 --> F

    F --> G[개방성 점수 저장]
    B --> H[모델 추론 (inference.py)]
```

## 구성 모듈
| 모듈                          | 설명                                                   |
| --------------------------- | ---------------------------------------------------- |
| `model_Identifier.py`       | 전체 파이프라인 실행, 플랫폼 연결 및 예외 처리                          |
| `huggingface_Fatcher.py`    | Hugging Face 모델 정보 수집 (`README`, `LICENSE`, `.py` 등) |
| `github_Fatcher.py`         | GitHub 저장소 정보 수집                                     |
| `arxiv_Fetcher.py`          | Hugging Face 태그 기반 arXiv 논문 수집 및 텍스트 추출              |
| `huggingface_Dispatcher.py` | Hugging Face 정보 필터링 (GPT-4o 사용)                      |
| `github_Dispatcher.py`      | GitHub 정보 필터링 (GPT-4o 사용)                            |
| `arxiv_Dispatcher.py`       | arXiv 논문 텍스트 필터링 (GPT-3.5 사용)                        |
| `openness_Evaluator.py`     | 16개 항목에 대해 점수화 및 최종 JSON 출력                          |
| `inference.py`              | 모델 추론 테스트 (Transformers 기반)                          |

## 출력파일
| 파일명                                 | 설명                   |
| ----------------------------------- | -------------------- |
| `huggingface_<model>.json`          | Hugging Face 원본 데이터  |
| `huggingface_filtered_<model>.json` | Hugging Face 필터링 결과  |
| `github_<model>.json`               | GitHub 원본 데이터        |
| `github_filtered_<model>.json`      | GitHub 필터링 결과        |
| `arxiv_fulltext_<model>.json`       | arXiv 논문 전체 텍스트      |
| `arxiv_filtered_<model>.json`       | arXiv 필터링 결과         |
| `openness_score_<model>.json`       | 최종 개방성 평가 결과 (점수 포함) |



## 현재 상황
### 문제점
- [ ] 3순위인 openai api를 이용한 hf->git, git->hf id 알아내기
- [ ] _Dispatcher.py에서 기존의 Fetcher에서 가저온 정보를 16가지 평가 항목에 관련된 내용을 뽑아 요약을 해야 하는데 output이 gpt가 너무 요약을 해서 한줄로 밖에 안나옴
- [ ] openness_Evaluator.py 가 신뢰성이 없음

### 목표
