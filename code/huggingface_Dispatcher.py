# huggingface_Dispatcher.py
# High-Recall 2-Pass  +  evidence({source, quote})  →  긴 요약
# - evidence를 객체 배열로 저장
# - 요약은 quote들만 사용
# - __evidence_sources 필드 제거

import os
import json
import re
from typing import Dict, List, Any
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

# ──────────────────────────────── 환경설정 ────────────────────────────────
load_dotenv()
_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise RuntimeError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
_client = OpenAI(api_key=_api_key)

# ──────────────────────────── 16개 평가 항목 라벨 ─────────────────────────
LABELS = {
    "1-1": "1-1 (가중치 Weights)",
    "1-2": "1-2 (코드 Code)",
    "1-3": "1-3 (라이선스 License)",
    "1-4": "1-4 (논문 Paper)",
    "1-5": "1-5 (아키텍처 Architecture)",
    "1-6": "1-6 (토크나이저 Tokenizer)",
    "2-1": "2-1 (하드웨어 Hardware)",
    "2-2": "2-2 (소프트웨어 Software)",
    "2-3": "2-3 (API)",
    "3-1": "3-1 (사전학습 Pre-training)",
    "3-2": "3-2 (파인튜닝 Fine-tuning)",
    "3-3": "3-3 (강화학습 Reinforcement Learning)",
    "4-1": "4-1 (사전학습 데이터 Pre-training Data)",
    "4-2": "4-2 (파인튜닝 데이터 Fine-tuning Data)",
    "4-3": "4-3 (강화학습 데이터 Reinforcement Learning Data)",
    "4-4": "4-4 (데이터 필터링 Data Filtering)",
}

# ──────────────────────────── 항목별 상세 설명 ────────────────────────────
EVAL_DESCRIPTIONS = {
    LABELS["1-1"]: "모델 가중치의 공개 여부, 위치, 접근 방식, 누구나 다운로드 가능한지에 관련된 모든 내용",
    LABELS["1-2"]: "모델 훈련 및 실행을 위한 코드가 공개되었는지, 어떤 부분이 공개되었는지에 관련된 모든 내용",
    LABELS["1-3"]: "라이선스의 존재 여부, 종류, 허용된 권리(사용, 수정, 배포, 상업적 이용)에 관련된 모든 내용",
    LABELS["1-4"]: "모델과 관련된 공식 논문, 기술 보고서, 블로그 등 문서의 존재와 링크에 관련된 모든 내용",
    LABELS["1-5"]: "모델 아키텍처(레이어 수, 하이퍼파라미터 등)와 구조 설계의 세부 정보에 관련된 모든 내용",
    LABELS["1-6"]: "어떤 토크나이저를 사용하는지, 이름과 구조, 다운로드 가능 여부에 관련된 모든 내용",
    LABELS["2-1"]: "모델 훈련에 사용된 하드웨어 종류(H100, TPU 등), 수량, 계산 자원 규모에 관련된 모든 내용",
    LABELS["2-2"]: "훈련에 사용된 소프트웨어(프레임워크, 라이브러리 등)의 종류, 버전, 설정에 관련된 모든 내용",
    LABELS["2-3"]: "모델이 접근 가능한 API의 존재 여부, 문서 링크, 사용 예제, 공개 여부에 관련된 모든 내용",
    LABELS["3-1"]: "사전학습 시 사용된 방법론, 절차, 데이터 흐름, 하이퍼파라미터 설정 등에 관련된 모든 내용",
    LABELS["3-2"]: "파인튜닝 방식, 목적, 데이터 사용 여부, 재현 가능한 파이프라인 존재 여부에 관련된 모든 내용",
    LABELS["3-3"]: "RLHF, DPO 등 강화학습 알고리즘 사용 여부, 구체적인 방식과 절차, 설정값 등에 관련된 모든 내용",
    LABELS["4-1"]: "사전학습에 사용된 데이터의 종류, 수량, 출처, 사용 범위 및 구성 방식에 관련된 모든 내용",
    LABELS["4-2"]: "파인튜닝에 사용된 데이터셋의 출처, 구성, 데이터 예시, 공개 여부 등에 관련된 모든 내용",
    LABELS["4-3"]: "강화학습에 사용된 데이터셋의 구성, 접근 가능 여부, 출처, 생성 방식에 관련된 모든 내용",
    LABELS["4-4"]: "데이터 필터링 또는 정제 방법, 사용된 기준, 필터링 과정과 그 영향에 관련된 모든 내용",
}

# ─────────────────────────────── 그룹 분할 ───────────────────────────────
ITEM_GROUPS: List[List[str]] = [
    ["1-1", "1-2", "1-3", "1-4"],
    ["1-5", "1-6", "2-1", "2-2"],
    ["2-3", "3-1", "3-2", "3-3"],
    ["4-1", "4-2", "4-3", "4-4"],
]

# ──────────────────────────── 하이퍼파라미터 ────────────────────────────
CHUNK_CHARS = 60_000
CHUNK_OVERLAP = 2_000
EVIDENCE_LIMIT_PER_KEY = 300
MODEL_NAME = os.getenv("OPENAI_MODEL_HF_DISPATCHER", "o3-mini")

# (참고) 사용하지는 않지만, 필요 시 태그 검증용
_SRC_TAG_RE = re.compile(r'^\s*\[([^\]]+)\]\s*.*$')

# ─────────────────────────────── 유틸 함수 ───────────────────────────────
def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def _chunk_text(s: str, chunk: int, overlap: int) -> List[str]:
    out, n, i = [], len(s), 0
    while i < n:
        end = min(i + chunk, n)
        out.append(s[i:end])
        if end == n:
            break
        i = end - overlap if end - overlap > i else end
    return out

def _dedup_evidences(evs: List[Dict[str, str]], limit: int) -> List[Dict[str, str]]:
    seen, out = set(), []
    for ev in evs:
        if not (isinstance(ev, dict)
                and isinstance(ev.get("source"), str)
                and isinstance(ev.get("quote"), str)):
            continue
        src = ev["source"].strip()
        qt  = ev["quote"].strip()
        if not src or not qt:
            continue
        key = (src, qt)
        if key in seen:
            continue
        seen.add(key)
        out.append({"source": src, "quote": qt})
        if len(out) >= limit:
            break
    return out

def _group_desc_map(ids: List[str]) -> Dict[str, str]:
    return {LABELS[i]: EVAL_DESCRIPTIONS[LABELS[i]] for i in ids}

# ─────────────────────────────── 프롬프트 ───────────────────────────────
_BASE_RECALL_SYS = """
당신은 Hugging Face 저장소에서 AI 모델 개방성 평가 정보를 추출하는 전문가입니다.
오직 제공된 payload(원문)만 사용하세요.
각 항목마다 evidence를 '객체 배열'로 반환하세요.
각 evidence 객체는 반드시 다음 필드를 가집니다:
- "source": payload 섹션 태그(예: "readme", "files", "py_files/파일명.py")
- "quote" : 해당 섹션에서 그대로 복사한 문장(수정·요약 금지)
근거가 없으면 빈 배열 [] 로 반환하세요.
반드시 JSON 객체만 반환하세요.
""".strip()

_BASE_SUMMARY_SYS = """
당신은 evidence의 quote만 사용해 각 항목을 길고 자세히 요약하는 전문가입니다.
반드시 JSON 객체만 반환하세요(텍스트 추가 금지).
""".strip()

def _build_recall_inst(group: List[str]) -> str:
    desc = _json(_group_desc_map(group))
    example = _json({
        LABELS[k]: [
            {"source": "readme", "quote": "원문 문장 예시 1"},
            {"source": "py_files/train.py", "quote": "원문 문장 예시 2"}
        ] for k in group
    })
    return (
        f"이번 그룹 항목 정의:\n{desc}\n"
        "각 키에 대해 evidence 객체 배열을 반환하세요. 예시 스키마:\n"
        f"{example}"
    )

def _build_summary_inst(group: List[str]) -> str:
    desc = _json(_group_desc_map(group))
    return (
        f"이번 그룹 항목 정의:\n{desc}\n"
        "이후 quote 배열이 주어집니다. 각 항목을 길게 요약하세요."
    )

# ─────────────────────────────── GPT 호출 ───────────────────────────────
def _chat_json(system: str, user: str) -> Dict[str, Any]:
    resp = _client.chat.completions.create(
        model=MODEL_NAME,
        reasoning_effort="medium",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    try:
        return json.loads(resp.choices[0].message.content.strip())
    except Exception:
        return {}

# ───────────────────────────── 페이로드 빌더 ─────────────────────────────
def _make_group_payload(hf: Dict, idx: int) -> Dict:
    py_src = hf.get("py_files", {}) or {}
    py_items = list(py_src.items())[:20]         # 최대 20개 파일
    py_files = {fn: (src[:20_000] if isinstance(src, str) else "")
                for fn, src in py_items}
    return {
        "model_id":          hf.get("model_id", ""),
        "files":             hf.get("files", [])[:2000],
        "readme":            hf.get("readme", ""),
        "license_file":      hf.get("license_file", ""),
        "config":            hf.get("config", ""),
        "generation_config": hf.get("generation_config", ""),
        "py_files":          py_files,
    }

def _payload_to_text(p: Dict) -> str:
    parts = [
        f"[model_id]\n{p.get('model_id','')}\n",
        f"[readme]\n{p.get('readme','')}\n",
        f"[license_file]\n{p.get('license_file','')}\n",
        f"[config]\n{p.get('config','')}\n",
        f"[generation_config]\n{p.get('generation_config','')}\n",
    ]
    for fn, code in p.get("py_files", {}).items():
        parts.append(f"[py_files/{fn}]\n{code}\n")
    if p.get("files"):
        parts.append("[files]\n" + "\n".join(map(str, p["files"])) + "\n")
    return "\n".join(parts)

# ───────────────────────────── evidence 수집 ─────────────────────────────
_ALLOWED_PREFIX = ("model_id", "readme", "license_file", "config",
                   "generation_config", "files", "py_files/")

def _is_valid_source(src: str) -> bool:
    return isinstance(src, str) and src.startswith(_ALLOWED_PREFIX)

def _recall_collect(group: List[str], text: str) -> Dict[str, List[Dict[str, str]]]:
    chunks = _chunk_text(text, CHUNK_CHARS, CHUNK_OVERLAP)
    out: Dict[str, List[Dict[str, str]]] = {LABELS[k]: [] for k in group}

    for chunk in chunks:
        ans = _chat_json(_BASE_RECALL_SYS, _build_recall_inst(group) +
                         "\n=== PAYLOAD ===\n" + chunk)

        for k in group:
            lbl = LABELS[k]
            evs = ans.get(lbl, [])
            if not isinstance(evs, list):
                continue
            # 타입 검증 + 중복 제거
            out[lbl].extend(_dedup_evidences(evs, EVIDENCE_LIMIT_PER_KEY))

    return out

# ─────────────────────────────── 요약 생성 ──────────────────────────────
def _summarize(group: List[str], evid: Dict[str, List[Dict[str, str]]]) -> Dict[str, str]:
    quotes = {
        LABELS[k]: [e["quote"] for e in evid[LABELS[k]]]
        for k in group
    }
    ans = _chat_json(_BASE_SUMMARY_SYS, _build_summary_inst(group) +
                     "\n=== EVIDENCE_QUOTES ===\n" + _json(quotes))
    return {LABELS[k]: ans.get(LABELS[k], "") for k in group}

# ─────────────────────────────── 병합 유틸 ───────────────────────────────
def _merge_for_final(summary: Dict[str, str],
                     evid: Dict[str, List[Dict[str, str]]]) -> Dict[str, Any]:
    final = {}
    for lbl, txt in summary.items():
        final[lbl] = txt.strip()
        final[f"{lbl}__evidence"] = evid.get(lbl, [])
    return final

def _merge_dicts(ds: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged = {}
    for d in ds:
        merged.update(d)
    return merged

# ────────────────────────────── 메인 함수 ────────────────────────────────
def filter_hf_features(model: str, save: bool = True, output_dir: str | Path = ".") -> Dict[str, Any]:
    base = model.replace("/", "_").lower()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 입력 JSON: 우선 output_dir에서 찾고, 없으면 루트에서 폴백
    path_in = output_dir / f"huggingface_{base}.json"
    if not path_in.exists():
        alt = Path(f"huggingface_{base}.json")
        if not alt.exists():
            raise FileNotFoundError(str(path_in))
        hf = json.load(open(alt, encoding="utf-8"))
    else:
        hf = json.load(open(path_in, encoding="utf-8"))

    parts = []
    for idx, grp in enumerate(ITEM_GROUPS, 1):
        try:
            payload = _make_group_payload(hf, idx - 1)
            text = _payload_to_text(payload)
            evid = _recall_collect(grp, text)
            summ = _summarize(grp, evid)
            part = _merge_for_final(summ, evid)
        except Exception as e:
            print(f"⚠️ 그룹 {idx} 처리 오류:", e)
            part = {}

        if save:
            out_path = output_dir / f"huggingface_filtered_{base}_{idx}.json"
            json.dump(part, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            print(f"✅ 그룹 {idx} 결과 저장:", out_path)
        parts.append(part)

    merged = _merge_dicts(parts)
    if save:
        out_merged = output_dir / f"huggingface_filtered_final_{base}.json"
        json.dump(merged, open(out_merged, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("✅ 최종 병합 결과 저장:", out_merged)
    return merged

# ─────────────────────────────── CLI 진입점 ──────────────────────────────
if __name__ == "__main__":
    import sys
    model_id = "bigscience/bloomz-560m"
    outdir = "."

    # 사용법: python huggingface_Dispatcher.py <org/model> [output_dir]
    if len(sys.argv) >= 2 and sys.argv[1]:
        model_id = sys.argv[1]
    if len(sys.argv) >= 3 and sys.argv[2]:
        outdir = sys.argv[2]

    print("▶ 실행 모델:", model_id)
    print("▶ 출력 폴더:", outdir)
    filter_hf_features(model_id, output_dir=outdir)